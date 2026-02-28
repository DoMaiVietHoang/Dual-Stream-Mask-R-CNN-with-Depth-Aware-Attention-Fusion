"""
Training Script for Dual-Stream Mask R-CNN with DAAF
Includes boundary loss for better crown separation
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from tqdm import tqdm
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model
from data import create_dataloader
from modules.losses import BoundaryLoss, TreeCrownSegmentationLoss
from config import Config


class ModelEMA:
    """Exponential Moving Average of model weights for stable evaluation.

    Updates both parameters AND buffer (BatchNorm running stats) so that
    the EMA model produces valid outputs in eval mode.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        import copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            # Update parameters (weights, biases)
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
            # Update buffers (BatchNorm running_mean, running_var, num_batches_tracked)
            for ema_buf, model_buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.data.copy_(model_buf.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 4,
    ema: Optional['ModelEMA'] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation

    Args:
        model: DualStreamMaskRCNN model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        accumulation_steps: Number of batches to accumulate before optimizer step
        ema: ModelEMA instance — updated every optimizer step
        logger: Logger instance

    Returns:
        Dictionary of average losses
    """
    model.train()

    total_losses = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_mask': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0,
        'loss_boundary': 0.0,
        'loss_total': 0.0
    }

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]

        # Extract pre-computed depth maps from targets if available
        depth_maps = None
        if len(targets) > 0 and 'depth' in targets[0]:
            depth_maps = torch.stack([t.pop('depth') for t in targets])

        # Forward pass
        loss_dict = model(images, targets, depth_maps=depth_maps)

        # Sum losses and scale by accumulation steps
        losses = sum(loss for loss in loss_dict.values())
        scaled_loss = losses / accumulation_steps
        scaled_loss.backward()

        # Optimizer step every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Update EMA every optimizer step (not just per epoch)
            if ema is not None:
                ema.update(model)

        # Accumulate unscaled losses for logging
        for key in loss_dict:
            if key in total_losses:
                total_losses[key] += loss_dict[key].item()
        total_losses['loss_total'] += losses.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls': f'{loss_dict.get("loss_classifier", 0):.4f}',
            'mask': f'{loss_dict.get("loss_mask", 0):.4f}',
            'bound': f'{loss_dict.get("loss_boundary", 0):.4f}'
        })

        # Clear CUDA cache periodically
        if batch_idx % 100 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

    # Average losses
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    if logger:
        logger.info(f'Epoch {epoch} - Average losses:')
        for key, value in avg_losses.items():
            logger.info(f'  {key}: {value:.4f}')

    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    Calculates P/R/F1 (box-based) and mask AP metrics (AP, AP50, AP70, AP75).

    Memory-efficient design: mask IoU is computed per-batch and immediately
    discarded. Only lightweight scalars (scores, iou_matrix rows) are kept,
    so RAM usage is O(total_predictions) not O(total_pixels).

    Args:
        max_batches: Maximum number of batches to evaluate (None = all batches)
    """
    model.eval()

    # Box-based P/R/F1 — incremental counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_instances = 0

    # For mask AP: store only (score, iou_row) per prediction — no pixel data kept
    # iou_row[j] = mask IoU of this prediction against GT mask j of its image
    # Structure: list of {'scores': [N], 'iou_rows': [N, M], 'num_gt': int}
    ap_records = []   # one entry per image processed
    total_gt_masks = 0
    num_batches_processed = 0

    pbar = tqdm(dataloader, desc='Evaluating')

    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            if logger:
                logger.info(f'Stopped evaluation after {max_batches} batches')
            break

        images = images.to(device)

        depth_maps = None
        if len(targets) > 0 and 'depth' in targets[0]:
            depth_maps = torch.stack([t['depth'] for t in targets]).to(device)

        try:
            predictions = model(images, depth_maps=depth_maps)
        except Exception as e:
            if logger:
                logger.error(f'Error in batch {batch_idx}: {e}')
            torch.cuda.empty_cache()
            continue

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # --- GT masks (reference size) ---
            gt_masks_raw = target.get('masks', None)
            if gt_masks_raw is not None and isinstance(gt_masks_raw, torch.Tensor) and len(gt_masks_raw) > 0:
                gt_masks = gt_masks_raw.bool().cpu()               # [M, H_gt, W_gt] bool
            else:
                gt_masks = torch.zeros((0,), dtype=torch.bool)

            num_gt = len(gt_masks)
            total_gt_masks += num_gt

            # --- Predicted masks ---
            # postprocess() resized masks to original_image_sizes which may differ
            # from GT mask size (1024×1024 from dataset). Resize pred masks to match GT.
            pred_masks_raw = pred.get('masks', None)
            pred_scores = pred['scores'].detach().cpu()            # [N]

            # Filter low-confidence predictions to reduce FP noise in AP calc
            # COCO uses maxDets=100; we also apply a min score threshold
            score_filter = pred_scores > 0.05
            if score_filter.any():
                pred_scores = pred_scores[score_filter]
                if pred_masks_raw is not None and len(pred_masks_raw) > 0:
                    pred_masks_raw = pred_masks_raw[score_filter]
                # Keep top-100 by score (COCO maxDets convention)
                if len(pred_scores) > 100:
                    topk_idx = pred_scores.argsort(descending=True)[:100]
                    pred_scores = pred_scores[topk_idx]
                    if pred_masks_raw is not None and len(pred_masks_raw) > 0:
                        pred_masks_raw = pred_masks_raw[topk_idx]
            else:
                pred_scores = torch.zeros(0)
                pred_masks_raw = None

            if pred_masks_raw is not None and len(pred_masks_raw) > 0:
                # pred_masks_raw is already bool after postprocess (> 0.5 thresholded)
                # Convert to float for interpolation
                pred_masks_float = pred_masks_raw[:, 0].float().cpu()  # [N, H_pred, W_pred]

                if num_gt > 0:
                    gt_h, gt_w = gt_masks.shape[-2], gt_masks.shape[-1]
                    pred_h, pred_w = pred_masks_float.shape[-2], pred_masks_float.shape[-1]

                    if (pred_h, pred_w) != (gt_h, gt_w):
                        # Resize pred masks to match GT spatial size.
                        # This happens when original_image_sizes != image_size
                        # (postprocess upsizes to orig, GT is always image_size).
                        if batch_idx == 0 and i == 0 and logger:
                            logger.info(
                                f'[eval] Resizing pred masks {(pred_h, pred_w)} -> '
                                f'{(gt_h, gt_w)} to match GT'
                            )
                        pred_masks_float = F.interpolate(
                            pred_masks_float.unsqueeze(0),  # [1, N, H, W]
                            size=(gt_h, gt_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # [N, gt_h, gt_w]

                pred_masks = pred_masks_float > 0.5   # [N, H_gt, W_gt] bool
                del pred_masks_float
            else:
                pred_masks = torch.zeros((0,), dtype=torch.bool)

            # Compute mask IoU matrix [N, M] on GPU for speed
            if len(pred_masks) > 0 and num_gt > 0:
                iou_matrix = compute_mask_iou(pred_masks, gt_masks, device=device)
            else:
                iou_matrix = torch.zeros((len(pred_scores), num_gt), dtype=torch.float32)

            # Debug: log first batch stats to diagnose AP issues
            if batch_idx == 0 and i == 0 and logger:
                n_pred = len(pred_scores)
                score_str = (f'[{pred_scores.min().item():.3f}, {pred_scores.max().item():.3f}]'
                             if n_pred > 0 else 'N/A')
                iou_str = (f'shape={tuple(iou_matrix.shape)}, '
                           f'max={iou_matrix.max().item():.4f}, '
                           f'mean={iou_matrix.mean().item():.4f}'
                           if iou_matrix.numel() > 0 else 'empty')
                logger.info(
                    f'[eval-debug] batch0/img0: '
                    f'num_pred={n_pred}, num_gt={num_gt}, '
                    f'scores={score_str}, iou_matrix={iou_str}'
                )

            # Store only scalars: scores [N] and iou_matrix [N, M]
            ap_records.append({
                'scores': pred_scores,       # [N]
                'iou_rows': iou_matrix,      # [N, M]
                'num_gt': num_gt,
            })

            # Free pixel tensors immediately
            del pred_masks, gt_masks
            if pred_masks_raw is not None:
                del pred_masks_raw

        # Box-based P/R/F1
        pred_for_metrics = []
        for pred in predictions:
            pred_for_metrics.append({
                'boxes': pred['boxes'].detach().cpu(),
                'scores': pred['scores'].detach().cpu(),
            })
        batch_metrics = calculate_metrics_batch(pred_for_metrics, targets)
        total_tp += batch_metrics['tp']
        total_fp += batch_metrics['fp']
        total_fn += batch_metrics['fn']
        total_iou += batch_metrics['total_iou']
        num_instances += batch_metrics['num_instances']
        num_batches_processed += 1

        del predictions
        if batch_idx % 20 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

        current_precision = total_tp / (total_tp + total_fp + 1e-8)
        current_recall = total_tp / (total_tp + total_fn + 1e-8)
        pbar.set_postfix({
            'batch': f'{batch_idx+1}/{len(dataloader)}',
            'P': f'{current_precision:.3f}',
            'R': f'{current_recall:.3f}'
        })

    if logger:
        logger.info(f'Calculating mask AP from {num_batches_processed} batches '
                    f'({total_gt_masks} GT masks total)...')
        # Debug: summarize ap_records to diagnose AP=0.0099 issue
        total_preds = sum(len(r['scores']) for r in ap_records)
        records_with_preds = sum(1 for r in ap_records if len(r['scores']) > 0)
        records_with_gt = sum(1 for r in ap_records if r['num_gt'] > 0)
        iou_max_overall = max(
            (r['iou_rows'].max().item() for r in ap_records if r['iou_rows'].numel() > 0),
            default=0.0
        )
        logger.info(
            f'[eval-debug] ap_records summary: '
            f'{len(ap_records)} images, {total_preds} total preds, '
            f'{records_with_preds} images have preds, '
            f'{records_with_gt} images have GT, '
            f'max IoU across all pairs = {iou_max_overall:.4f}'
        )

    # Basic box-based metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = total_iou / (num_instances + 1e-8)

    # Mask AP from pre-computed IoU records (no pixel data needed here)
    ap_metrics = calculate_ap_metrics(ap_records, total_gt_masks)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'AP': ap_metrics['AP'],
        'AP50': ap_metrics['AP50'],
        'AP75': ap_metrics['AP75'],
        'AP70': ap_metrics['AP70'],
    }

    del ap_records
    torch.cuda.empty_cache()

    if logger:
        logger.info('Validation metrics:')
        for key, value in metrics.items():
            logger.info(f'  {key}: {value:.4f}')

    return metrics


def calculate_metrics_batch(
    predictions: List[Dict],
    targets: List[Dict]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a single batch
    Returns counters instead of final metrics to allow incremental computation
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_instances = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        target_boxes = target['boxes'].cpu()

        if len(pred_boxes) == 0:
            total_fn += len(target_boxes)
            continue

        if len(target_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        # Filter by score threshold
        score_thresh = 0.5
        mask = pred_scores > score_thresh
        pred_boxes = pred_boxes[mask]

        if len(pred_boxes) == 0:
            total_fn += len(target_boxes)
            continue

        # Calculate IoU
        iou_matrix = box_iou(pred_boxes, target_boxes)

        # Match predictions to targets
        iou_thresh = 0.5
        matched_targets = set()

        for i in range(len(pred_boxes)):
            best_iou, best_j = iou_matrix[i].max(0)
            best_j = best_j.item()
            best_iou = best_iou.item()

            if best_iou >= iou_thresh and best_j not in matched_targets:
                total_tp += 1
                total_iou += best_iou
                matched_targets.add(best_j)
                num_instances += 1
            else:
                total_fp += 1

        total_fn += len(target_boxes) - len(matched_targets)

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_iou': total_iou,
        'num_instances': num_instances
    }


def compute_mask_iou(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute pixel-level IoU between N predicted masks and M ground-truth masks.
    Uses GPU if available for large matrix operations.

    Args:
        pred_masks: [N, H, W] bool tensor
        gt_masks:   [M, H, W] bool tensor
        device:     GPU device for acceleration (None = CPU)

    Returns:
        iou_matrix: [N, M] float tensor on CPU
    """
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.float32)

    # Move to GPU for fast matrix multiply
    if device is not None and device.type == 'cuda':
        pred_flat = pred_masks.view(N, -1).float().to(device)
        gt_flat = gt_masks.view(M, -1).float().to(device)
    else:
        pred_flat = pred_masks.view(N, -1).float()
        gt_flat = gt_masks.view(M, -1).float()

    intersection = torch.mm(pred_flat, gt_flat.t())

    pred_area = pred_flat.sum(dim=1, keepdim=True)
    gt_area = gt_flat.sum(dim=1, keepdim=True)

    union = pred_area + gt_area.t() - intersection

    iou_matrix = intersection / (union + 1e-8)
    return iou_matrix.cpu()


def calculate_ap_metrics(
    ap_records: List[Dict],
    total_gt_masks: int
) -> Dict[str, float]:
    """
    Calculate mask AP at multiple IoU thresholds in a single pass.

    Instead of sorting and matching 13 times independently, we sort once
    and record per-prediction best IoU, then compute AP for each threshold.
    """
    if total_gt_masks == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AP70': 0.0}

    coco_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    all_thresholds = sorted(set(coco_thresholds + [0.7]))  # include AP70

    # Build flat arrays
    all_scores = []
    all_record_idx = []
    all_pred_idx = []

    for rec_idx, rec in enumerate(ap_records):
        n = len(rec['scores'])
        if n > 0:
            for pred_i in range(n):
                all_scores.append(rec['scores'][pred_i].item())
                all_record_idx.append(rec_idx)
                all_pred_idx.append(pred_i)

    if len(all_scores) == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AP70': 0.0}

    # Sort by score descending (once)
    order = sorted(range(len(all_scores)),
                   key=lambda i: all_scores[i], reverse=True)

    # For each threshold, maintain independent GT match tracking
    gt_matched = {}
    for t in all_thresholds:
        gt_matched[t] = [torch.zeros(rec['num_gt'], dtype=torch.bool)
                         for rec in ap_records]

    # best_iou[idx] for each prediction in sorted order
    # We need per-threshold TP/FP
    tp_lists = {t: [] for t in all_thresholds}
    fp_lists = {t: [] for t in all_thresholds}

    for idx in order:
        rec_idx = all_record_idx[idx]
        pred_i = all_pred_idx[idx]
        rec = ap_records[rec_idx]
        num_gt = rec['num_gt']

        if num_gt == 0:
            for t in all_thresholds:
                tp_lists[t].append(0)
                fp_lists[t].append(1)
            continue

        iou_row = rec['iou_rows'][pred_i]  # [M]

        for t in all_thresholds:
            iou_vals = iou_row.clone()
            iou_vals[gt_matched[t][rec_idx]] = -1.0
            best_iou, best_j = iou_vals.max(0)

            if best_iou.item() >= t:
                tp_lists[t].append(1)
                fp_lists[t].append(0)
                gt_matched[t][rec_idx][best_j.item()] = True
            else:
                tp_lists[t].append(0)
                fp_lists[t].append(1)

    # Compute AP for each threshold
    recall_points = torch.linspace(0, 1, 101)
    ap_per_threshold = {}

    for t in all_thresholds:
        tp_cum = torch.tensor(tp_lists[t], dtype=torch.float32).cumsum(0)
        fp_cum = torch.tensor(fp_lists[t], dtype=torch.float32).cumsum(0)

        recalls = torch.cat([torch.zeros(1), tp_cum / total_gt_masks])
        precisions = torch.cat([torch.ones(1), tp_cum / (tp_cum + fp_cum + 1e-8)])

        # 101-point interpolated AP
        ap = 0.0
        for rp in recall_points:
            mask = recalls >= rp
            ap += precisions[mask].max().item() if mask.any() else 0.0
        ap_per_threshold[t] = ap / 101.0

    coco_ap = sum(ap_per_threshold[t] for t in coco_thresholds) / len(coco_thresholds)

    return {
        'AP': coco_ap,
        'AP50': ap_per_threshold[0.5],
        'AP75': ap_per_threshold[0.75],
        'AP70': ap_per_threshold[0.7],
    }


def calculate_single_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU between two boxes

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    # Intersection coordinates
    x1_inter = max(box1[0].item(), box2[0].item())
    y1_inter = max(box1[1].item(), box2[1].item())
    x2_inter = min(box1[2].item(), box2[2].item())
    y2_inter = min(box1[3].item(), box2[3].item())

    # Intersection area
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-8)

    return float(iou)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)


def main(args):
    """Main training function"""
    
    # Setup
    output_dir = os.path.join(
        args.output_dir, 
        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f'Arguments: {args}')
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create dataloaders
    logger.info('Creating dataloaders...')
    train_loader = create_dataloader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_augmentation=True,
        with_depth=True
    )

    # Use larger batch size for validation if specified
    val_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size

    val_loader = create_dataloader(
        data_root=args.data_root,
        split='val',
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_augmentation=False,
        with_depth=True
    )
    
    # Create model
    logger.info('Building model...')
    model = build_model(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        lambda_boundary=args.lambda_boundary
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {num_params:,}')
    logger.info(f'Trainable parameters: {num_trainable:,}')
    
    # Optimizer with differential learning rates
    backbone_params = []
    daaf_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone.backbone' in name:
            # Pretrained ResNet-50 (RGB) and ResNet-18 (Depth) backbone layers
            backbone_params.append(param)
        elif 'backbone.daaf' in name or 'backbone.fpn' in name:
            # DAAF fusion modules and FPN convolutions
            daaf_params.append(param)
        else:
            # RPN, ROI heads, box/mask predictors, depth generator
            head_params.append(param)

    base_lr = args.learning_rate
    backbone_lr_scale = 0.1

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': base_lr * backbone_lr_scale},
        {'params': daaf_params, 'lr': base_lr},
        {'params': head_params, 'lr': base_lr},
    ], weight_decay=args.weight_decay)

    logger.info(f'Param groups: backbone={len(backbone_params)} (lr={base_lr * backbone_lr_scale}), '
                f'daaf={len(daaf_params)} (lr={base_lr}), heads={len(head_params)} (lr={base_lr})')

    # Learning rate scheduler: warmup + multi-step decay
    # MultiStepLR keeps LR high longer than cosine, preventing premature decay
    warmup_epochs = 3
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    step_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 35],
        gamma=0.1
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, step_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint:
            best_f1 = checkpoint['metrics'].get('f1', 0.0)
    
    # Model EMA for stable evaluation
    # Updated every optimizer step (~len(train_loader)/accumulation_steps times per epoch)
    # Buffers (BatchNorm running stats) are copied directly from training model
    ema = ModelEMA(model, decay=0.9998)
    updates_per_epoch = len(train_loader) // args.accumulation_steps
    logger.info(f'Model EMA initialized (decay=0.9998, ~{updates_per_epoch} updates/epoch)')

    # Training loop
    logger.info('Starting training...')
    logger.info(f'Gradient accumulation steps: {args.accumulation_steps} '
                f'(effective batch size: {args.batch_size * args.accumulation_steps})')
    epochs_without_improvement = 0
    best_ap50 = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        # Train (EMA updated every optimizer step inside train_one_epoch)
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            ema=ema,
            logger=logger
        )

        # Log to TensorBoard
        for key, value in train_losses.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        scheduler.step()

        # Evaluate using EMA model for stable metrics
        if (epoch + 1) % args.eval_interval == 0:
            # For first 3 epochs, also evaluate training model to diagnose issues
            if epoch < 3:
                logger.info(f'[diagnostic] Evaluating TRAINING model (epoch {epoch})...')
                train_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    logger=logger,
                    max_batches=args.max_eval_batches
                )
                logger.info(f'[diagnostic] Training model: AP50={train_metrics["AP50"]:.4f}, '
                            f'P={train_metrics["precision"]:.4f}, R={train_metrics["recall"]:.4f}')

            logger.info(f'Evaluating EMA model (epoch {epoch})...')
            metrics = evaluate(
                model=ema.ema_model,
                dataloader=val_loader,
                device=device,
                logger=logger,
                max_batches=args.max_eval_batches
            )

            # Log to TensorBoard
            for key, value in metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)

            # Save checkpoint — monitor AP50 as primary metric
            is_best = metrics['AP50'] > best_ap50
            if is_best:
                best_ap50 = metrics['AP50']
                best_f1 = metrics['f1']
                epochs_without_improvement = 0
                logger.info(f'New best AP50: {best_ap50:.4f} '
                            f'(AP75={metrics["AP75"]:.4f}, AP70={metrics["AP70"]:.4f}, '
                            f'F1={best_f1:.4f})')
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                model=ema.ema_model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                output_dir=output_dir,
                is_best=is_best
            )

            # Early stopping
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                logger.info(f'Early stopping: AP50 no improvement for {args.early_stopping_patience} epochs')
                break

    logger.info(f'Training complete. Best AP50: {best_ap50:.4f}, Best F1: {best_f1:.4f}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual-Stream Mask R-CNN')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (including background)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--lambda-boundary', type=float, default=1.0,
                        help='Weight for boundary loss')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Evaluation batch size (default: same as --batch-size)')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Base learning rate (backbone uses 0.1x)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (AdamW standard)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=1024,
                        help='Input image size')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * steps)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Stop if no F1 improvement for N epochs (0 = disabled)')

    # Evaluation
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluation interval (epochs)')
    parser.add_argument('--max-eval-batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (None = all)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)