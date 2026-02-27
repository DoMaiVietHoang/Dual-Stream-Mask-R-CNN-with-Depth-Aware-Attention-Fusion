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

    Mask AP uses pixel-level mask IoU for TP/FP matching — the correct metric
    for instance segmentation tasks like tree crown delineation.

    Args:
        max_batches: Maximum number of batches to evaluate (None = all batches)
    """
    model.eval()

    # For basic metrics (P/R/F1) - box-based, calculated incrementally
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_instances = 0

    # For mask AP — accumulate (pred_masks, pred_scores, gt_masks) per image
    all_predictions_seg = []
    all_targets_seg = []
    num_batches_processed = 0

    pbar = tqdm(dataloader, desc='Evaluating')

    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            if logger:
                logger.info(f'Stopped evaluation after {max_batches} batches')
            break

        images = images.to(device)

        # Extract pre-computed depth maps from targets if available
        depth_maps = None
        if len(targets) > 0 and 'depth' in targets[0]:
            depth_maps = torch.stack([t['depth'] for t in targets]).to(device)

        # Get predictions
        try:
            predictions = model(images, depth_maps=depth_maps)
        except Exception as e:
            if logger:
                logger.error(f'Error in batch {batch_idx}: {e}')
            torch.cuda.empty_cache()
            continue

        image_id_base = batch_idx * len(predictions)

        for i, pred in enumerate(predictions):
            img_id = image_id_base + i

            # Binary masks: pred['masks'] is [N, 1, H, W] float in [0,1]
            pred_masks_raw = pred.get('masks', None)
            if pred_masks_raw is not None and len(pred_masks_raw) > 0:
                # Threshold to binary [N, H, W] bool, store as uint8 to save RAM
                # bool tensor: 1 bit/pixel → uint8: 1 byte/pixel (8x more but simpler)
                pred_masks_bin = (pred_masks_raw[:, 0] > 0.5).cpu().to(torch.uint8)
            else:
                pred_masks_bin = torch.zeros(
                    (0, images.shape[2], images.shape[3]), dtype=torch.uint8
                )

            all_predictions_seg.append({
                'masks': pred_masks_bin,                          # [N, H, W] uint8
                'scores': pred['scores'].detach().cpu(),          # [N]
                'boxes': pred['boxes'].detach().cpu(),            # [N, 4] for P/R/F1
                'image_id': img_id
            })

        for i, target in enumerate(targets):
            img_id = image_id_base + i
            # GT masks: [M, H, W] uint8 from dataset
            gt_masks = target.get('masks', None)
            if gt_masks is not None and isinstance(gt_masks, torch.Tensor):
                gt_masks = gt_masks.to(torch.uint8).cpu()
            else:
                gt_masks = torch.zeros(
                    (0, images.shape[2], images.shape[3]), dtype=torch.uint8
                )
            all_targets_seg.append({
                'masks': gt_masks,                                    # [M, H, W] uint8
                'boxes': target['boxes'].cpu(),                       # for P/R/F1
                'image_id': img_id
            })

        # Box-based P/R/F1 tracking (fast, incremental)
        batch_metrics = calculate_metrics_batch(
            [{'boxes': p['boxes'], 'scores': p['scores']} for p in all_predictions_seg[-len(predictions):]],
            targets
        )
        total_tp += batch_metrics['tp']
        total_fp += batch_metrics['fp']
        total_fn += batch_metrics['fn']
        total_iou += batch_metrics['total_iou']
        num_instances += batch_metrics['num_instances']
        num_batches_processed += 1

        del predictions
        torch.cuda.empty_cache() if batch_idx % 20 == 0 and batch_idx > 0 else None

        current_precision = total_tp / (total_tp + total_fp + 1e-8)
        current_recall = total_tp / (total_tp + total_fn + 1e-8)
        pbar.set_postfix({
            'batch': f'{batch_idx+1}/{len(dataloader)}',
            'P': f'{current_precision:.3f}',
            'R': f'{current_recall:.3f}'
        })

    if logger:
        logger.info(f'Calculating mask AP from {num_batches_processed} batches...')

    # Basic box-based metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = total_iou / (num_instances + 1e-8)

    # Mask segmentation AP (pixel-level IoU)
    ap_metrics = calculate_ap_metrics(all_predictions_seg, all_targets_seg)

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

    del all_predictions_seg, all_targets_seg
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


def compute_mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-level IoU between N predicted masks and M ground-truth masks.

    Args:
        pred_masks: [N, H, W] bool tensor
        gt_masks:   [M, H, W] bool tensor

    Returns:
        iou_matrix: [N, M] float tensor, each entry is mask IoU between pred i and gt j
    """
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.float32)

    # Flatten to [N, H*W] and [M, H*W]
    pred_flat = pred_masks.view(N, -1).float()   # [N, HW]
    gt_flat = gt_masks.view(M, -1).float()       # [M, HW]

    # intersection[i, j] = sum(pred_i AND gt_j)
    intersection = torch.mm(pred_flat, gt_flat.t())  # [N, M]

    pred_area = pred_flat.sum(dim=1, keepdim=True)   # [N, 1]
    gt_area = gt_flat.sum(dim=1, keepdim=True)       # [M, 1]

    union = pred_area + gt_area.t() - intersection   # [N, M]

    iou_matrix = intersection / (union + 1e-8)
    return iou_matrix


def calculate_ap_metrics(
    predictions: List[Dict],
    targets: List[Dict]
) -> Dict[str, float]:
    """
    Calculate mask segmentation AP metrics (AP, AP50, AP70, AP75).

    IoU is computed at the pixel level between predicted binary masks and
    ground-truth binary masks — the correct metric for instance segmentation.

    Args:
        predictions: List of dicts with 'masks' [N,H,W], 'scores' [N], 'image_id'
        targets:     List of dicts with 'masks' [M,H,W], 'image_id'

    Returns:
        Dict with AP, AP50, AP70, AP75 values
    """
    # Group by image_id and concatenate masks/scores (handles batch_size>1)
    pred_by_image: Dict = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = {'masks': [], 'scores': []}
        if len(pred['masks']) > 0:
            pred_by_image[img_id]['masks'].append(pred['masks'])
            pred_by_image[img_id]['scores'].append(pred['scores'])

    for img_id in pred_by_image:
        masks_list = pred_by_image[img_id]['masks']
        scores_list = pred_by_image[img_id]['scores']
        pred_by_image[img_id]['masks'] = (
            torch.cat(masks_list, dim=0) if masks_list
            else torch.zeros((0,), dtype=torch.bool)
        )
        pred_by_image[img_id]['scores'] = (
            torch.cat(scores_list, dim=0) if scores_list
            else torch.zeros(0)
        )

    target_by_image: Dict = {}
    for target in targets:
        img_id = target['image_id']
        target_by_image[img_id] = target

    coco_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ap_results = {}
    ap_results['AP50'] = calculate_ap_at_iou(pred_by_image, target_by_image, 0.5)
    ap_results['AP75'] = calculate_ap_at_iou(pred_by_image, target_by_image, 0.75)
    ap_results['AP70'] = calculate_ap_at_iou(pred_by_image, target_by_image, 0.7)
    ap_values = [calculate_ap_at_iou(pred_by_image, target_by_image, t) for t in coco_thresholds]
    ap_results['AP'] = sum(ap_values) / len(ap_values)

    return ap_results


def calculate_ap_at_iou(
    pred_by_image: Dict,
    target_by_image: Dict,
    iou_threshold: float
) -> float:
    """
    Calculate mask AP at a specific IoU threshold using 101-point AUC interpolation.

    Protocol (COCO-style):
    - All predicted masks are sorted globally by confidence score descending
    - Each prediction is matched to the unmatched GT mask with the highest
      pixel-level mask IoU; if IoU >= threshold it is a TP, otherwise FP
    - AP = area under the precision-recall curve (101-point interpolation)

    Args:
        pred_by_image: {img_id: {'masks': Tensor[N,H,W], 'scores': Tensor[N]}}
        target_by_image: {img_id: {'masks': Tensor[M,H,W]}}
        iou_threshold: mask IoU threshold for TP matching

    Returns:
        AP value in [0, 1]
    """
    # Flatten all predictions into parallel lists
    all_pred_masks = []
    all_pred_scores = []
    all_pred_image_ids = []

    for img_id, pred in pred_by_image.items():
        masks = pred['masks']
        scores = pred['scores']
        if len(masks) == 0:
            continue
        for i in range(len(masks)):
            all_pred_masks.append(masks[i])        # [H, W] bool
            all_pred_scores.append(scores[i].item())
            all_pred_image_ids.append(img_id)

    if len(all_pred_masks) == 0:
        return 0.0

    total_gt = sum(len(t['masks']) for t in target_by_image.values())
    if total_gt == 0:
        return 0.0

    # Sort by confidence descending
    order = sorted(range(len(all_pred_scores)),
                   key=lambda i: all_pred_scores[i], reverse=True)

    gt_matched = {img_id: torch.zeros(len(t['masks']), dtype=torch.bool)
                  for img_id, t in target_by_image.items()}

    tp = []
    fp = []

    for idx in order:
        img_id = all_pred_image_ids[idx]

        if img_id not in target_by_image:
            fp.append(1); tp.append(0)
            continue

        gt_masks = target_by_image[img_id]['masks']
        if len(gt_masks) == 0:
            fp.append(1); tp.append(0)
            continue

        pred_mask = all_pred_masks[idx].unsqueeze(0)  # [1, H, W]

        # Compute IoU between this single prediction and all GT masks of this image
        iou_vals = compute_mask_iou(pred_mask, gt_masks)[0]  # [M]

        # Exclude already-matched GT masks
        iou_vals[gt_matched[img_id]] = -1.0

        best_iou, best_j = iou_vals.max(0)
        best_iou = best_iou.item()
        best_j = best_j.item()

        if best_iou >= iou_threshold:
            tp.append(1); fp.append(0)
            gt_matched[img_id][best_j] = True
        else:
            tp.append(0); fp.append(1)

    tp_cumsum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_cumsum = torch.tensor(fp, dtype=torch.float32).cumsum(0)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Prepend sentinel (recall=0, precision=1)
    recalls = torch.cat([torch.zeros(1), recalls])
    precisions = torch.cat([torch.ones(1), precisions])

    # 101-point interpolation (COCO-style AUC)
    ap = 0.0
    for t in torch.linspace(0, 1, 101):
        mask = recalls >= t
        ap += precisions[mask].max().item() if mask.any() else 0.0
    ap /= 101.0

    return ap


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
    
    # Training loop
    logger.info('Starting training...')
    logger.info(f'Gradient accumulation steps: {args.accumulation_steps} '
                f'(effective batch size: {args.batch_size * args.accumulation_steps})')
    epochs_without_improvement = 0
    best_ap50 = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            logger=logger
        )

        # Log to TensorBoard
        for key, value in train_losses.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        scheduler.step()

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            metrics = evaluate(
                model=model,
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
                model=model,
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