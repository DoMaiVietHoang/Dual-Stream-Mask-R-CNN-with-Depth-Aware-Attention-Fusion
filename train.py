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
    Evaluate model on validation set
    Calculates P/R/F1 and AP metrics (AP, AP50, AP70)

    Args:
        max_batches: Maximum number of batches to evaluate (None = all batches)
    """
    model.eval()

    # For basic metrics (P/R/F1) - calculate incrementally
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_instances = 0

    # For AP metrics - accumulate minimal data (only boxes, scores, labels, image_ids)
    all_predictions_minimal = []
    all_targets_minimal = []
    num_batches_processed = 0

    pbar = tqdm(dataloader, desc='Evaluating')

    for batch_idx, (images, targets) in enumerate(pbar):
        # Limit evaluation batches for faster validation
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

        # Extract minimal info for AP calculation (to save memory)
        for i, pred in enumerate(predictions):
            pred_minimal = {
                'boxes': pred['boxes'].detach().cpu(),
                'scores': pred['scores'].detach().cpu(),
                'labels': pred['labels'].detach().cpu() if 'labels' in pred else torch.ones(len(pred['boxes']), dtype=torch.int64),
                'image_id': batch_idx * len(predictions) + i
            }
            all_predictions_minimal.append(pred_minimal)

        # Extract minimal target info
        for i, target in enumerate(targets):
            target_minimal = {
                'boxes': target['boxes'].cpu() if isinstance(target['boxes'], torch.Tensor) else target['boxes'],
                'labels': target['labels'].cpu() if isinstance(target['labels'], torch.Tensor) else target['labels'],
                'image_id': batch_idx * len(targets) + i
            }
            all_targets_minimal.append(target_minimal)

        # Calculate basic metrics for this batch (for progress tracking)
        batch_metrics = calculate_metrics_batch(
            [{'boxes': p['boxes'], 'scores': p['scores']} for p in all_predictions_minimal[-len(predictions):]],
            targets
        )

        total_tp += batch_metrics['tp']
        total_fp += batch_metrics['fp']
        total_fn += batch_metrics['fn']
        total_iou += batch_metrics['total_iou']
        num_instances += batch_metrics['num_instances']
        num_batches_processed += 1

        # Free GPU memory immediately
        del predictions

        # Update progress with current metrics
        current_precision = total_tp / (total_tp + total_fp + 1e-8)
        current_recall = total_tp / (total_tp + total_fn + 1e-8)
        pbar.set_postfix({
            'batch': f'{batch_idx+1}/{len(dataloader)}',
            'P': f'{current_precision:.3f}',
            'R': f'{current_recall:.3f}'
        })

        # Clear CUDA cache periodically
        if batch_idx % 20 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

    # Calculate final metrics
    if logger:
        logger.info(f'Calculating metrics from {num_batches_processed} batches...')

    # Basic metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = total_iou / (num_instances + 1e-8)

    # Calculate AP metrics (AP, AP50, AP70)
    ap_metrics = calculate_ap_metrics(all_predictions_minimal, all_targets_minimal)

    # Combine all metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'AP': ap_metrics['AP'],
        'AP50': ap_metrics['AP50'],
        'AP70': ap_metrics['AP70']
    }

    # Free memory
    del all_predictions_minimal, all_targets_minimal
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


def calculate_ap_metrics(
    predictions: List[Dict],
    targets: List[Dict]
) -> Dict[str, float]:
    """
    Calculate AP metrics (AP, AP50, AP70) from predictions and targets

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels', 'image_id'
        targets: List of dicts with 'boxes', 'labels', 'image_id'

    Returns:
        Dict with AP, AP50, AP70 values
    """
    # Group predictions and targets by image_id
    pred_by_image = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)

    target_by_image = {}
    for target in targets:
        img_id = target['image_id']
        target_by_image[img_id] = target

    # Calculate AP at different IoU thresholds
    iou_thresholds = {
        'AP50': 0.5,
        'AP70': 0.7,
        'AP': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # COCO-style
    }

    ap_results = {}

    # Calculate AP50
    ap_results['AP50'] = calculate_ap_at_iou(pred_by_image, target_by_image, 0.5)

    # Calculate AP70
    ap_results['AP70'] = calculate_ap_at_iou(pred_by_image, target_by_image, 0.7)

    # Calculate AP (average over multiple IoU thresholds)
    ap_values = []
    for iou_thresh in iou_thresholds['AP']:
        ap = calculate_ap_at_iou(pred_by_image, target_by_image, iou_thresh)
        ap_values.append(ap)
    ap_results['AP'] = sum(ap_values) / len(ap_values)

    return ap_results


def calculate_ap_at_iou(
    pred_by_image: Dict,
    target_by_image: Dict,
    iou_threshold: float
) -> float:
    """
    Calculate Average Precision at a specific IoU threshold

    Args:
        pred_by_image: Predictions grouped by image_id
        target_by_image: Targets grouped by image_id
        iou_threshold: IoU threshold for matching

    Returns:
        Average Precision value
    """
    # Collect all predictions with their scores
    all_predictions = []

    for img_id, preds in pred_by_image.items():
        for pred in preds:
            all_predictions.append({
                'image_id': img_id,
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels']
            })

    if len(all_predictions) == 0:
        return 0.0

    # Flatten and sort all predictions by score (descending)
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []

    for pred in all_predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        for i in range(len(boxes)):
            all_pred_boxes.append(boxes[i])
            all_pred_scores.append(scores[i].item())
            all_pred_image_ids.append(pred['image_id'])

    if len(all_pred_boxes) == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = sorted(range(len(all_pred_scores)), key=lambda i: all_pred_scores[i], reverse=True)

    # Count total ground truth boxes
    total_gt = 0
    for img_id, target in target_by_image.items():
        total_gt += len(target['boxes'])

    if total_gt == 0:
        return 0.0

    # Track which ground truth boxes have been matched
    gt_matched = {img_id: [False] * len(target['boxes'])
                  for img_id, target in target_by_image.items()}

    # Calculate precision and recall at each prediction
    tp = []
    fp = []

    for idx in sorted_indices:
        pred_box = all_pred_boxes[idx]
        img_id = all_pred_image_ids[idx]

        if img_id not in target_by_image:
            fp.append(1)
            tp.append(0)
            continue

        target = target_by_image[img_id]
        gt_boxes = target['boxes']

        if len(gt_boxes) == 0:
            fp.append(1)
            tp.append(0)
            continue

        # Calculate IoU with all ground truth boxes in this image
        ious = []
        for gt_box in gt_boxes:
            iou = calculate_single_iou(pred_box, gt_box)
            ious.append(iou)

        # Find best matching ground truth
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        # Check if it's a true positive
        if max_iou >= iou_threshold and not gt_matched[img_id][max_iou_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[img_id][max_iou_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Calculate cumulative TP and FP
    tp_cumsum = torch.tensor(tp).cumsum(0).float()
    fp_cumsum = torch.tensor(fp).cumsum(0).float()

    # Calculate precision and recall
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Calculate AP using 11-point interpolation
    ap = 0.0
    for recall_threshold in torch.linspace(0, 1, 11):
        precisions_above = precisions[recalls >= recall_threshold]
        if len(precisions_above) > 0:
            ap += precisions_above.max().item()
    ap /= 11.0

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

            # Save checkpoint
            is_best = metrics['f1'] > best_f1
            if is_best:
                best_f1 = metrics['f1']
                epochs_without_improvement = 0
                logger.info(f'New best F1: {best_f1:.4f}')
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
                logger.info(f'Early stopping: no improvement for {args.early_stopping_patience} epochs')
                break

    logger.info(f'Training complete. Best F1: {best_f1:.4f}')
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