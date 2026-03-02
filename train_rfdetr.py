"""
Training Script for Dual-Stream RF-DETR with DAAF
Replaces Mask R-CNN backbone with RF-DETR (DINOv2 + DAAF depth fusion)

Usage:
    python train_rfdetr.py \
        --data-root /path/to/dataset \
        --variant base \
        --num-classes 1 \
        --batch-size 2 \
        --num-epochs 50 \
        --output-dir ./outputs_rfdetr

Dataset format: COCO-style (same as train.py)
Targets for RF-DETR need boxes in (cx, cy, w, h) normalized format.
"""

import os
import sys
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_stream_rfdetr import build_dual_stream_rfdetr, DualStreamLWDETR
from data import create_dataloader
from train import (
    compute_mask_iou, calculate_ap_metrics, calculate_ap_at_iou,
    calculate_metrics_batch, setup_logging
)


# ---------------------------------------------------------------------------
# Target format conversion: dataset uses xyxy absolute → RF-DETR needs cxcywh norm
# ---------------------------------------------------------------------------

def convert_targets_for_rfdetr(
    targets: List[Dict], image_h: int, image_w: int
) -> List[Dict]:
    """
    Convert target boxes from xyxy (absolute pixels) to cxcywh (normalised [0,1])
    as required by RF-DETR / LWDETR.

    Also ensures masks are present for segmentation loss.
    """
    out = []
    for t in targets:
        boxes = t['boxes'].clone().float()  # [N, 4] xyxy absolute
        if len(boxes) > 0:
            # xyxy → cxcywh
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w  = boxes[:, 2] - boxes[:, 0]
            h  = boxes[:, 3] - boxes[:, 1]
            boxes_cxcywh = torch.stack([cx, cy, w, h], dim=1)
            # Normalise
            boxes_cxcywh[:, [0, 2]] /= image_w
            boxes_cxcywh[:, [1, 3]] /= image_h
            boxes_cxcywh = boxes_cxcywh.clamp(0, 1)
        else:
            boxes_cxcywh = boxes

        new_t = {
            'boxes' : boxes_cxcywh,
            'labels': t['labels'],
        }
        if 'masks' in t:
            new_t['masks'] = t['masks']

        out.append(new_t)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DualStreamLWDETR,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:

    model.train()
    total_losses: Dict[str, float] = {}
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        image_h, image_w = images.shape[-2], images.shape[-1]

        # Extract depth
        depth_maps = None
        if 'depth' in targets[0]:
            depth_maps = torch.stack([t.pop('depth') for t in targets]).to(device)

        # Move targets to device
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Convert box format for RF-DETR
        rfdetr_targets = convert_targets_for_rfdetr(targets, image_h, image_w)

        try:
            loss_dict = model(images, targets=rfdetr_targets, depth_maps=depth_maps)
        except Exception as e:
            if logger:
                logger.error(f'Train batch {batch_idx} error: {e}')
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        # Only sum the main loss terms (ignore aux for scaling)
        weight_dict = model.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict.get(k, 1.0)
            for k in loss_dict
            if k in weight_dict
        )
        scaled_loss = losses / accumulation_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate for logging
        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        total_losses.setdefault('loss_total', 0.0)
        total_losses['loss_total'] += losses.item()

        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls' : f'{loss_dict.get("loss_ce", 0):.4f}',
            'bbox': f'{loss_dict.get("loss_bbox", 0):.4f}',
        })

        if batch_idx % 100 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

    n = len(dataloader)
    avg = {k: v / n for k, v in total_losses.items()}
    if logger:
        logger.info(f'Epoch {epoch} losses: ' +
                    ', '.join(f'{k}={v:.4f}' for k, v in avg.items()
                              if not k.startswith('loss_') or k in
                              ('loss_ce','loss_bbox','loss_giou','loss_mask_ce',
                               'loss_mask_dice','loss_boundary','loss_total')))
    return avg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: DualStreamLWDETR,
    dataloader: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    max_batches: Optional[int] = None,
    score_thresh: float = 0.3,
) -> Dict[str, float]:

    model.eval()

    total_tp = total_fp = total_fn = 0
    total_iou_sum = 0.0
    num_instances = 0
    ap_records = []
    total_gt_masks = 0

    pbar = tqdm(dataloader, desc='Evaluating')

    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        depth_maps = None
        if 'depth' in targets[0]:
            depth_maps = torch.stack([t['depth'] for t in targets]).to(device)

        try:
            results = model(images, depth_maps=depth_maps)
        except Exception as e:
            if logger:
                logger.error(f'Eval batch {batch_idx} error: {e}')
            torch.cuda.empty_cache()
            continue

        # PostProcess output: list of {'scores','labels','boxes','masks'}
        for i, (res, target) in enumerate(zip(results, targets)):
            # ── Mask AP ────────────────────────────────────────────────────
            gt_masks_raw = target.get('masks', None)
            if (gt_masks_raw is not None and isinstance(gt_masks_raw, torch.Tensor)
                    and len(gt_masks_raw) > 0):
                gt_masks = gt_masks_raw.bool().cpu()
            else:
                gt_masks = torch.zeros((0,), dtype=torch.bool)

            num_gt = len(gt_masks)
            total_gt_masks += num_gt

            pred_masks_raw = res.get('masks', None)
            pred_scores = res['scores'].cpu()

            if pred_masks_raw is not None and len(pred_masks_raw) > 0:
                # masks from PostProcess: [K, 1, H, W] bool
                pm_float = pred_masks_raw[:, 0].float().cpu()

                if num_gt > 0:
                    gt_h, gt_w = gt_masks.shape[-2], gt_masks.shape[-1]
                    pr_h, pr_w = pm_float.shape[-2], pm_float.shape[-1]
                    if (pr_h, pr_w) != (gt_h, gt_w):
                        pm_float = F.interpolate(
                            pm_float.unsqueeze(0), size=(gt_h, gt_w),
                            mode='bilinear', align_corners=False
                        ).squeeze(0)

                pred_masks = pm_float > 0.5
                del pm_float
            else:
                pred_masks = torch.zeros((0,), dtype=torch.bool)

            if len(pred_masks) > 0 and num_gt > 0:
                iou_matrix = compute_mask_iou(pred_masks, gt_masks)
            else:
                iou_matrix = torch.zeros((len(pred_scores), num_gt), dtype=torch.float32)

            ap_records.append({
                'scores'  : pred_scores,
                'iou_rows': iou_matrix,
                'num_gt'  : num_gt,
            })
            del pred_masks, gt_masks

            # ── Box P/R/F1 ─────────────────────────────────────────────────
            gt_boxes = target['boxes'].cpu()   # xyxy absolute
            pred_boxes = res['boxes'].cpu()    # xyxy absolute from PostProcess
            filt = pred_scores > score_thresh
            pred_boxes_filt = pred_boxes[filt]

            if len(pred_boxes_filt) == 0:
                total_fn += len(gt_boxes)
                continue
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes_filt)
                continue

            from torchvision.ops import box_iou as _box_iou
            iou_mat = _box_iou(pred_boxes_filt, gt_boxes)
            matched = set()
            for pi in range(len(pred_boxes_filt)):
                best_iou, best_j = iou_mat[pi].max(0)
                best_j = best_j.item()
                if best_iou.item() >= 0.5 and best_j not in matched:
                    total_tp += 1
                    total_iou_sum += best_iou.item()
                    matched.add(best_j)
                    num_instances += 1
                else:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched)

        del results
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()

        p = total_tp / (total_tp + total_fp + 1e-8)
        r = total_tp / (total_tp + total_fn + 1e-8)
        pbar.set_postfix({'P': f'{p:.3f}', 'R': f'{r:.3f}'})

    if logger:
        total_preds = sum(len(rec['scores']) for rec in ap_records)
        iou_max = max(
            (rec['iou_rows'].max().item() for rec in ap_records if rec['iou_rows'].numel() > 0),
            default=0.0
        )
        logger.info(f'[eval] {total_gt_masks} GT masks, {total_preds} preds, '
                    f'max_mask_IoU={iou_max:.4f}')

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou  = total_iou_sum / (num_instances + 1e-8)

    ap_metrics = calculate_ap_metrics(ap_records, total_gt_masks)
    del ap_records
    torch.cuda.empty_cache()

    metrics = {
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1,
        'mean_iou' : mean_iou,
        **ap_metrics,
    }

    if logger:
        logger.info('Validation: ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    ckpt = {
        'epoch'              : epoch,
        'model_state_dict'   : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics'            : metrics,
    }
    torch.save(ckpt, os.path.join(output_dir, 'checkpoint_latest.pth'))
    torch.save(ckpt, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
    if is_best:
        torch.save(ckpt, os.path.join(output_dir, 'checkpoint_best.pth'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_dir = os.path.join(
        args.output_dir,
        f'rfdetr_{args.variant}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f'Args: {args}')

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Dataloaders
    train_loader = create_dataloader(
        data_root=args.data_root, split='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, use_augmentation=True, with_depth=True,
    )
    val_loader = create_dataloader(
        data_root=args.data_root, split='val',
        batch_size=args.eval_batch_size or args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size, use_augmentation=False, with_depth=True,
    )

    # Model
    logger.info(f'Building Dual-Stream RF-DETR ({args.variant})...')
    model = build_dual_stream_rfdetr(
        num_classes=args.num_classes,
        variant=args.variant,
        lambda_boundary=args.lambda_boundary,
        depth_pretrained=True,
        use_depth_generator=True,
        depth_model_type='small',
        pretrain_weights=args.pretrain_weights,
        freeze_encoder=args.freeze_encoder,
        resolution=args.image_size,
        segmentation=True,
        device=args.device,
    )
    model = model.to(device)

    num_params     = sum(p.numel() for p in model.parameters())
    num_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Parameters: {num_params:,} total, {num_trainable:,} trainable')

    # Differential learning rates
    encoder_params = []
    daaf_params    = []
    other_params   = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lwdetr.backbone.backbone.encoder' in name:
            encoder_params.append(param)
        elif 'depth_encoder' in name or 'daaf' in name:
            daaf_params.append(param)
        else:
            other_params.append(param)

    base_lr = args.learning_rate
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': base_lr * 0.1},   # DINOv2 fine-tune slow
        {'params': daaf_params,    'lr': base_lr},          # new DAAF full lr
        {'params': other_params,   'lr': base_lr},          # transformer / heads
    ], weight_decay=args.weight_decay)

    logger.info(f'LR groups: encoder={len(encoder_params)} params ({base_lr*0.1:.2e}), '
                f'daaf={len(daaf_params)} ({base_lr:.2e}), '
                f'other={len(other_params)} ({base_lr:.2e})')

    # Scheduler: warmup + cosine
    warmup_epochs   = 3
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=base_lr * 1e-2
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # Resume
    start_epoch = 0
    best_ap50   = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_ap50   = ckpt.get('metrics', {}).get('AP50', 0.0)
        logger.info(f'Resumed from epoch {start_epoch}, best AP50={best_ap50:.4f}')

    # Training loop
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            accumulation_steps=args.accumulation_steps, logger=logger
        )
        for k, v in train_losses.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()

        if (epoch + 1) % args.eval_interval == 0:
            metrics = evaluate(
                model, val_loader, device, logger=logger,
                max_batches=args.max_eval_batches,
            )
            for k, v in metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            is_best = metrics['AP50'] > best_ap50
            if is_best:
                best_ap50      = metrics['AP50']
                epochs_no_improve = 0
                logger.info(f'New best AP50={best_ap50:.4f} at epoch {epoch}')
            else:
                epochs_no_improve += 1

            save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best)

            if (args.early_stopping_patience > 0
                    and epochs_no_improve >= args.early_stopping_patience):
                logger.info('Early stopping triggered.')
                break

    logger.info(f'Training complete. Best AP50={best_ap50:.4f}')
    writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser('Train Dual-Stream RF-DETR')

    # Data
    p.add_argument('--data-root',   required=True)
    p.add_argument('--output-dir',  default='./outputs_rfdetr')

    # Model
    p.add_argument('--variant',      default='base',
                   choices=['nano','small','medium','base','large'])
    p.add_argument('--num-classes',  type=int, default=1)
    p.add_argument('--pretrain-weights', default=None,
                   help='Path to RF-DETR pretrained .pth (None = train from scratch)')
    p.add_argument('--freeze-encoder', action='store_true', default=False)
    p.add_argument('--lambda-boundary', type=float, default=0.5)

    # Training
    p.add_argument('--batch-size',        type=int,   default=2)
    p.add_argument('--eval-batch-size',   type=int,   default=None)
    p.add_argument('--num-epochs',        type=int,   default=50)
    p.add_argument('--learning-rate',     type=float, default=1e-4)
    p.add_argument('--weight-decay',      type=float, default=1e-4)
    p.add_argument('--num-workers',       type=int,   default=4)
    p.add_argument('--image-size',        type=int,   default=1024)
    p.add_argument('--accumulation-steps',type=int,   default=4)
    p.add_argument('--early-stopping-patience', type=int, default=10)

    # Evaluation
    p.add_argument('--eval-interval',    type=int,   default=1)
    p.add_argument('--max-eval-batches', type=int,   default=None)

    # Device
    p.add_argument('--device',  default='cuda')
    p.add_argument('--resume',  default=None)

    args = p.parse_args()
    main(args)
