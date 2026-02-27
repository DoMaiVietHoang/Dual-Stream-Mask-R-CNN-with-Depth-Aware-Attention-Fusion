"""
Quick diagnostic script to identify why mask AP = 0.0099.
Runs 3 batches of validation and prints detailed stats about:
  - num predictions vs GT per image
  - IoU matrix values
  - AP calculation intermediate values

Usage:
    python debug_ap.py --data-root <path> --checkpoint <path.pth>
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model
from data import create_dataloader


def compute_mask_iou(pred_masks, gt_masks):
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.float32)
    pred_flat = pred_masks.view(N, -1).float()
    gt_flat   = gt_masks.view(M, -1).float()
    intersection = torch.mm(pred_flat, gt_flat.t())
    pred_area = pred_flat.sum(dim=1, keepdim=True)
    gt_area   = gt_flat.sum(dim=1, keepdim=True)
    union = pred_area + gt_area.t() - intersection
    return intersection / (union + 1e-8)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model = build_model(num_classes=2, pretrained=False)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys')
    model = model.to(device)
    model.eval()

    # Dataloader (small batch, few batches)
    val_loader = create_dataloader(
        data_root=args.data_root,
        split='val',
        batch_size=1,
        num_workers=0,
        image_size=1024,
        use_augmentation=False,
        with_depth=True
    )

    print(f'\nRunning {args.num_batches} validation batches...\n')

    total_preds = 0
    total_gt    = 0
    all_ious    = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= args.num_batches:
                break

            images = images.to(device)
            depth_maps = None
            if 'depth' in targets[0]:
                depth_maps = torch.stack([t['depth'] for t in targets]).to(device)

            predictions = model(images, depth_maps=depth_maps)

            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # GT
                gt_m = target.get('masks', None)
                gt_masks = gt_m.bool().cpu() if (gt_m is not None and len(gt_m) > 0) else None
                num_gt = len(gt_masks) if gt_masks is not None else 0

                # Pred
                pred_m = pred.get('masks', None)
                pred_scores = pred['scores'].cpu()
                num_pred = len(pred_scores)

                if pred_m is not None and num_pred > 0:
                    # [N, 1, H, W] bool → [N, H, W] float for potential resize
                    pm_float = pred_m[:, 0].float().cpu()

                    if gt_masks is not None:
                        gt_h, gt_w = gt_masks.shape[-2], gt_masks.shape[-1]
                        pr_h, pr_w = pm_float.shape[-2], pm_float.shape[-1]
                        if (pr_h, pr_w) != (gt_h, gt_w):
                            print(f'  !! SIZE MISMATCH: pred={pr_h}x{pr_w}, gt={gt_h}x{gt_w} — resizing')
                            pm_float = F.interpolate(
                                pm_float.unsqueeze(0), size=(gt_h, gt_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(0)

                    pred_masks = pm_float > 0.5  # [N, H, W] bool
                else:
                    pred_masks = None

                # IoU matrix
                if pred_masks is not None and gt_masks is not None and num_pred > 0 and num_gt > 0:
                    iou_mat = compute_mask_iou(pred_masks, gt_masks)
                    max_iou = iou_mat.max().item()
                    mean_iou = iou_mat.mean().item()
                    # Count TPs at 0.5
                    row_max = iou_mat.max(dim=1).values
                    tp50 = (row_max >= 0.5).sum().item()
                    all_ious.extend(row_max.tolist())
                    iou_str = f'IoU max={max_iou:.4f} mean={mean_iou:.4f} TP@0.5={tp50}/{num_pred}'
                else:
                    iou_str = 'IoU=N/A (no pred or no gt)'

                # Pred mask coverage (are masks non-empty?)
                mask_coverage = 'N/A'
                if pred_masks is not None and num_pred > 0:
                    areas = pred_masks.float().view(num_pred, -1).sum(dim=1)
                    mask_coverage = f'pred mask areas: min={areas.min().item():.0f} max={areas.max().item():.0f} mean={areas.mean().item():.0f}'

                print(f'  Batch {batch_idx}, img {i}: '
                      f'num_pred={num_pred}, num_gt={num_gt}, '
                      f'score_range=[{pred_scores.min().item():.3f},{pred_scores.max().item():.3f}]'
                      if num_pred > 0 else
                      f'  Batch {batch_idx}, img {i}: num_pred=0, num_gt={num_gt}')
                print(f'    {iou_str}')
                if mask_coverage != 'N/A':
                    print(f'    {mask_coverage}')

                total_preds += num_pred
                total_gt += num_gt

    print(f'\n=== Summary ===')
    print(f'Total predictions: {total_preds}')
    print(f'Total GT masks:    {total_gt}')
    if all_ious:
        ious = np.array(all_ious)
        print(f'Per-pred max IoU stats:')
        print(f'  mean={ious.mean():.4f}  median={np.median(ious):.4f}  '
              f'max={ious.max():.4f}  min={ious.min():.4f}')
        print(f'  TP@0.5: {(ious >= 0.5).sum()}/{len(ious)} ({100*(ious>=0.5).mean():.1f}%)')
        print(f'  TP@0.3: {(ious >= 0.3).sum()}/{len(ious)} ({100*(ious>=0.3).mean():.1f}%)')
    else:
        print('No IoU values computed (no predictions or no GT in these batches)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--num-batches', type=int, default=5)
    args = parser.parse_args()
    main(args)
