# Dual-Stream Mask R-CNN with Depth-Aware Attention Fusion (DAAF)

Instance segmentation of individual tree crowns from high-resolution aerial imagery using a dual-stream RGB+Depth architecture with a custom attention-based fusion module.

---

## Overview

Tree crown delineation in dense canopy is challenging due to overlapping and touching crowns, irregular shapes, and varying illumination. This work proposes a **Dual-Stream Mask R-CNN** that fuses RGB spectral features with monocular pseudo-depth estimates through a novel **Depth-Aware Attention Fusion (DAAF)** module, enabling the model to exploit geometric cues for better instance separation.

### Key Contributions

- **DAAF Module**: Channel attention + spatial attention fusion of RGB and Depth feature pyramids at each FPN scale
- **Boundary-Aware Loss**: Sobel-based edge loss supervises crown boundary sharpness
- **Dice Loss Integration**: Per-instance Dice loss directly optimises mask IoU, complementing pixel-wise BCE
- **Pseudo-Depth Generation**: Depth Anything V2 produces depth maps from RGB at inference, requiring no depth sensor

---

## Architecture

```
RGB Image [B, 3, 1024, 1024]  ──► ResNet-50 ──► [C1, C2, C3, C4]  ──┐
                                                                        ├──► DAAF ──► FPN ──► RPN ──► ROI Heads
Depth Map [B, 1, 1024, 1024]  ──► ResNet-18 ──► [D1, D2, D3, D4]  ──┘
```

**Loss function:**

```
L_total = L_cls + L_box + (L_bce + λ_dice · L_dice) + λ_bound · L_bound
```

| Component | Details |
|-----------|---------|
| RGB backbone | ResNet-50 (ImageNet-1K V2), output channels [256, 512, 1024, 2048] |
| Depth backbone | ResNet-18 (ImageNet-1K V1), 1-channel input, output channels [64, 128, 256, 512] |
| Fusion | DAAF: channel attention + spatial attention per FPN level |
| FPN | 4 lateral levels + 1 extra max-pool level (P2–P6) |
| Mask head | 4× Conv-BN-ReLU with residual shortcuts, 28×28 → 56×56 deconv |
| Depth source | Depth Anything V2 (small) — online or pre-computed |

---

## Results

Evaluated on the BAMFOREST dataset (2,006 validation images, 25,278 GT instances).

| Model | AP | AP50 | AP70 | AP75 | Precision | Recall | F1 | mIoU |
|-------|----|------|------|------|-----------|--------|----|------|
| Dual-Stream + DAAF + Boundary | 53.21 | 73.14 | 43.32 | — | ~0.50 | ~0.78 | ~0.61 | ~0.77 |

---

## Requirements

```bash
pip install torch torchvision
pip install albumentations pycocotools opencv-python tqdm tensorboard
```

Tested on:
- Python 3.10
- PyTorch 2.x + CUDA 11.8
- 1× NVIDIA GPU with ≥ 16 GB VRAM (for batch size 1, image size 1024×1024)

---

## Dataset Structure

```
BAMFOREST/
├── images/
│   ├── train/          # RGB images (.png / .jpg / .tif)
│   └── val/
├── annotations/
│   ├── train.json      # COCO-format instance segmentation annotations
│   └── val.json
└── depth/              # Pre-computed depth maps (optional)
    ├── train/          # <image_stem>_depth.npy
    └── val/
```

Annotations follow the COCO JSON format with `segmentation` (polygon or RLE), `bbox`, and `category_id` fields.

### Pre-computing Depth Maps (Recommended)

Pre-computing depth maps avoids repeated inference during training and saves ~2 min/epoch:

```bash
python Depth_generation/main.py \
    --data-root /path/to/BAMFOREST \
    --split train

python Depth_generation/main.py \
    --data-root /path/to/BAMFOREST \
    --split val
```

---

## Training

```bash
python train.py \
    --data-root /path/to/BAMFOREST \
    --output-dir ./out \
    --num-classes 2 \
    --num-epochs 50 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 0.0001 \
    --lambda-boundary 1.0 \
    --lambda-dice 0.5 \
    --image-size 1024 \
    --device cuda
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | required | Path to dataset root |
| `--num-epochs` | 50 | Total training epochs |
| `--batch-size` | 1 | Per-GPU batch size |
| `--accumulation-steps` | 4 | Gradient accumulation (effective batch = batch × steps) |
| `--learning-rate` | 1e-4 | Base LR; backbone uses 0.1× |
| `--lambda-boundary` | 1.0 | Weight for boundary (Sobel) loss |
| `--lambda-dice` | 0.5 | Weight for Dice loss |
| `--image-size` | 1024 | Input resolution |
| `--early-stopping-patience` | 10 | Stop if AP50 stagnates for N epochs |
| `--resume` | None | Path to checkpoint to resume from |

### Learning Rate Schedule

1. **Warmup** (epochs 0–2): linear ramp from 1% → 100% of base LR
2. **Multi-step decay**: ×0.1 at epoch 20, ×0.1 at epoch 35

### Optimizer

AdamW with differential learning rates:

| Parameter group | LR |
|---|---|
| Backbone (ResNet-50, ResNet-18) | `1e-5` (0.1×) |
| DAAF + FPN | `1e-4` (1×) |
| RPN + ROI heads | `1e-4` (1×) |

### Model EMA

An exponential moving average (decay = 0.9998) of model weights is maintained throughout training. Checkpoints and validation metrics are computed on the EMA model for more stable evaluation.

---

## Resuming Training

```bash
python train.py \
    --data-root /path/to/BAMFOREST \
    --resume ./out/train_YYYYMMDD_HHMMSS/checkpoint_best.pth \
    ...
```

---

## Output Directory Structure

```
out/
└── train_YYYYMMDD_HHMMSS/
    ├── train.log
    ├── tensorboard/
    ├── checkpoint_latest.pth
    └── checkpoint_best.pth       # Best AP50 on validation set
```

---

## Project Structure

```
Dual-Stream-Mask-R-CNN-with-Depth-Aware-Attention-Fusion/
├── train.py                        # Main training script
├── dual_stream_mask_rcnn.py        # Model: DualStreamMaskRCNN, CustomRoIHeads, DAAF integration
├── dataset.py                      # TreeCrownDataset, TreeCrownDatasetWithDepth
├── config.py                       # Default hyperparameters
├── debug_ap.py                     # AP diagnostic utility
├── visualization_inference.py      # Inference + visualisation script
├── models/
│   └── __init__.py                 # Exports build_model()
├── data/
│   └── __init__.py                 # Exports create_dataloader()
├── modules/
│   ├── daaf.py                     # DAAF, ChannelAttention, SpatialAttention, MultiScaleDAAF
│   ├── dual_stream_backbone.py     # RGBStream (ResNet-50), DepthStream (ResNet-18), DualStreamBackbone
│   ├── depth_generator.py          # DepthGenerator wrapping Depth Anything V2
│   └── losses.py                   # BoundaryLoss, DiceLoss, CombinedMaskLoss
└── Depth_generation/
    └── main.py                     # Offline depth pre-computation
```

---

## Loss Functions

### BCE Mask Loss
Standard `maskrcnn_loss` from torchvision — binary cross-entropy on per-class ROI-projected masks.

### Dice Loss *(new)*
Per-instance soft Dice computed on sigmoid predictions:

```
L_dice = 1 - mean_i [ (2 · |P_i ∩ G_i| + ε) / (|P_i| + |G_i| + ε) ]
```

Dice loss directly optimises mask overlap (same quantity as IoU-based AP), correcting the pixel-count bias of BCE in small or elongated crowns.

### Boundary Loss
Sobel edge magnitude L1 loss:

```
L_bound = || ∂(sigmoid(P)) - ∂(G) ||_1
```

Encourages crisp, accurate crown boundaries and better separation of touching instances.

### Combined Mask Loss

```
loss_mask   = L_bce  +  λ_dice  · L_dice
loss_boundary = λ_bound · L_bound
```

Both terms are reported separately in training logs for monitoring.

---

## Data Augmentation

Training augmentation pipeline (Albumentations):

| Transform | Probability |
|-----------|------------|
| Resize 1024×1024 | always |
| Horizontal flip | 0.5 |
| Vertical flip | 0.5 |
| Random rotate 90° | 0.5 |
| Shift / scale / rotate (±5%, ±10%, ±15°) | 0.5 |
| Color jitter or brightness/contrast | 0.5 |
| Gaussian noise or blur | 0.2 |

Depth maps are registered as `mask`-type targets so only geometric transforms are applied (no colour/noise).
