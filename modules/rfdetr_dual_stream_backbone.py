"""
Dual-Stream Backbone for RF-DETR with Depth-Aware Attention Fusion (DAAF)

Design:
  - RGB stream  : DINOv2 ViT (windowed) — identical to stock RF-DETR backbone
  - Depth stream: ResNet-18 with 1-ch input — lightweight geometric encoder
  - Fusion      : MultiScaleDAAF applied to the 4 ViT intermediate features
                  BEFORE the MultiScaleProjector, so the projector still maps
                  to hidden_dim=256 for the transformer — no transformer changes.

Integration point in LWDETR.forward():
    features, poss = self.backbone(samples)   ← replaced by DualStreamRFDETRBackbone
    srcs = [feat.decompose()[0] for feat in features]
    hs = self.transformer(srcs, ...)

The wrapper accepts a NestedTensor whose .tensors holds RGB images AND a
separate depth_tensor argument passed as a keyword through the LWDETR
forward override (see DualStreamLWDETR below).

Architecture data flow (RF-DETR-Base example):
  Input RGB  [B, 3, H, W]      ←  DINOv2-S/14 windowed
  Input Depth[B, 1, H, W]      ←  ResNet-18 depth stream

  ViT intermediate features:     4 × [B, 384, H/14, W/14]
  ResNet-18 features (4 levels): [B,64,H/4,W/4]  [B,128,H/8,W/8]
                                  [B,256,H/16,W/16] [B,512,H/32,W/32]
  DAAF: resizes each depth level to match ViT spatial → fuse → [B,384,H/14,W/14] × 4
  MultiScaleProjector: 4×384 → hidden_dim=256 at P4 (one feature level for base)
  Output: List[NestedTensor]  [B, 256, H/14, W/14]   (same as stock RF-DETR)
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# RF-DETR imports (must be installed or on path)
RF_DETR_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rf-detr", "src"
)
if RF_DETR_SRC not in sys.path:
    sys.path.insert(0, RF_DETR_SRC)

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.util.misc import NestedTensor

from modules.daaf import DAAF


# ---------------------------------------------------------------------------
# Depth encoder: ResNet-18 with single-channel input
# Outputs 4 feature levels at strides 4, 8, 16, 32
# ---------------------------------------------------------------------------

class DepthEncoder(nn.Module):
    """
    Lightweight depth feature extractor based on ResNet-18.
    Accepts [B, 1, H, W] depth maps (normalized to [0,1]).
    Returns 4 feature maps at strides 4/8/16/32.
    """

    OUT_CHANNELS = [64, 128, 256, 512]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Replace first conv: 3-channel RGB → 1-channel depth
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # initialise with channel-average of pretrained RGB weights
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1    = resnet.bn1
        self.relu   = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # → [B,  64, H/4,  W/4 ]
        self.layer2 = resnet.layer2   # → [B, 128, H/8,  W/8 ]
        self.layer3 = resnet.layer3   # → [B, 256, H/16, W/16]
        self.layer4 = resnet.layer4   # → [B, 512, H/32, W/32]

    def forward(self, depth: torch.Tensor):
        x  = self.relu(self.bn1(self.conv1(depth)))
        x  = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]   # strides 4,8,16,32


# ---------------------------------------------------------------------------
# Multi-level DAAF: one DAAF per ViT feature level
# ViT outputs 4 levels, all at stride 14 (same spatial size H/14 × W/14)
# Depth levels come at 4 different strides — DAAF handles the spatial resize
# ---------------------------------------------------------------------------

class MultiLevelDAAF(nn.Module):
    """
    Apply DAAF fusion at each of the 4 ViT feature levels.

    ViT features: all [B, vit_dim, H/14, W/14]
    Depth features: [B, 64|128|256|512, H/4|H/8|H/16|H/32, ...]
                    (depth levels are resized inside DAAF to match ViT spatial)

    Output: 4 fused tensors [B, vit_dim, H/14, W/14]
    """

    def __init__(self, vit_dim: int, depth_channels: list, out_dim: int):
        """
        Args:
            vit_dim      : ViT embedding dim (384 for small, 768 for base)
            depth_channels: [64, 128, 256, 512]
            out_dim      : fusion output channels (set = vit_dim to keep projector compat)
        """
        super().__init__()
        self.daaf_levels = nn.ModuleList([
            DAAF(rgb_channels=vit_dim, depth_channels=d_ch, out_channels=out_dim)
            for d_ch in depth_channels
        ])

    def forward(self, vit_feats: list, depth_feats: list) -> list:
        """
        Args:
            vit_feats  : list of 4 tensors [B, vit_dim, Hv, Wv]
            depth_feats: list of 4 tensors [B, d_ch_i, H_i, W_i]
        Returns:
            fused: list of 4 tensors [B, out_dim, Hv, Wv]
        """
        fused = []
        for daaf, f_vit, f_depth in zip(self.daaf_levels, vit_feats, depth_feats):
            fused.append(daaf(f_vit, f_depth))
        return fused


# ---------------------------------------------------------------------------
# Main backbone wrapper: replaces the stock RF-DETR Backbone
# ---------------------------------------------------------------------------

class DualStreamRFDETRBackbone(nn.Module):
    """
    Drop-in replacement for RF-DETR's Backbone module.

    Keeps the DINOv2 ViT encoder and MultiScaleProjector from the original
    Backbone but inserts a depth encoder + DAAF fusion between them so that
    depth information modulates every ViT feature level before projection.

    Interface is identical to rf-detr Backbone.forward():
        Input : NestedTensor  (RGB images + padding mask)
                depth_tensor  [B, 1, H, W]  — extra argument
        Output: List[NestedTensor]           — same as stock backbone

    Usage inside DualStreamLWDETR.forward():
        features, poss = self.backbone(samples, depth_tensor=depth)
    """

    def __init__(
        self,
        rfdetr_backbone: Backbone,
        depth_pretrained: bool = True,
    ):
        """
        Args:
            rfdetr_backbone : the stock RF-DETR Backbone instance (already built
                              with pretrained DINOv2 weights loaded)
            depth_pretrained: whether to initialise the depth ResNet-18 with
                              ImageNet weights
        """
        super().__init__()

        # ── RGB stream: DINOv2 encoder (frozen or fine-tuned) ──────────────
        self.encoder   = rfdetr_backbone.encoder     # DinoV2 instance
        self.projector = rfdetr_backbone.projector   # MultiScaleProjector
        self.projector_scale = rfdetr_backbone.projector_scale

        # ── Depth stream: ResNet-18 ────────────────────────────────────────
        self.depth_encoder = DepthEncoder(pretrained=depth_pretrained)

        # Determine ViT embedding dim from projector's first sampling stage
        # MultiScaleProjector.stages_sampling[0] is a ModuleList of per-level
        # upsampling layers; when scale==1.0 the layer list is empty (Identity).
        # The in_channels passed to the projector equal encoder._out_feature_channels.
        vit_dim = rfdetr_backbone.encoder._out_feature_channels[0]
        # _out_feature_channels is a list, one per out_feature_index; all equal
        # the ViT embed_dim (384 for small, 768 for base).

        # ── DAAF: fuse each ViT level with corresponding depth level ──────
        self.daaf = MultiLevelDAAF(
            vit_dim=vit_dim,
            depth_channels=DepthEncoder.OUT_CHANNELS,   # [64,128,256,512]
            out_dim=vit_dim,    # keep same dim so projector input is unchanged
        )

        self._export = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor_list: NestedTensor, depth_tensor: torch.Tensor = None):
        """
        Args:
            tensor_list : NestedTensor — batched RGB images + masks
            depth_tensor: [B, 1, H, W] depth map in [0,1]; if None the
                          depth stream outputs zeros (graceful degradation)
        Returns:
            List[NestedTensor] — multi-scale fused features for transformer
        """
        rgb = tensor_list.tensors   # [B, 3, H, W]

        # 1. DINOv2 ViT features — 4 intermediate levels, all H/14 × W/14
        vit_feats = self.encoder(rgb)   # list of [B, vit_dim, Hv, Wv]

        # 2. Depth encoder
        if depth_tensor is not None:
            depth_feats = self.depth_encoder(depth_tensor)  # 4 levels
        else:
            # Graceful degradation: zero-fill depth features
            depth_feats = [
                torch.zeros(rgb.shape[0], d_ch, 1, 1, device=rgb.device)
                for d_ch in DepthEncoder.OUT_CHANNELS
            ]

        # 3. DAAF fusion: produces 4 fused tensors [B, vit_dim, Hv, Wv]
        fused_feats = self.daaf(vit_feats, depth_feats)

        # 4. MultiScaleProjector: maps fused features → hidden_dim (256)
        #    Interface: projector(list_of_tensors) → list_of_tensors
        projected = self.projector(fused_feats)

        # 5. Wrap with masks (same logic as stock Backbone.forward)
        out = []
        for feat in projected:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))

        return out

    # Keep position encoding building unchanged (delegated to encoder)
    def get_named_param_lr_pairs(self, args, prefix="backbone.0"):
        return self.encoder.get_named_param_lr_pairs(args, prefix)
