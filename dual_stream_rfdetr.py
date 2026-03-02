"""
Dual-Stream RF-DETR with Depth-Aware Attention Fusion (DAAF)

Architecture:
  RGB stream   → DINOv2 ViT (windowed) — RF-DETR backbone, pretrained
  Depth stream → ResNet-18 (1-ch input) — geometric encoder, pretrained
  Fusion       → MultiLevelDAAF between ViT features and depth features
  Projector    → MultiScaleProjector → hidden_dim=256
  Decoder      → Deformable transformer (RF-DETR unchanged)
  Seg head     → SegmentationHead (RF-DETR unchanged)
  Boundary loss→ Added on top of seg head output

Usage:
    model = build_dual_stream_rfdetr(num_classes=1, variant='base',
                                     lambda_boundary=0.5)
    losses = model(images, targets, depth_maps=depth)   # training
    results = model(images, depth_maps=depth)            # inference
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── RF-DETR on path ────────────────────────────────────────────────────────
RF_DETR_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "rf-detr", "src")
if RF_DETR_SRC not in sys.path:
    sys.path.insert(0, RF_DETR_SRC)

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone import build_backbone
from rfdetr.models.lwdetr import LWDETR, SetCriterion, PostProcess, build_criterion_and_postprocessors
from rfdetr.models.transformer import build_transformer
from rfdetr.models.segmentation_head import SegmentationHead
from rfdetr.models.matcher import build_matcher
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from rfdetr.config import (
    RFDETRBaseConfig, RFDETRLargeConfig, RFDETRSmallConfig,
    RFDETRMediumConfig, RFDETRNanoConfig,
)

# ── Local modules ──────────────────────────────────────────────────────────
from modules.rfdetr_dual_stream_backbone import DualStreamRFDETRBackbone
from modules.losses import BoundaryLoss
from modules.depth_generator import DepthGenerator


# ---------------------------------------------------------------------------
# DualStream Joiner — replaces RF-DETR's Joiner
# Joiner[0] = backbone, Joiner[1] = position_embedding
# We need to intercept forward to pass depth_tensor to backbone
# ---------------------------------------------------------------------------

class DualStreamJoiner(nn.Module):
    """
    Wraps DualStreamRFDETRBackbone + position embedding.
    Accepts an optional depth_tensor in forward().
    """

    def __init__(self, dual_stream_backbone: DualStreamRFDETRBackbone,
                 position_embedding: nn.Module):
        super().__init__()
        self.backbone = dual_stream_backbone
        self.position_embedding = position_embedding

    def forward(self, tensor_list: NestedTensor,
                depth_tensor: Optional[torch.Tensor] = None):
        features = self.backbone(tensor_list, depth_tensor=depth_tensor)
        pos = []
        for feat in features:
            pos.append(
                self.position_embedding(feat, align_dim_orders=False)
                .to(feat.tensors.dtype)
            )
        return features, pos


# ---------------------------------------------------------------------------
# DualStreamLWDETR — LWDETR extended with depth input + boundary loss
# ---------------------------------------------------------------------------

class DualStreamLWDETR(nn.Module):
    """
    Dual-Stream RF-DETR model.

    Extends LWDETR by:
    1. Replacing the backbone with DualStreamJoiner (RGB + depth fusion)
    2. Adding optional pseudo-depth generation (Depth Anything V2)
    3. Adding boundary loss on top of segmentation masks

    Forward (training):
        losses = model(images, targets, depth_maps=depth)

    Forward (inference):
        results = model(images, depth_maps=depth)   # returns PostProcess output

    Args:
        joiner          : DualStreamJoiner (backbone + pos encoding)
        transformer     : RF-DETR transformer
        segmentation_head: SegmentationHead (optional, set None for det-only)
        num_classes     : number of foreground classes (excl. background)
        num_queries     : number of decoder queries
        criterion       : SetCriterion for loss computation
        postprocessor   : PostProcess for inference decoding
        lambda_boundary : weight for boundary loss (0 = disabled)
        use_depth_generator: auto-generate depth if depth_maps not supplied
        depth_model_type: 'small' | 'base' | 'large' for Depth Anything V2
        aux_loss        : use auxiliary decoder-layer losses
        group_detr      : number of training groups
        two_stage       : enable two-stage encoder proposals
        lite_refpoint_refine: lightweight reference point refinement
        bbox_reparam    : bounding box reparameterization
    """

    def __init__(
        self,
        joiner: DualStreamJoiner,
        transformer: nn.Module,
        segmentation_head: Optional[nn.Module],
        num_classes: int,
        num_queries: int,
        criterion: SetCriterion,
        postprocessor: PostProcess,
        lambda_boundary: float = 0.5,
        use_depth_generator: bool = True,
        depth_model_type: str = 'small',
        aux_loss: bool = True,
        group_detr: int = 13,
        two_stage: bool = True,
        lite_refpoint_refine: bool = True,
        bbox_reparam: bool = True,
    ):
        super().__init__()

        # Build inner LWDETR using the dual-stream joiner
        self.lwdetr = LWDETR(
            backbone=joiner,
            transformer=transformer,
            segmentation_head=segmentation_head,
            num_classes=num_classes + 1,   # LWDETR uses +1 convention
            num_queries=num_queries,
            aux_loss=aux_loss,
            group_detr=group_detr,
            two_stage=two_stage,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        self.criterion    = criterion
        self.postprocessor = postprocessor
        self.num_classes  = num_classes

        # Boundary loss on predicted masks
        self.lambda_boundary = lambda_boundary
        if lambda_boundary > 0:
            self.boundary_loss_fn = BoundaryLoss(edge_type='sobel')
        else:
            self.boundary_loss_fn = None

        # Optional pseudo-depth generation
        self.use_depth_generator = use_depth_generator
        if use_depth_generator:
            self.depth_generator = DepthGenerator(
                model_type=depth_model_type, pretrained=True
            )
        else:
            self.depth_generator = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_generate_depth(
        self, images: torch.Tensor,
        depth_maps: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return depth_maps if provided, else generate with DepthGenerator."""
        if depth_maps is not None:
            return depth_maps
        if self.depth_generator is not None:
            with torch.no_grad():
                return self.depth_generator(images)
        return None

    @staticmethod
    def _normalize_images(images: torch.Tensor) -> torch.Tensor:
        """ImageNet normalization: uint8 [0,255] or float [0,1] → normalized."""
        if images.dtype == torch.uint8 or images.max() > 1.0:
            images = images.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1,3,1,1)
        return (images - mean) / std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        depth_maps: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            images    : [B, 3, H, W]  uint8 [0,255] or float [0,1]
            targets   : list of dicts with 'boxes' (cxcywh, normalised),
                        'labels', and optionally 'masks'
            depth_maps: [B, 1, H, W] in [0,1]; auto-generated if None

        Returns:
            training  : dict of losses
            inference : list of dicts {'scores','labels','boxes',['masks']}
        """
        depth = self._get_or_generate_depth(images, depth_maps)
        images_norm = self._normalize_images(images)

        # Build NestedTensor (no padding for fixed-size inputs)
        nested = nested_tensor_from_tensor_list(list(images_norm))

        # Patch Joiner.forward to pass depth_tensor through
        # DualStreamJoiner.forward(tensor_list, depth_tensor)
        # but LWDETR calls self.backbone(samples) — we override via hook
        features, poss = self.lwdetr.backbone(nested, depth_tensor=depth)

        srcs  = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        if self.training:
            refpt = self.lwdetr.refpoint_embed.weight
            qfeat = self.lwdetr.query_feat.weight
        else:
            refpt = self.lwdetr.refpoint_embed.weight[:self.lwdetr.num_queries]
            qfeat = self.lwdetr.query_feat.weight[:self.lwdetr.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.lwdetr.transformer(
            srcs, masks, poss, refpt, qfeat
        )

        # Build outputs dict
        if hs is not None:
            if self.lwdetr.bbox_reparam:
                d = self.lwdetr.bbox_embed(hs)
                coord = torch.cat([
                    d[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2],
                    d[..., 2:].exp() * ref_unsigmoid[..., 2:],
                ], dim=-1)
            else:
                coord = (self.lwdetr.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            cls_out = self.lwdetr.class_embed(hs)
            out = {"pred_logits": cls_out[-1], "pred_boxes": coord[-1]}

            if self.lwdetr.segmentation_head is not None:
                # Dense forward on the last decoder layer only → [B, N*g, H_m, W_m]
                # Avoids materialising all L layers at once (memory efficiency).
                # hs[-1] is the last layer output [B, N*g, C]; wrap in list for seg head.
                last_masks = self.lwdetr.segmentation_head.forward(
                    features[0].tensors, hs[-1:], nested.tensors.shape[-2:]
                )[0]   # [B, N*g, H_m, W_m]
                out["pred_masks"] = last_masks

            if self.lwdetr.aux_loss:
                # For aux outputs: skip mask loss (too memory-intensive to compute
                # [B, N*g, H_m, W_m] for every decoder layer at 1024×1024 input).
                # Box and class aux losses still apply.
                out["aux_outputs"] = self.lwdetr._set_aux_loss(cls_out, coord, None)

        if self.lwdetr.two_stage:
            g = self.lwdetr.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(g, dim=1)
            cls_enc = torch.cat([
                self.lwdetr.transformer.enc_out_class_embed[gi](hs_enc_list[gi])
                for gi in range(g)
            ], dim=1)

            # NOTE: We intentionally skip enc mask prediction.
            # At 1024×1024, hs_enc has ~N_enc*g tokens which makes the einsum
            # [B, N_enc*g, H_m, W_m] prohibitively large (~several GiB).
            # Only box+class losses are computed for enc_outputs.
            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        # ── Training: compute losses ──────────────────────────────────────
        if self.training:
            assert targets is not None
            # criterion only computes box + class losses (mask losses disabled)
            losses = self.criterion(out, targets)

            # ── Manual mask losses on last-layer predictions ──────────────
            if self.lwdetr.segmentation_head is not None and "pred_masks" in out:
                pred_masks = out["pred_masks"]   # [B, N*g, H_m, W_m]
                mask_losses = self._compute_mask_losses(pred_masks, targets)
                losses.update(mask_losses)

                # Boundary loss
                if self.boundary_loss_fn is not None:
                    b_loss = self._compute_boundary_loss(pred_masks, targets)
                    losses["loss_boundary"] = self.lambda_boundary * b_loss

            return losses

        # ── Inference: post-process ───────────────────────────────────────
        target_sizes = torch.tensor(
            [images.shape[-2:]], device=images.device
        ).repeat(images.shape[0], 1)
        return self.postprocessor(out, target_sizes)

    # ------------------------------------------------------------------
    # Mask loss helper  (CE + Dice on matched pairs, last decoder layer only)
    # ------------------------------------------------------------------

    def _compute_mask_losses(
        self,
        pred_masks: torch.Tensor,
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCE-CE and Dice mask losses on Hungarian-matched pairs.

        pred_masks : [B, N_q*group_detr, H_m, W_m]  (logits)
        targets    : list of dicts with 'boxes' (cxcywh norm), 'labels', 'masks'
        """
        device = pred_masks.device
        B = pred_masks.shape[0]

        # Greedy matching: for each GT mask find the best-matching predicted mask
        # (avoids re-running Hungarian matcher or storing extra tensors)
        src_masks_list = []
        tgt_masks_list = []

        for b in range(B):
            gt_m = targets[b].get('masks', None)
            if gt_m is None or len(gt_m) == 0:
                continue
            gt_m = gt_m.float().to(device)  # [M, H, W]
            pm = pred_masks[b]               # [N_q*g, H_m, W_m]

            # Resize pred to GT spatial size
            Hg, Wg = gt_m.shape[-2], gt_m.shape[-1]
            pm_r = F.interpolate(
                pm.unsqueeze(0), size=(Hg, Wg), mode='bilinear', align_corners=False
            ).squeeze(0)   # [N_q*g, Hg, Wg]

            # Greedy matching: for each GT find best-matching pred (max dot-product)
            with torch.no_grad():
                pm_sig = torch.sigmoid(pm_r)
                pm_flat = pm_sig.flatten(1)           # [N_q*g, HW]
                gt_flat = (gt_m > 0.5).float().flatten(1)  # [M, HW]
                scores  = torch.mm(pm_flat, gt_flat.t())   # [N_q*g, M]
                best_src = scores.argmax(dim=0)            # [M]

            matched_pred = pm_r[best_src]   # [M, Hg, Wg]
            src_masks_list.append(matched_pred)
            tgt_masks_list.append(gt_m)

        if len(src_masks_list) == 0:
            zero = pred_masks.sum() * 0.0
            return {'loss_mask_ce': zero, 'loss_mask_dice': zero}

        src_all = torch.cat(src_masks_list, dim=0)   # [N_total, Hg, Wg]
        tgt_all = torch.cat(tgt_masks_list, dim=0)   # [N_total, Hg, Wg]

        # BCE loss
        loss_ce = F.binary_cross_entropy_with_logits(
            src_all, tgt_all, reduction='mean'
        )

        # Dice loss
        src_sig = torch.sigmoid(src_all).flatten(1)
        tgt_f   = tgt_all.flatten(1)
        inter   = (src_sig * tgt_f).sum(-1)
        loss_dice = (1 - (2 * inter + 1) / (src_sig.sum(-1) + tgt_f.sum(-1) + 1)).mean()

        return {
            'loss_mask_ce'  : self.criterion.mask_ce_loss_coef   * loss_ce,
            'loss_mask_dice': self.criterion.mask_dice_loss_coef * loss_dice,
        }

    # ------------------------------------------------------------------
    # Boundary loss helper
    # ------------------------------------------------------------------

    def _compute_boundary_loss(
        self,
        pred_masks,
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        Compute BoundaryLoss on matched predicted vs GT masks.

        pred_masks: [B, N_queries, H_m, W_m]  (logits, dense tensor)
                    OR a dict from sparse_forward (skipped in that case)
        GT masks in targets[i]['masks']: [M_i, H, W]
        We use max-score matching (simple): take the max-logit pred per GT.
        """
        # Guard: sparse_forward returns dicts — skip boundary loss gracefully
        if isinstance(pred_masks, dict):
            return torch.tensor(0.0, requires_grad=True)

        losses = []
        B = pred_masks.shape[0]

        for b in range(B):
            gt_m = targets[b].get('masks', None)
            if gt_m is None or len(gt_m) == 0:
                continue

            gt_m = gt_m.float().to(pred_masks.device)   # [M, H, W]
            pm   = pred_masks[b]                          # [N_q, Hm, Wm]

            # Resize pred masks to GT spatial size
            Hg, Wg = gt_m.shape[-2], gt_m.shape[-1]
            pm_resized = F.interpolate(
                pm.unsqueeze(0), size=(Hg, Wg), mode='bilinear', align_corners=False
            ).squeeze(0)   # [N_q, Hg, Wg]

            pm_sig = torch.sigmoid(pm_resized)  # [N_q, Hg, Wg]

            # Match: for each GT mask find the pred with highest IoU (approx via dot)
            pm_flat = pm_sig.view(pm_sig.shape[0], -1)   # [N_q, HW]
            gt_flat = (gt_m > 0.5).float().view(gt_m.shape[0], -1)  # [M, HW]
            inter   = torch.mm(pm_flat, gt_flat.t())      # [N_q, M]
            best_pred_per_gt = inter.argmax(dim=0)         # [M]

            matched_preds = pm_sig[best_pred_per_gt]       # [M, Hg, Wg]
            b_loss = self.boundary_loss_fn(matched_preds, gt_m)
            losses.append(b_loss)

        if len(losses) == 0:
            return pred_masks.sum() * 0.0  # zero with grad attached to model params
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
    'nano'  : RFDETRNanoConfig,
    'small' : RFDETRSmallConfig,
    'medium': RFDETRMediumConfig,
    'base'  : RFDETRBaseConfig,
    'large' : RFDETRLargeConfig,
}


def build_dual_stream_rfdetr(
    num_classes: int = 1,
    variant: str = 'base',
    lambda_boundary: float = 0.5,
    depth_pretrained: bool = True,
    use_depth_generator: bool = True,
    depth_model_type: str = 'small',
    pretrain_weights: Optional[str] = None,
    freeze_encoder: bool = False,
    resolution: int = 1024,
    segmentation: bool = True,
    device: str = 'cuda',
) -> DualStreamLWDETR:
    """
    Build the Dual-Stream RF-DETR model.

    Args:
        num_classes     : number of foreground classes (1 for tree crown)
        variant         : 'nano' | 'small' | 'medium' | 'base' | 'large'
        lambda_boundary : boundary loss weight (0 = disabled)
        depth_pretrained: initialise depth ResNet-18 with ImageNet weights
        use_depth_generator: auto-generate depth via Depth Anything V2
        depth_model_type: Depth Anything V2 model size
        pretrain_weights: path to RF-DETR pretrained .pth file (None = random)
        freeze_encoder  : freeze DINOv2 weights
        resolution      : input resolution (default 1024 for aerial imagery)
        segmentation    : include segmentation head
        device          : 'cuda' or 'cpu'

    Returns:
        DualStreamLWDETR model
    """
    assert variant in _VARIANT_CONFIGS, \
        f"variant must be one of {list(_VARIANT_CONFIGS)}"

    cfg_cls = _VARIANT_CONFIGS[variant]
    cfg = cfg_cls(
        num_classes=num_classes,
        resolution=resolution,
        device=device,
        segmentation_head=segmentation,
        pretrain_weights=pretrain_weights,
    )

    # ── 1. Build stock RF-DETR backbone (DINOv2 + projector) ─────────────
    stock_joiner = build_backbone(
        encoder=cfg.encoder,
        vit_encoder_num_layers=cfg.out_feature_indexes[-1] + 1,
        pretrained_encoder=cfg.pretrain_weights,
        window_block_indexes=None,
        drop_path=0.0,
        out_channels=cfg.hidden_dim,
        out_feature_indexes=cfg.out_feature_indexes,
        projector_scale=cfg.projector_scale,
        use_cls_token=False,
        hidden_dim=cfg.hidden_dim,
        position_embedding='sine',
        freeze_encoder=freeze_encoder,
        layer_norm=cfg.layer_norm,
        target_shape=(resolution, resolution),
        rms_norm=getattr(cfg, 'rms_norm', False),
        backbone_lora=False,
        force_no_pretrain=(pretrain_weights is None),
        gradient_checkpointing=False,
        load_dinov2_weights=True,
        patch_size=cfg.patch_size,
        num_windows=cfg.num_windows,
        positional_encoding_size=cfg.positional_encoding_size,
    )
    # stock_joiner is Joiner(backbone=Backbone, position_embedding=...)
    stock_backbone    = stock_joiner[0]   # Backbone instance
    position_embedding = stock_joiner[1]

    # ── 2. Wrap backbone with depth stream + DAAF ─────────────────────────
    dual_backbone = DualStreamRFDETRBackbone(
        rfdetr_backbone=stock_backbone,
        depth_pretrained=depth_pretrained,
    )
    joiner = DualStreamJoiner(dual_backbone, position_embedding)

    # ── 3. Transformer ────────────────────────────────────────────────────
    # Build a minimal args namespace for build_transformer
    class _Args:
        pass
    args = _Args()
    args.hidden_dim          = cfg.hidden_dim
    args.num_feature_levels  = len(cfg.projector_scale)
    args.dec_layers          = cfg.dec_layers
    args.nheads              = cfg.sa_nheads
    args.sa_nheads           = cfg.sa_nheads
    args.ca_nheads           = cfg.ca_nheads
    args.dim_feedforward      = 2048
    args.dropout             = 0.0
    args.dec_n_points        = cfg.dec_n_points
    args.two_stage           = cfg.two_stage
    args.lite_refpoint_refine = cfg.lite_refpoint_refine
    args.bbox_reparam        = cfg.bbox_reparam
    args.num_queries         = cfg.num_queries if hasattr(cfg, 'num_queries') else 300
    args.group_detr          = cfg.group_detr
    args.use_cls_token       = False
    args.decoder_norm        = 'LN'   # 'LN' or 'Identity'
    args.segmentation_head   = segmentation   # used by build_criterion_and_postprocessors

    transformer = build_transformer(args)

    # ── 4. Segmentation head ──────────────────────────────────────────────
    seg_head = None
    if segmentation:
        seg_head = SegmentationHead(
            cfg.hidden_dim,
            cfg.dec_layers,
            downsample_ratio=cfg.mask_downsample_ratio,
        )

    # ── 5. Criterion & postprocessor ──────────────────────────────────────
    # We intentionally build the criterion WITHOUT mask losses.
    # Mask losses (CE + Dice) are computed manually in DualStreamLWDETR.forward
    # only for the final decoder layer, avoiding OOM from enc_outputs mask tensors
    # at 1024×1024 resolution with group_detr=13.
    args.num_classes         = num_classes
    args.focal_alpha         = 0.25
    args.cls_loss_coef       = cfg.cls_loss_coef
    args.bbox_loss_coef      = 5.0
    args.giou_loss_coef      = 2.0
    args.mask_ce_loss_coef   = 2.0
    args.mask_dice_loss_coef = 2.0
    args.aux_loss            = True
    args.ia_bce_loss         = getattr(cfg, 'ia_bce_loss', True)
    args.use_varifocal_loss  = False
    args.use_position_supervised_loss = False
    args.num_select          = getattr(cfg, 'num_select', 300)
    args.mask_point_sample_ratio = 16
    args.sum_group_losses    = False
    args.device              = device
    # Build criterion without mask losses (we handle masks separately)
    args.segmentation_head   = False   # disables mask losses in criterion

    # Matcher args
    args.set_cost_class      = 2.0
    args.set_cost_bbox       = 5.0
    args.set_cost_giou       = 2.0

    criterion, postprocessor = build_criterion_and_postprocessors(args)

    # Restore mask loss coefficients for use in weight_dict (added manually below)
    # These are stored on the criterion so train_rfdetr.py can use them for logging
    criterion.mask_ce_loss_coef   = args.mask_ce_loss_coef
    criterion.mask_dice_loss_coef = args.mask_dice_loss_coef
    if segmentation:
        criterion.weight_dict['loss_mask_ce']   = args.mask_ce_loss_coef
        criterion.weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef

    # ── 6. Load pretrained RF-DETR weights ────────────────────────────────
    if pretrain_weights and os.path.isfile(pretrain_weights):
        state = torch.load(pretrain_weights, map_location='cpu')
        if 'model' in state:
            state = state['model']
        # Load into LWDETR sub-module; new depth/DAAF weights won't exist → strict=False
        missing, unexpected = dual_backbone.load_state_dict(state, strict=False)
        print(f'[dual_stream_rfdetr] Loaded pretrained weights: '
              f'{len(missing)} missing, {len(unexpected)} unexpected keys')

    # ── 7. Assemble final model ───────────────────────────────────────────
    model = DualStreamLWDETR(
        joiner=joiner,
        transformer=transformer,
        segmentation_head=seg_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        criterion=criterion,
        postprocessor=postprocessor,
        lambda_boundary=lambda_boundary,
        use_depth_generator=use_depth_generator,
        depth_model_type=depth_model_type,
        aux_loss=True,
        group_detr=cfg.group_detr,
        two_stage=cfg.two_stage,
        lite_refpoint_refine=cfg.lite_refpoint_refine,
        bbox_reparam=cfg.bbox_reparam,
    )

    return model
