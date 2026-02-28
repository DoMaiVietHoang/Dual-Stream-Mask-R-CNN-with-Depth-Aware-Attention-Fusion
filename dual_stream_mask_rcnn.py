"""
Dual-Stream Mask R-CNN with Depth-Aware Attention Fusion (DAAF)
Main model implementation for Tree Crown Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    maskrcnn_loss,
    maskrcnn_inference,
    fastrcnn_loss,
    project_masks_on_boxes,
)
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from modules.dual_stream_backbone import DualStreamBackbone
from modules.daaf import MultiScaleDAAF
from modules.depth_generator import DepthGenerator
from modules.losses import BoundaryLoss


class CustomRoIHeads(RoIHeads):
    """
    Custom RoIHeads that adds boundary loss to the standard mask loss.

    L_mask_total = L_mask_bce + lambda_boundary * L_boundary

    where L_boundary = ||∂(pred) - ∂(gt)|| using Sobel edge detection.
    All box detection logic is inherited unchanged from torchvision RoIHeads.
    """

    def __init__(self, *args, lambda_boundary=0.5, boundary_edge_type='sobel', **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_boundary = lambda_boundary
        self.boundary_loss_fn = BoundaryLoss(edge_type=boundary_edge_type)

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Identical to RoIHeads.forward() except the mask loss block,
        which adds boundary loss alongside standard BCE mask loss.
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {"boxes": boxes[i], "labels": labels[i], "scores": scores[i]}
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

                # Standard BCE mask loss
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )

                # Boundary loss for crown separation
                loss_boundary = self._compute_boundary_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )

                loss_mask = {
                    "loss_mask": rcnn_loss_mask,
                    "loss_boundary": self.lambda_boundary * loss_boundary,
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses

    def _compute_boundary_loss(self, mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        """
        Compute boundary loss between predicted and GT mask boundaries.

        Uses the same class-selection and target-projection as maskrcnn_loss,
        then applies Sobel edge detection + L1 loss.

        Args:
            mask_logits: (N, num_classes, H, W) raw logits from mask predictor
            proposals: list of (N_i, 4) positive proposal boxes per image
            gt_masks: list of (M_i, H_img, W_img) ground truth masks per image
            gt_labels: list of (M_i,) ground truth labels per image
            mask_matched_idxs: list of (N_i,) matched GT indices per image

        Returns:
            boundary_loss: scalar tensor
        """
        discretization_size = mask_logits.shape[-1]

        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            project_masks_on_boxes(m, p, i, discretization_size)
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels_cat = torch.cat(labels, dim=0)
        mask_targets_cat = torch.cat(mask_targets, dim=0)

        if mask_targets_cat.numel() == 0:
            return mask_logits.sum() * 0

        # Select per-class logits and apply sigmoid → [0, 1]
        idx = torch.arange(labels_cat.shape[0], device=labels_cat.device)
        pred_masks = torch.sigmoid(mask_logits[idx, labels_cat])  # (N, H, W)

        # BoundaryLoss handles [B, H, W] input (unsqueezes internally)
        boundary_loss = self.boundary_loss_fn(pred_masks, mask_targets_cat)

        return boundary_loss


class DualStreamBackboneWrapper(nn.Module):
    """
    Wrapper for Dual-Stream Backbone that integrates with torchvision's detection framework
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        
        self.backbone = DualStreamBackbone(pretrained=pretrained)
        rgb_channels, depth_channels = self.backbone.get_feature_channels()
        
        # Multi-scale DAAF for feature fusion
        self.daaf = MultiScaleDAAF(
            rgb_channels_list=rgb_channels,
            depth_channels_list=depth_channels,
            out_channels=out_channels
        )
        
        # Lateral connections for FPN
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1)
            for _ in range(4)
        ])
        
        # FPN top-down pathway
        self.fpn_output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(4)
        ])
        
        # Extra level for P5 -> P6
        self.extra_block = nn.MaxPool2d(2, stride=2)
        
        self.out_channels = out_channels
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: RGB images [B, 3, H, W]
            depth: Depth maps [B, 1, H, W]
        Returns:
            features: Dict of FPN features {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}
        """
        # Extract features from both streams
        rgb_features, depth_features = self.backbone(rgb, depth)
        
        # Convert to lists
        rgb_list = list(rgb_features.values())
        depth_list = list(depth_features.values())
        
        # Fuse features using DAAF
        fused_features = self.daaf(rgb_list, depth_list)
        
        # FPN: Build feature pyramid
        # Start from highest level (smallest resolution)
        fpn_features = []
        
        # Process from top to bottom
        prev_feature = None
        for i in range(3, -1, -1):
            lateral = self.fpn_lateral[i](fused_features[i])
            
            if prev_feature is not None:
                # Upsample and add
                upsampled = F.interpolate(
                    prev_feature, 
                    size=lateral.shape[2:], 
                    mode='nearest'
                )
                lateral = lateral + upsampled
                
            output = self.fpn_output[i](lateral)
            fpn_features.insert(0, output)
            prev_feature = lateral
        
        # Create output dict
        out = OrderedDict()
        for i, feat in enumerate(fpn_features):
            out[str(i)] = feat
            
        # Add extra pooled level
        out['pool'] = self.extra_block(fpn_features[-1])
        
        return out


class DualStreamMaskRCNN(nn.Module):
    """
    Dual-Stream Mask R-CNN with Depth-Aware Attention Fusion
    
    Architecture:
    1. Pseudo-Depth Generation (Depth Anything V2)
    2. Dual-Stream Encoder (ResNet-50 RGB + ResNet-18 Depth)
    3. Depth-Aware Attention Fusion (DAAF)
    4. Feature Pyramid Network
    5. RPN + ROI Heads (standard Mask R-CNN)
    
    Loss function:
    L_total = L_cls + L_box + L_mask + λ * L_bound
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # background + tree
        pretrained_backbone: bool = True,
        # Image settings
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        # RPN settings
        rpn_anchor_generator: Optional[AnchorGenerator] = None,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        # Box settings
        box_roi_pool: Optional[MultiScaleRoIAlign] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        # Mask settings
        mask_roi_pool: Optional[MultiScaleRoIAlign] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
        # Depth generator
        depth_model_type: str = 'small',
        # Loss settings
        lambda_boundary: float = 0.5,
    ):
        super().__init__()
        
        # Depth generator
        self.depth_generator = DepthGenerator(
            model_type=depth_model_type,
            pretrained=True
        )
        
        # Dual-stream backbone with DAAF
        out_channels = 256
        self.backbone = DualStreamBackboneWrapper(
            pretrained=pretrained_backbone,
            out_channels=out_channels
        )
        
        # Anchor generator tuned for tree crown detection on 1024x1024 aerial images.
        # Tree crowns are roughly circular (aspect ratio ~1.0) and span 32-256px.
        # Removing extreme anchors (512px) reduces false proposals on background.
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (96,), (128,), (256,))
            aspect_ratios = ((0.75, 1.0, 1.33),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # RPN head
        rpn_head = RPNHead(
            out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0]
        )
        
        # RPN
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train,
            testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train,
            testing=rpn_post_nms_top_n_test
        )
        
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )
        
        # Box ROI pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
        
        # Box head
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )
        
        # Box predictor
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes
            )
        
        # Mask ROI pooling — 28×28 for higher mask resolution
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=28,
                sampling_ratio=2
            )
        
        # Mask head
        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        
        # Mask predictor
        if mask_predictor is None:
            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(
                mask_predictor_in_channels,
                mask_dim_reduced,
                num_classes
            )
        
        # ROI heads with boundary loss
        self.roi_heads = CustomRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            None,  # bbox_reg_weights
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # Mask
            mask_roi_pool,
            mask_head,
            mask_predictor,
            # Boundary loss for crown separation
            lambda_boundary=lambda_boundary,
        )
        
        # Image normalization
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.max_size = max_size
        
        # Loss settings
        self.lambda_boundary = lambda_boundary
        self.num_classes = num_classes
        
    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize RGB images.
        Handles uint8 [0, 255] or float [0, 1] input.
        Applies ImageNet mean/std normalization.
        """
        # Convert to float [0, 1] first
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images.float() / 255.0

        mean = torch.tensor(self.image_mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        depth_maps: Optional[torch.Tensor] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Args:
            images: RGB images [B, 3, H, W], values in [0, 255] uint8 or [0, 1] float
            targets: List of dicts with 'boxes', 'labels', 'masks' (training only)
            depth_maps: Pre-computed depth maps [B, 1, H, W] in [0, 1] (optional)
            
        Returns:
            If training: losses dict
            If inference: list of prediction dicts
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        
        # Generate pseudo-depth if not provided
        if depth_maps is None:
            with torch.no_grad():
                depth_maps = self.depth_generator(images)
        
        # Normalize RGB images
        images_normalized = self.normalize_images(images)
        
        # Extract features using dual-stream backbone + DAAF
        features = self.backbone(images_normalized, depth_maps)
        
        # Create image list for RPN
        from torchvision.models.detection.image_list import ImageList
        image_sizes = [img.shape[-2:] for img in images]
        image_list = ImageList(images_normalized, image_sizes)
        
        if self.training:
            assert targets is not None
            
            # RPN
            proposals, proposal_losses = self.rpn(image_list, features, targets)
            
            # ROI heads
            detections, detector_losses = self.roi_heads(
                features, proposals, image_sizes, targets
            )
            
            # Combine losses
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            
            return losses
        else:
            # RPN
            proposals, _ = self.rpn(image_list, features, None)
            
            # ROI heads
            detections, _ = self.roi_heads(features, proposals, image_sizes, None)
            
            # Post-process detections
            detections = self.postprocess(detections, image_sizes, original_image_sizes)
            
            return detections
    
    def postprocess(
        self,
        detections: List[Dict[str, torch.Tensor]],
        image_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Paste masks into full image at bounding box locations and rescale boxes."""
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        for i, (det, im_size, orig_size) in enumerate(zip(detections, image_sizes, original_sizes)):
            # Scale boxes from image_sizes to original_sizes first
            if 'boxes' in det:
                boxes = det['boxes']
                scale_x = orig_size[1] / im_size[1]
                scale_y = orig_size[0] / im_size[0]
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                detections[i]['boxes'] = boxes

            # Paste 28x28 masks into full image at box locations
            if 'masks' in det:
                masks = det['masks']  # [N, 1, 28, 28] float probabilities
                boxes = det['boxes']  # [N, 4] already scaled to orig_size
                if masks.numel() > 0 and len(boxes) > 0:
                    # paste_masks_in_image expects [N, 1, H, W] float and boxes [N, 4]
                    # returns [N, 1, H_orig, W_orig] float
                    pasted = paste_masks_in_image(
                        masks, boxes, orig_size, padding=1
                    )
                    detections[i]['masks'] = pasted > 0.5
                else:
                    h, w = orig_size
                    detections[i]['masks'] = torch.zeros(
                        (0, 1, h, w), dtype=torch.bool, device=masks.device
                    )

        return detections


class TwoMLPHead(nn.Module):
    """Box head with two FC layers"""
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """Box predictor"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class MaskRCNNHeads(nn.Module):
    """Enhanced mask head with residual connections for better spatial refinement.

    Architecture: 4 conv blocks with a residual shortcut every 2 layers,
    operating at 28×28 (from mask_roi_pool output_size=28).
    """

    def __init__(self, in_channels, layers, dilation):
        super().__init__()
        self.blocks = nn.ModuleList()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(next_feature, layer_features, 3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm2d(layer_features),
                nn.ReLU(inplace=True),
            ))
            next_feature = layer_features

        # 1×1 projections for residual shortcuts (every 2 layers)
        self.shortcut_01 = nn.Conv2d(in_channels, layers[1], 1) if in_channels != layers[1] else nn.Identity()
        self.shortcut_23 = nn.Conv2d(layers[1], layers[3], 1) if layers[1] != layers[3] else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = x + self.shortcut_01(identity)  # residual after block 0-1

        identity = x
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        x = x + self.shortcut_23(identity)  # residual after block 2-3
        return x


class MaskRCNNPredictor(nn.Sequential):
    """Mask predictor with 2-step upsampling for higher resolution output.

    28×28 → 56×56 (deconv) → 56×56 (logits)
    """

    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(OrderedDict([
            ('conv5_mask', nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ('bn5', nn.BatchNorm2d(dim_reduced)),
            ('relu', nn.ReLU(inplace=True)),
            ('mask_fcn_logits', nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if 'weight' in name and 'bn' not in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    lambda_boundary: float = 0.5
) -> DualStreamMaskRCNN:
    """
    Build Dual-Stream Mask R-CNN model

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained backbone
        lambda_boundary: Weight for boundary loss

    Returns:
        model: DualStreamMaskRCNN instance
    """
    model = DualStreamMaskRCNN(
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        lambda_boundary=lambda_boundary,
        # For 1024x1024 images
        min_size=1024,
        max_size=1024,
        # NMS threshold: balance between separating touching crowns and reducing FP
        box_nms_thresh=0.5,
        # Max detections per image — typical dense tiles have ~30-50 crowns
        box_detections_per_img=100,
        # More RPN proposals kept after NMS so small/overlapping crowns survive
        rpn_post_nms_top_n_train=3000,
        rpn_post_nms_top_n_test=1500,
    )

    return model