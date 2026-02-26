"""
Loss Functions for Tree Crown Segmentation
Includes Boundary Loss to penalize incorrect predictions at tree crown edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SobelFilter(nn.Module):
    """
    Sobel edge detection filter
    """
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x):
        """
        Extract edges using Sobel operator
        
        Args:
            x: Input tensor [B, 1, H, W] or [B, H, W]
        Returns:
            edges: Edge magnitude [B, 1, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Compute gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return edges


class LaplacianFilter(nn.Module):
    """
    Laplacian edge detection filter
    """
    def __init__(self):
        super().__init__()
        
        # Laplacian kernel
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('laplacian', laplacian)
        
    def forward(self, x):
        """
        Extract edges using Laplacian operator
        
        Args:
            x: Input tensor [B, 1, H, W] or [B, H, W]
        Returns:
            edges: Edge response [B, 1, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        edges = F.conv2d(x, self.laplacian, padding=1)
        edges = torch.abs(edges)
        
        return edges


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for Tree Crown Segmentation
    
    Penalizes incorrect predictions at tree crown edges to ensure
    the model prioritizes separating touching crowns.
    
    L_bound = ||∂(pred) - ∂(gt)||
    
    where ∂ denotes the boundary extracted using Sobel or Laplacian operators
    """
    def __init__(self, edge_type='sobel', reduction='mean'):
        """
        Args:
            edge_type: 'sobel' or 'laplacian'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        
        if edge_type == 'sobel':
            self.edge_filter = SobelFilter()
        elif edge_type == 'laplacian':
            self.edge_filter = LaplacianFilter()
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")
            
        self.reduction = reduction
        
    def forward(self, pred_mask, gt_mask):
        """
        Compute boundary loss between predicted and ground truth masks
        
        Args:
            pred_mask: Predicted mask [B, 1, H, W] or [B, H, W], values in [0, 1]
            gt_mask: Ground truth mask [B, 1, H, W] or [B, H, W], values in {0, 1}
        Returns:
            loss: Boundary loss scalar
        """
        # Ensure correct shape
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(1)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
            
        # Ensure same dtype
        gt_mask = gt_mask.float()
        
        # Extract boundaries
        pred_boundary = self.edge_filter(pred_mask)
        gt_boundary = self.edge_filter(gt_mask)
        
        # Normalize boundaries
        pred_boundary = pred_boundary / (pred_boundary.max() + 1e-8)
        gt_boundary = gt_boundary / (gt_boundary.max() + 1e-8)
        
        # Compute loss (L1 or L2)
        loss = F.l1_loss(pred_boundary, gt_boundary, reduction=self.reduction)
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for mask segmentation
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted mask [B, 1, H, W], values in [0, 1]
            target: Ground truth mask [B, 1, H, W], values in {0, 1}
        Returns:
            loss: 1 - Dice coefficient
        """
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W] or probabilities
            target: Ground truth [B, H, W] or [B, 1, H, W]
        Returns:
            loss: Focal loss
        """
        if pred.dim() == 4 and pred.size(1) > 1:
            # Multi-class
            ce_loss = F.cross_entropy(pred, target.long().squeeze(1), reduction='none')
            pt = torch.exp(-ce_loss)
        else:
            # Binary
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            if target.dim() == 4:
                target = target.squeeze(1)
            ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
            pt = torch.exp(-ce_loss)
            
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedMaskLoss(nn.Module):
    """
    Combined loss for mask prediction:
    L_mask = L_bce + L_dice + λ * L_boundary
    """
    def __init__(self, lambda_boundary=0.5, edge_type='sobel'):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss(edge_type=edge_type)
        self.lambda_boundary = lambda_boundary
        
    def forward(self, pred_logits, gt_mask):
        """
        Args:
            pred_logits: Predicted mask logits [B, 1, H, W]
            gt_mask: Ground truth mask [B, 1, H, W]
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        pred_prob = torch.sigmoid(pred_logits)
        
        bce = self.bce_loss(pred_logits, gt_mask.float())
        dice = self.dice_loss(pred_prob, gt_mask)
        boundary = self.boundary_loss(pred_prob, gt_mask)
        
        total_loss = bce + dice + self.lambda_boundary * boundary
        
        loss_dict = {
            'loss_bce': bce,
            'loss_dice': dice,
            'loss_boundary': boundary,
            'loss_mask_total': total_loss
        }
        
        return total_loss, loss_dict


class TreeCrownSegmentationLoss(nn.Module):
    """
    Complete loss function for the Dual-Stream Mask R-CNN model
    
    L_total = L_cls + L_box + L_mask + λ * L_bound
    
    This ensures the model prioritizes separating touching crowns
    rather than solely optimizing Intersection over Union (IoU)
    """
    def __init__(self, lambda_boundary=0.5, edge_type='sobel'):
        """
        Args:
            lambda_boundary: Balancing coefficient for boundary loss (default: 0.5)
            edge_type: Type of edge detector ('sobel' or 'laplacian')
        """
        super().__init__()
        
        self.lambda_boundary = lambda_boundary
        self.boundary_loss = BoundaryLoss(edge_type=edge_type)
        
    def forward(
        self,
        classification_loss: torch.Tensor,
        box_regression_loss: torch.Tensor,
        mask_loss: torch.Tensor,
        pred_masks: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss
        
        Args:
            classification_loss: L_cls from RPN/classifier
            box_regression_loss: L_box from box regressor
            mask_loss: L_mask from mask head
            pred_masks: Predicted masks for boundary loss (optional)
            gt_masks: Ground truth masks for boundary loss (optional)
            
        Returns:
            total_loss: L_total
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {
            'loss_cls': classification_loss,
            'loss_box': box_regression_loss,
            'loss_mask': mask_loss,
        }
        
        # Base loss
        total_loss = classification_loss + box_regression_loss + mask_loss
        
        # Add boundary loss if masks are provided
        if pred_masks is not None and gt_masks is not None:
            boundary_loss = self.boundary_loss(pred_masks, gt_masks)
            loss_dict['loss_boundary'] = boundary_loss
            total_loss = total_loss + self.lambda_boundary * boundary_loss
        
        loss_dict['loss_total'] = total_loss
        
        return total_loss, loss_dict