"""
Pseudo-Depth Generation Module
Uses Depth Anything V2 to generate depth maps from RGB images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings


class DepthGenerator(nn.Module):
    """
    Generates pseudo-depth maps from RGB images using Depth Anything V2
    
    The depth map provides critical structural cues to distinguish 
    adjacent tree crowns that may appear similar in color/texture.
    """
    
    def __init__(self, model_type='small', device='cuda', pretrained=True):
        """
        Args:
            model_type: 'small', 'base', or 'large' for Depth Anything V2 variants
            device: Device to run the model on
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.pretrained = pretrained
        self.depth_model = None
        self.transform = None
        
    def _load_model(self):
        """Lazy loading of the depth model"""
        if self.depth_model is not None:
            return
            
        try:
            # Try loading Depth Anything V2 from torch hub
            self.depth_model = torch.hub.load(
                'depth-anything/Depth-Anything-V2',
                f'depth_anything_v2_vit{self.model_type[0]}',  # vits, vitb, vitl
                pretrained=self.pretrained
            )
            self.depth_model = self.depth_model.to(self.device)
            self.depth_model.eval()
        except Exception as e:
            warnings.warn(f"Could not load Depth Anything V2: {e}. Using fallback depth estimation.")
            self.depth_model = FallbackDepthEstimator()
            self.depth_model = self.depth_model.to(self.device)
    
    def normalize_depth(self, depth_raw):
        """
        Apply min-max normalization to scale depth to [0, 1]
        
        D_norm = (D_raw - min(D_raw)) / (max(D_raw) - min(D_raw))
        
        Args:
            depth_raw: Raw depth map [B, 1, H, W] or [B, H, W]
        Returns:
            depth_norm: Normalized depth map [B, 1, H, W]
        """
        if depth_raw.dim() == 3:
            depth_raw = depth_raw.unsqueeze(1)
            
        # Per-sample normalization
        b = depth_raw.shape[0]
        depth_norm = torch.zeros_like(depth_raw)
        
        for i in range(b):
            d = depth_raw[i]
            d_min = d.min()
            d_max = d.max()
            
            if d_max - d_min > 1e-8:
                depth_norm[i] = (d - d_min) / (d_max - d_min)
            else:
                depth_norm[i] = torch.zeros_like(d)
        
        return depth_norm
    
    @torch.no_grad()
    def forward(self, rgb_images, target_size=None):
        """
        Generate depth maps from RGB images
        
        Args:
            rgb_images: RGB images [B, 3, H, W] in range [0, 1] or [0, 255]
            target_size: Optional (H, W) to resize output depth maps
        Returns:
            depth_norm: Normalized depth maps [B, 1, H, W] in range [0, 1]
        """
        self._load_model()
        
        # Ensure input is in correct range [0, 255]
        if rgb_images.max() <= 1.0:
            rgb_images = rgb_images * 255.0
        
        # Get original size
        original_size = rgb_images.shape[2:]
        
        # Generate depth
        if isinstance(self.depth_model, FallbackDepthEstimator):
            depth_raw = self.depth_model(rgb_images)
        else:
            # Depth Anything V2 expects RGB in [0, 255]
            depth_raw = self.depth_model(rgb_images)
        
        # Ensure correct shape
        if depth_raw.dim() == 3:
            depth_raw = depth_raw.unsqueeze(1)
        
        # Resize to target size if specified
        if target_size is not None and depth_raw.shape[2:] != target_size:
            depth_raw = F.interpolate(
                depth_raw, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        elif depth_raw.shape[2:] != original_size:
            depth_raw = F.interpolate(
                depth_raw,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize to [0, 1]
        depth_norm = self.normalize_depth(depth_raw)
        
        return depth_norm


class FallbackDepthEstimator(nn.Module):
    """
    Simple fallback depth estimator using image gradients
    Used when Depth Anything V2 is not available
    """
    def __init__(self):
        super().__init__()
        
        # Simple encoder-decoder for depth estimation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Args:
            x: RGB images [B, 3, H, W]
        Returns:
            depth: Estimated depth [B, 1, H, W]
        """
        # Normalize input
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.encoder(x)
        depth = self.decoder(features)
        
        # Resize to input size
        if depth.shape[2:] != x.shape[2:]:
            depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return depth


class DepthPreprocessor:
    """
    Preprocessing utilities for depth maps
    """
    
    @staticmethod
    def apply_colormap(depth, colormap='magma'):
        """
        Apply colormap to depth for visualization
        
        Args:
            depth: Depth map [H, W] or [B, 1, H, W] in [0, 1]
            colormap: Colormap name ('magma', 'viridis', 'plasma', etc.)
        Returns:
            colored_depth: RGB visualization [H, W, 3] or [B, 3, H, W]
        """
        import matplotlib.pyplot as plt
        
        cmap = plt.get_cmap(colormap)
        
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().cpu().numpy()
        else:
            depth_np = depth
            
        if depth_np.ndim == 4:
            # [B, 1, H, W] -> [B, H, W]
            depth_np = depth_np.squeeze(1)
            
        colored = cmap(depth_np)[..., :3]  # Remove alpha channel
        
        if isinstance(depth, torch.Tensor):
            colored = torch.from_numpy(colored).permute(0, 3, 1, 2).float()
            
        return colored
    
    @staticmethod
    def edge_enhance(depth, sigma=1.0):
        """
        Enhance edges in depth map using Sobel operator
        
        Args:
            depth: Depth map [B, 1, H, W]
            sigma: Gaussian smoothing sigma
        Returns:
            enhanced: Edge-enhanced depth [B, 1, H, W]
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        enhanced = depth + 0.5 * edges
        
        # Normalize
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
        
        return enhanced