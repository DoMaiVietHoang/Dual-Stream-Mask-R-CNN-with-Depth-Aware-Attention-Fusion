"""
Depth-Aware Attention Fusion (DAAF) Module
Core contribution: Fuses RGB and Depth features using attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Evaluates importance of each feature channel to determine 
    whether to prioritize spectral or depth information
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Channel attention weights [B, C, 1, 1]
        """
        b, c, _, _ = x.size()
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        # MLP
        y = self.mlp(y).view(b, c, 1, 1)
        return y


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on critical spatial regions, particularly canopy boundaries
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Spatial attention weights [B, 1, H, W]
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(concat))  # [B, 1, H, W]
        
        return attention


class DAAF(nn.Module):
    """
    Depth-Aware Attention Fusion Module
    
    Fuses RGB features (F_rgb) and Depth features (F_d) using:
    1. Channel Attention (M_c): Determines importance of each feature channel
    2. Spatial Attention (M_s): Focuses on boundary regions
    
    Final fusion formula:
        F_fused = F_rgb * (1 + M_c) + F_d * M_s
    """
    def __init__(self, rgb_channels, depth_channels, out_channels, reduction_ratio=16):
        """
        Args:
            rgb_channels: Number of channels in RGB feature map
            depth_channels: Number of channels in Depth feature map
            out_channels: Number of output channels
            reduction_ratio: Reduction ratio for channel attention MLP
        """
        super().__init__()
        
        # 1x1 Conv to align channels after concatenation
        self.conv_cat = nn.Conv2d(
            rgb_channels + depth_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention modules
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Project RGB and Depth to same channels if needed
        self.rgb_proj = nn.Conv2d(rgb_channels, out_channels, 1) if rgb_channels != out_channels else nn.Identity()
        self.depth_proj = nn.Conv2d(depth_channels, out_channels, 1) if depth_channels != out_channels else nn.Identity()
        
    def forward(self, f_rgb, f_d):
        """
        Args:
            f_rgb: RGB feature map [B, C_rgb, H, W]
            f_d: Depth feature map [B, C_d, H, W]
        Returns:
            f_fused: Fused feature map [B, C_out, H, W]
        """
        # Handle size mismatch (depth might be different resolution)
        if f_rgb.shape[2:] != f_d.shape[2:]:
            f_d = F.interpolate(f_d, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and process: F_cat = Conv_1x1([F_rgb || F_d])
        f_cat = torch.cat([f_rgb, f_d], dim=1)
        f_cat = self.relu(self.bn(self.conv_cat(f_cat)))
        
        # Compute attention weights
        m_c = self.channel_attention(f_cat)  # [B, C, 1, 1]
        m_s = self.spatial_attention(f_cat)  # [B, 1, H, W]
        
        # Project RGB and Depth to same channels
        f_rgb_proj = self.rgb_proj(f_rgb)
        f_d_proj = self.depth_proj(f_d)
        
        # Handle size mismatch after projection
        if f_rgb_proj.shape[2:] != f_d_proj.shape[2:]:
            f_d_proj = F.interpolate(f_d_proj, size=f_rgb_proj.shape[2:], mode='bilinear', align_corners=False)
        
        # Fusion formula: F_fused = F_rgb * (1 + M_c) + F_d * M_s
        f_fused = f_rgb_proj * (1 + m_c) + f_d_proj * m_s
        
        return f_fused


class MultiScaleDAAF(nn.Module):
    """
    Multi-scale DAAF for FPN-style feature fusion
    Applies DAAF at each pyramid level
    """
    def __init__(self, rgb_channels_list, depth_channels_list, out_channels=256):
        """
        Args:
            rgb_channels_list: List of channels for each RGB feature level [C1, C2, C3, C4]
            depth_channels_list: List of channels for each Depth feature level
            out_channels: Output channels for all levels
        """
        super().__init__()
        
        self.daaf_modules = nn.ModuleList([
            DAAF(rgb_c, depth_c, out_channels)
            for rgb_c, depth_c in zip(rgb_channels_list, depth_channels_list)
        ])
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: List of RGB feature maps at different scales
            depth_features: List of Depth feature maps at different scales
        Returns:
            fused_features: List of fused feature maps
        """
        fused_features = []
        for daaf, f_rgb, f_d in zip(self.daaf_modules, rgb_features, depth_features):
            fused = daaf(f_rgb, f_d)
            fused_features.append(fused)
        
        return fused_features