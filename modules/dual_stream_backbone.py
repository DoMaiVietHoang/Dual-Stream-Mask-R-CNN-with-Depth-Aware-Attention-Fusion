"""
Dual-Stream Encoder Backbone
RGB Stream: ResNet-50 for high-fidelity color and texture features
Depth Stream: ResNet-18 (lighter) for geometric features and boundaries
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from typing import Dict, List


class RGBStream(nn.Module):
    """
    RGB Feature Extraction Stream using ResNet-50
    Extracts high-fidelity features regarding color and leaf texture
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        
        # Extract layers for FPN
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 256 channels, stride 4
        self.layer2 = resnet.layer2  # Output: 512 channels, stride 8
        self.layer3 = resnet.layer3  # Output: 1024 channels, stride 16
        self.layer4 = resnet.layer4  # Output: 2048 channels, stride 32
        
        # Output channels at each stage
        self.out_channels = [256, 512, 1024, 2048]
        
    def forward(self, x):
        """
        Args:
            x: RGB image [B, 3, H, W]
        Returns:
            features: Dict of feature maps at each stage
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # stride 4
        c2 = self.layer2(c1)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32
        
        return OrderedDict([
            ('feat1', c1),
            ('feat2', c2),
            ('feat3', c3),
            ('feat4', c4),
        ])


class DepthStream(nn.Module):
    """
    Depth Feature Extraction Stream using ResNet-18 (lighter network)
    Extracts geometric features and object boundaries
    
    Using a shallower network reduces computational cost while maintaining 
    efficiency, as depth maps typically contain fewer high-frequency details
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Modify first conv to accept 1-channel depth input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize with average of pretrained RGB weights
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 64 channels, stride 4
        self.layer2 = resnet.layer2  # Output: 128 channels, stride 8
        self.layer3 = resnet.layer3  # Output: 256 channels, stride 16
        self.layer4 = resnet.layer4  # Output: 512 channels, stride 32
        
        # Output channels at each stage
        self.out_channels = [64, 128, 256, 512]
        
    def forward(self, x):
        """
        Args:
            x: Depth map [B, 1, H, W]
        Returns:
            features: Dict of feature maps at each stage
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # stride 4
        c2 = self.layer2(c1)  # stride 8
        c3 = self.layer3(c2)  # stride 16
        c4 = self.layer4(c3)  # stride 32
        
        return OrderedDict([
            ('feat1', c1),
            ('feat2', c2),
            ('feat3', c3),
            ('feat4', c4),
        ])


class DualStreamBackbone(nn.Module):
    """
    Dual-Stream Encoder combining RGB and Depth streams
    
    Instead of stacking RGB and Depth channels at input (Input Fusion),
    which often dilutes distinct characteristics, we use separate branches.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.rgb_stream = RGBStream(pretrained=pretrained)
        self.depth_stream = DepthStream(pretrained=pretrained)
        
        # Channel information for FPN/DAAF
        self.rgb_channels = self.rgb_stream.out_channels
        self.depth_channels = self.depth_stream.out_channels
        
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image [B, 3, H, W]
            depth: Depth map [B, 1, H, W]
        Returns:
            rgb_features: Dict of RGB feature maps
            depth_features: Dict of Depth feature maps
        """
        rgb_features = self.rgb_stream(rgb)
        depth_features = self.depth_stream(depth)
        
        return rgb_features, depth_features
    
    def get_feature_channels(self):
        """Return channel counts for RGB and Depth streams"""
        return self.rgb_channels, self.depth_channels


class DualStreamBackboneWithFPN(nn.Module):
    """
    Dual-Stream Backbone with Feature Pyramid Network
    Integrates DAAF for multi-scale feature fusion
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        
        from .daaf import MultiScaleDAAF
        
        self.backbone = DualStreamBackbone(pretrained=pretrained)
        rgb_channels, depth_channels = self.backbone.get_feature_channels()
        
        # Multi-scale DAAF for feature fusion
        self.daaf = MultiScaleDAAF(
            rgb_channels_list=rgb_channels,
            depth_channels_list=depth_channels,
            out_channels=out_channels
        )
        
        # FPN on fused features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels] * 4,
            out_channels=out_channels,
            extra_blocks=None  # Can add LastLevelMaxPool if needed
        )
        
        self.out_channels = out_channels
        
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image [B, 3, H, W]
            depth: Depth map [B, 1, H, W]
        Returns:
            fpn_features: Dict of FPN feature maps
        """
        # Extract features from both streams
        rgb_features, depth_features = self.backbone(rgb, depth)
        
        # Convert to lists for DAAF
        rgb_list = list(rgb_features.values())
        depth_list = list(depth_features.values())
        
        # Fuse features using DAAF
        fused_features = self.daaf(rgb_list, depth_list)
        
        # Create OrderedDict for FPN
        fused_dict = OrderedDict([
            (f'feat{i}', feat) for i, feat in enumerate(fused_features)
        ])
        
        # Apply FPN
        fpn_features = self.fpn(fused_dict)
        
        return fpn_features