"""
Configuration for Dual-Stream Mask R-CNN with Depth-Aware Attention Fusion
for Tree Crown Segmentation
"""

class Config:
    # Image settings
    IMAGE_SIZE = 1024
    
    # Model settings
    NUM_CLASSES = 2  # background + tree
    
    # Backbone settings
    RGB_BACKBONE = 'resnet50'
    DEPTH_BACKBONE = 'resnet18'
    
    # Training settings
    BATCH_SIZE = 2
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 50
    WARMUP_EPOCHS = 3
    GRAD_CLIP_MAX_NORM = 1.0
    BACKBONE_LR_SCALE = 0.1

    # Loss weights
    LAMBDA_BOUNDARY = 0.5
    
    # RPN settings
    RPN_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
    RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5
    
    # ROI settings
    ROI_POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    
    # Data augmentation
    AUGMENTATION = True
    
    # Depth Anything V2 model
    DEPTH_MODEL = 'depth-anything-v2-small'
    
    # Device
    DEVICE = 'cuda'