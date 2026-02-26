"""
Visualize Mask R-CNN Inference Results
Shows original image, predicted boxes, masks, and overlays
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from PIL import Image
import cv2
from typing import List, Dict

from dual_stream_mask_rcnn import build_model


def visualize_predictions(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    score_threshold: float = 0.5,
    output_path: str = None,
    show_scores: bool = True,
    figsize: tuple = (20, 5)
):
    """
    Visualize model predictions with masks overlay

    Args:
        image_path: Path to input image
        model: Trained model
        device: Device (cuda/cpu)
        score_threshold: Minimum confidence score to display
        output_path: Path to save visualization (if None, will show)
        show_scores: Whether to show confidence scores on boxes
        figsize: Figure size (width, height)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Prepare input
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Get first image predictions
    pred = predictions[0]

    # Filter by score threshold
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    masks = pred['masks'].cpu().numpy() if 'masks' in pred else None

    keep_idx = scores >= score_threshold
    scores = scores[keep_idx]
    boxes = boxes[keep_idx]
    labels = labels[keep_idx]
    if masks is not None:
        masks = masks[keep_idx]

    print(f"Found {len(scores)} predictions with score >= {score_threshold}")

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1. Original Image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Bounding Boxes
    axes[1].imshow(image_np)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[1].add_patch(rect)

        if show_scores:
            axes[1].text(
                x1, y1 - 5,
                f'{score:.2f}',
                color='red',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
    axes[1].set_title(f'Bounding Boxes ({len(boxes)} detections)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Instance Masks
    if masks is not None and len(masks) > 0:
        # Create colored instance mask
        instance_mask = np.zeros((*image_np.shape[:2], 3), dtype=np.uint8)

        # Generate random colors for each instance
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(masks), 3))

        for i, mask in enumerate(masks):
            # Mask is [1, H, W], take first channel and threshold
            mask_binary = mask[0] > 0.5
            instance_mask[mask_binary] = colors[i]

        axes[2].imshow(instance_mask)
        axes[2].set_title('Instance Masks', fontsize=14, fontweight='bold')
    else:
        axes[2].imshow(image_np)
        axes[2].text(
            0.5, 0.5, 'No masks predicted',
            transform=axes[2].transAxes,
            ha='center', va='center',
            fontsize=16, color='red'
        )
        axes[2].set_title('Instance Masks', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 4. Overlay (Image + Masks + Boxes)
    if masks is not None and len(masks) > 0:
        # Create overlay
        overlay = image_np.copy()

        # Add colored masks with transparency
        for i, mask in enumerate(masks):
            mask_binary = mask[0] > 0.5
            color_mask = np.zeros_like(image_np)
            color_mask[mask_binary] = colors[i]
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

        axes[3].imshow(overlay)

        # Add boxes on top
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='white', facecolor='none'
            )
            axes[3].add_patch(rect)

            if show_scores:
                axes[3].text(
                    x1, y1 - 5,
                    f'Tree {i+1}: {score:.2f}',
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
                )
    else:
        axes[3].imshow(image_np)

    axes[3].set_title('Overlay (Image + Masks + Boxes)', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()

    return {
        'num_detections': len(scores),
        'scores': scores,
        'boxes': boxes,
        'labels': labels
    }


def visualize_batch(
    image_dir: str,
    model_path: str,
    output_dir: str,
    num_samples: int = 5,
    score_threshold: float = 0.5,
    device: str = 'cuda'
):
    """
    Visualize predictions on multiple images

    Args:
        image_dir: Directory containing images
        model_path: Path to trained model checkpoint
        output_dir: Directory to save visualizations
        num_samples: Number of images to process
        score_threshold: Minimum confidence score
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")

    model = build_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    # Load state dict with strict=False to ignore mismatched keys
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✅ Loaded checkpoint with strict mode")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {e}")
        print("🔄 Trying with strict=False...")

        # Filter out depth_generator keys if they cause issues
        state_dict = checkpoint['model_state_dict']
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        # Find mismatched keys
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys:
            print(f"  Missing keys in checkpoint: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys in checkpoint: {len(unexpected_keys)}")
            # Remove depth_generator keys if they're the problem
            if any('depth_generator' in k for k in unexpected_keys):
                print("  Removing depth_generator keys from checkpoint...")
                state_dict = {k: v for k, v in state_dict.items()
                             if not k.startswith('depth_generator.depth_model.')}

        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded checkpoint with strict=False")

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Get image files
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))
    ]

    # Limit to num_samples
    image_files = image_files[:num_samples]

    print(f"\nProcessing {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Output path
        base_name = os.path.splitext(img_file)[0]
        out_file = base_name + '_prediction.png'
        out_path = os.path.join(output_dir, out_file)

        print(f"\nProcessing: {img_file}")

        try:
            results = visualize_predictions(
                image_path=img_path,
                model=model,
                device=device,
                score_threshold=score_threshold,
                output_path=out_path,
                show_scores=True
            )

            print(f"  Detections: {results['num_detections']}")
            if results['num_detections'] > 0:
                print(f"  Score range: {results['scores'].min():.3f} - {results['scores'].max():.3f}")

        except Exception as e:
            print(f"  ⚠️ Failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize Mask R-CNN inference results'
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single image to visualize'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save visualization (show if not provided)'
    )

    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images (for batch processing)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_visualizations',
        help='Output directory for batch visualizations'
    )

    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=5,
        help='Number of samples to visualize in batch mode'
    )

    parser.add_argument(
        '--score-threshold', '-s',
        type=float,
        default=0.5,
        help='Minimum confidence score to display predictions'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Single image mode
    if args.image:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading model from {args.model}...")
        model = build_model(num_classes=2, pretrained=False)
        checkpoint = torch.load(args.model, map_location=device)

        # Load state dict with error handling
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✅ Loaded checkpoint with strict mode")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed, trying with strict=False...")

            # Filter out problematic keys
            state_dict = checkpoint['model_state_dict']
            if any('depth_generator.depth_model.' in k for k in state_dict.keys()):
                print("  Removing depth_generator.depth_model keys from checkpoint...")
                state_dict = {k: v for k, v in state_dict.items()
                             if not k.startswith('depth_generator.depth_model.')}

            model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded checkpoint with strict=False")

        model = model.to(device)

        print(f"Model loaded on {device}")

        # Visualize
        results = visualize_predictions(
            image_path=args.image,
            model=model,
            device=device,
            score_threshold=args.score_threshold,
            output_path=args.output,
            show_scores=True
        )

        print(f"\nResults:")
        print(f"  Detections: {results['num_detections']}")
        if results['num_detections'] > 0:
            print(f"  Score range: {results['scores'].min():.3f} - {results['scores'].max():.3f}")

    # Batch mode
    elif args.image_dir:
        visualize_batch(
            image_dir=args.image_dir,
            model_path=args.model,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            score_threshold=args.score_threshold,
            device=args.device
        )

    else:
        parser.print_help()
        print("\n❌ Please provide either:")
        print("   - Single image: --image <path> --model <model.pth>")
        print("   - Batch mode: --image-dir <dir> --model <model.pth>")
