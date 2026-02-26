"""
Visualize Depth Map with Colorbar
Creates a 4-panel visualization:
- Original image
- Depth map (grayscale)
- Depth map (colored with heatmap)
- Colorbar showing normalized height (0-1)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import cv2


def visualize_depth_map(
    image_path: str,
    depth_npy_path: str = None,
    output_path: str = None,
    colormap: str = 'inferno',
    figsize: tuple = (20, 5)
):
    """
    Visualize depth map with original image, grayscale, colored depth, and colorbar

    Args:
        image_path: Path to original image
        depth_npy_path: Path to depth .npy file (if None, will auto-detect)
        output_path: Path to save visualization (if None, will show)
        colormap: Matplotlib colormap name (inferno, viridis, plasma, etc.)
        figsize: Figure size (width, height)
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Auto-detect depth file if not provided
    if depth_npy_path is None:
        # Try to find corresponding depth file
        base_name = os.path.splitext(image_path)[0]
        depth_npy_path = base_name + '_depth.npy'

        if not os.path.exists(depth_npy_path):
            # Try in different directory structure
            img_dir = os.path.dirname(image_path)
            img_name = os.path.basename(image_path)
            depth_npy_path = os.path.join(
                img_dir,
                img_name.replace('.', '_depth.npy')
            )

    # Load depth map
    if not os.path.exists(depth_npy_path):
        raise FileNotFoundError(f"Depth file not found: {depth_npy_path}")

    depth_map = np.load(depth_npy_path)

    # Normalize depth to 0-1
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1. Original Image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Depth Grayscale
    axes[1].imshow(depth_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Depth Map (Grayscale)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Depth Colored
    im = axes[2].imshow(depth_norm, cmap=colormap, vmin=0, vmax=1)
    axes[2].set_title(f'Depth Map (Colored - {colormap})', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 4. Colorbar
    axes[3].axis('off')

    # Add colorbar on the right side
    cbar = plt.colorbar(
        im,
        ax=axes[3],
        orientation='vertical',
        fraction=0.8,
        pad=0.05
    )
    cbar.set_label('Normalized Height (0 = min, 1 = max)',
                   rotation=270,
                   labelpad=25,
                   fontsize=12,
                   fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add min/max values as text
    # axes[3].text(
    #     0.5, 0.3,
    #     f'Min depth: {depth_min:.2f}\nMax depth: {depth_max:.2f}',
    #     transform=axes[3].transAxes,
    #     fontsize=11,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_batch(
    image_dir: str,
    depth_dir: str,
    output_dir: str,
    num_samples: int = 5,
    colormap: str = 'inferno'
):
    """
    Visualize multiple depth maps

    Args:
        image_dir: Directory containing original images
        depth_dir: Directory containing depth .npy files
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        colormap: Matplotlib colormap name
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get image files
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))
    ]

    # Limit to num_samples
    image_files = image_files[:num_samples]

    print(f"Visualizing {len(image_files)} samples...")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Find corresponding depth file
        base_name = os.path.splitext(img_file)[0]
        depth_file = base_name + '_depth.npy'
        depth_path = os.path.join(depth_dir, depth_file)

        if not os.path.exists(depth_path):
            print(f"⚠️  Skipping {img_file}: depth file not found")
            continue

        # Output path
        out_file = base_name + '_visualization.png'
        out_path = os.path.join(output_dir, out_file)

        try:
            visualize_depth_map(
                image_path=img_path,
                depth_npy_path=depth_path,
                output_path=out_path,
                colormap=colormap
            )
        except Exception as e:
            print(f"⚠️  Failed to visualize {img_file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize depth maps with heatmap colorbar'
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single image to visualize'
    )

    parser.add_argument(
        '--depth', '-d',
        type=str,
        default=None,
        help='Path to corresponding depth .npy file (auto-detect if not provided)'
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
        '--depth-dir',
        type=str,
        help='Directory containing depth files (for batch processing)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./depth_visualizations',
        help='Output directory for batch visualizations'
    )

    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=5,
        help='Number of samples to visualize in batch mode'
    )

    parser.add_argument(
        '--colormap', '-c',
        type=str,
        default='inferno',
        choices=['inferno', 'viridis', 'plasma', 'magma', 'jet', 'turbo'],
        help='Colormap for depth visualization'
    )

    args = parser.parse_args()

    # Single image mode
    if args.image:
        visualize_depth_map(
            image_path=args.image,
            depth_npy_path=args.depth,
            output_path=args.output,
            colormap=args.colormap
        )

    # Batch mode
    elif args.image_dir and args.depth_dir:
        visualize_batch(
            image_dir=args.image_dir,
            depth_dir=args.depth_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            colormap=args.colormap
        )

    else:
        parser.print_help()
        print("\n❌ Please provide either:")
        print("   - Single image: --image <path>")
        print("   - Batch mode: --image-dir <dir> --depth-dir <dir>")
