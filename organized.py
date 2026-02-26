"""
Script to organize depth files for training
Renames and copies depth files to match dataset structure
"""
import os
import shutil
import argparse
from pathlib import Path

def organize_depth_files(
    depth_source_dir: str,
    rgb_image_dir: str,
    output_depth_dir: str,
    depth_extension: str = '.npytif.npy',
    copy_files: bool = True
):
    """
    Organize depth files to match RGB images

    Args:
        depth_source_dir: Source directory containing depth files
        rgb_image_dir: Directory containing RGB images
        output_depth_dir: Output directory for organized depth files
        depth_extension: Extension of depth files to process
        copy_files: If True, copy files; if False, move files
    """

    # Create output directory
    os.makedirs(output_depth_dir, exist_ok=True)

    # Get all RGB image files
    rgb_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        rgb_files.extend(Path(rgb_image_dir).glob(f'*{ext}'))

    rgb_basenames = {f.stem: f.suffix for f in rgb_files}

    print(f"Found {len(rgb_basenames)} RGB images in {rgb_image_dir}")

    # Get all depth files
    depth_files = list(Path(depth_source_dir).glob(f'*{depth_extension}'))
    print(f"Found {len(depth_files)} depth files in {depth_source_dir}")

    if len(depth_files) == 0:
        print(f"\n⚠️  No files with extension '{depth_extension}' found!")
        print(f"Available files in {depth_source_dir}:")
        all_files = list(Path(depth_source_dir).iterdir())
        for f in all_files[:10]:
            print(f"  - {f.name}")
        return

    processed = 0
    skipped = 0

    print(f"\nProcessing depth files...")
    print("="*70)

    for depth_file in depth_files:
        # Extract base name
        # Example: Stadtwald_31_1815_depth.npytif.npy -> Stadtwald_31_1815
        filename = depth_file.name

        # Remove depth_extension
        base = filename.replace(depth_extension, '')

        # Remove '_depth' if present
        if base.endswith('_depth'):
            base = base[:-6]

        # Check if this RGB image exists
        if base in rgb_basenames:
            # Create new filename: base_depth.npy
            new_filename = f"{base}_depth.npy"
            output_path = os.path.join(output_depth_dir, new_filename)

            # Copy or move file
            if copy_files:
                shutil.copy2(depth_file, output_path)
                action = "Copied"
            else:
                shutil.move(str(depth_file), output_path)
                action = "Moved"

            print(f"✓ {action}: {filename}")
            print(f"    → {new_filename}")
            processed += 1
        else:
            print(f"⚠️  Skipped: {filename} (no matching RGB image)")
            skipped += 1

    print("="*70)
    print(f"\n✓ Processed: {processed} files")
    print(f"⚠️  Skipped: {skipped} files")
    print(f"\nOutput directory: {output_depth_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Organize depth files to match dataset structure'
    )

    parser.add_argument(
        '--depth-source',
        type=str,
        required=True,
        help='Source directory containing depth files (e.g., Depth_dataset/BAMFOREST/train)'
    )

    parser.add_argument(
        '--rgb-images',
        type=str,
        required=True,
        help='Directory containing RGB images (e.g., dataset/images/train)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized depth files (e.g., dataset/depth/train)'
    )

    parser.add_argument(
        '--depth-ext',
        type=str,
        default='.npytif.npy',
        help='Extension of depth files to process (default: .npytif.npy)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying'
    )

    args = parser.parse_args()

    print("="*70)
    print("Depth File Organizer")
    print("="*70)
    print(f"Depth source: {args.depth_source}")
    print(f"RGB images:   {args.rgb_images}")
    print(f"Output:       {args.output}")
    print(f"Depth ext:    {args.depth_ext}")
    print(f"Action:       {'MOVE' if args.move else 'COPY'}")
    print("="*70)

    # Check if directories exist
    if not os.path.exists(args.depth_source):
        print(f"❌ Error: Depth source directory not found: {args.depth_source}")
        return

    if not os.path.exists(args.rgb_images):
        print(f"❌ Error: RGB images directory not found: {args.rgb_images}")
        return

    # Confirm action
    if args.move:
        response = input("\n⚠️  Files will be MOVED (not copied). Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    organize_depth_files(
        depth_source_dir=args.depth_source,
        rgb_image_dir=args.rgb_images,
        output_depth_dir=args.output,
        depth_extension=args.depth_ext,
        copy_files=not args.move
    )


if __name__ == '__main__':
    main()
