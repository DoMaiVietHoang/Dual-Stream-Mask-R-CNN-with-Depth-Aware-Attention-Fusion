"""
Dataset for Tree Crown Segmentation
Supports 1024x1024 images with instance segmentation annotations
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from typing import Dict, List, Tuple, Optional, Callable
import cv2


class TreeCrownDataset(Dataset):
    """
    Dataset for Tree Crown Instance Segmentation
    
    Expected structure:
    data_root/
        images/
            train/
                image1.png
                image2.png
            val/
                ...
        annotations/
            train.json  (COCO format)
            val.json
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 1024,
        transforms: Optional[Callable] = None,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train' or 'val'
            image_size: Target image size (default 1024)
            transforms: Custom transforms (if None, default transforms are used)
            use_augmentation: Whether to use data augmentation (training only)
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and split == 'train'
        
        # Load COCO annotations
        ann_file = os.path.join(data_root, 'annotations', f'{split}.json')
        if os.path.exists(ann_file):
            self.coco = COCO(ann_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            # Fallback: load images without annotations
            self.coco = None
            img_dir = os.path.join(data_root, 'images', split)
            self.image_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(img_dir)
                if f.endswith(('.png', '.jpg', '.tif'))
            ]

        # Validate image files exist
        self._validate_images()

        # Set up transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._get_default_transforms()
            
    def _validate_images(self):
        """Validate that image files exist and are readable"""
        valid_ids = []
        invalid_count = 0
        corrupt_count = 0

        print(f"Validating {len(self.image_ids)} images...")

        for image_id in self.image_ids:
            # Get image path
            if self.coco is not None:
                img_info = self.coco.loadImgs(image_id)[0]
                img_path = os.path.join(
                    self.data_root, 'images', self.split, img_info['file_name']
                )
            else:
                img_path = os.path.join(
                    self.data_root, 'images', self.split, f'{image_id}.png'
                )

            # Try to actually load the image to verify it's not corrupted
            valid = False
            paths_to_try = [img_path]

            # Add alternative extensions
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                alt_path = os.path.join(
                    self.data_root, 'images', self.split, f'{base_name}{ext}'
                )
                if alt_path not in paths_to_try:
                    paths_to_try.append(alt_path)

            for path in paths_to_try:
                if os.path.exists(path):
                    try:
                        # Actually try to load the image
                        img = Image.open(path)
                        img.load()  # Force load to catch truncated images
                        img.convert('RGB')
                        img.close()
                        valid = True
                        break
                    except (OSError, IOError, Exception) as e:
                        # Image is corrupted or can't be loaded
                        corrupt_count += 1
                        if corrupt_count <= 5:
                            print(f"⚠️  Skipping corrupted file: {path} ({type(e).__name__})")
                        continue

            if valid:
                valid_ids.append(image_id)
            else:
                invalid_count += 1
                if invalid_count + corrupt_count <= 10:
                    print(f"⚠️  Skipping invalid file: {img_path}")

        self.image_ids = valid_ids

        if invalid_count > 0:
            print(f"⚠️  Skipped {invalid_count} missing images")
        if corrupt_count > 0:
            print(f"⚠️  Skipped {corrupt_count} corrupted images")
        print(f"✓ {len(valid_ids)} valid images loaded")

    def _get_default_transforms(self) -> A.Compose:
        """Get default augmentation pipeline.
        Note: Normalization (ImageNet mean/std) is handled in the model's
        normalize_images() to avoid double normalization. Images output
        as uint8 [0, 255] tensors via ToTensorV2.
        """
        if self.use_augmentation:
            return A.Compose([
                # Always resize to target size first, then apply augmentations
                A.Resize(self.image_size, self.image_size),
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ], p=0.2),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _load_image(self, image_id) -> np.ndarray:
        """Load image as numpy array"""
        if self.coco is not None:
            img_info = self.coco.loadImgs(image_id)[0]
            img_path = os.path.join(
                self.data_root, 'images', self.split, img_info['file_name']
            )
        else:
            img_path = os.path.join(
                self.data_root, 'images', self.split, f'{image_id}.png'
            )

        # Try to load image, with fallback to other extensions
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        paths_to_try = [img_path]

        # Add alternative extensions
        for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
            alt_path = os.path.join(
                self.data_root, 'images', self.split, f'{base_name}{ext}'
            )
            if alt_path not in paths_to_try:
                paths_to_try.append(alt_path)

        last_error = None
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    # Enable loading truncated images as a fallback
                    from PIL import ImageFile
                    ImageFile.LOAD_TRUNCATED_IMAGES = True

                    image = Image.open(path)
                    image.load()  # Force load
                    image = image.convert('RGB')
                    return np.array(image)
                except (OSError, IOError, FileNotFoundError) as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue

        # If no valid image found, raise the last error or a generic one
        if last_error:
            raise OSError(f"Could not load image {img_path}: {last_error}")
        else:
            raise FileNotFoundError(f"Image not found: {img_path}")
    
    def _load_annotations(self, image_id) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load annotations for an image
        
        Returns:
            boxes: [N, 4] in xyxy format
            masks: [N, H, W] binary masks
            labels: [N] class labels
        """
        if self.coco is None:
            # Return empty annotations
            return np.zeros((0, 4)), np.zeros((0, self.image_size, self.image_size)), np.zeros(0)
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        if len(anns) == 0:
            return np.zeros((0, 4)), np.zeros((0, self.image_size, self.image_size)), np.zeros(0)
        
        img_info = self.coco.loadImgs(image_id)[0]
        height, width = img_info['height'], img_info['width']
        
        boxes = []
        masks = []
        labels = []
        
        for ann in anns:
            # Get bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Get mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    # RLE format
                    mask = coco_mask.decode(ann['segmentation'])
                else:
                    # Polygon format
                    rles = coco_mask.frPyObjects(ann['segmentation'], height, width)
                    mask = coco_mask.decode(coco_mask.merge(rles))
                masks.append(mask)
            else:
                # Create empty mask
                masks.append(np.zeros((height, width), dtype=np.uint8))
            
            # Get label (1 for tree, assuming single class)
            labels.append(ann.get('category_id', 1))
        
        boxes = np.array(boxes, dtype=np.float32)
        masks = np.stack(masks, axis=0)
        labels = np.array(labels, dtype=np.int64)
        
        return boxes, masks, labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample
        
        Returns:
            image: Tensor [3, H, W]
            target: Dict with 'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        original_size = image.shape[:2]
        
        # Load annotations
        boxes, masks, labels = self._load_annotations(image_id)
        
        # Apply transforms
        if len(boxes) > 0:
            # Prepare masks for albumentations
            mask_list = [masks[i] for i in range(masks.shape[0])]
            
            transformed = self.transforms(
                image=image,
                masks=mask_list,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist()
            )
            
            image = transformed['image']
            
            # Handle transformed masks and boxes
            if len(transformed['masks']) > 0:
                masks = np.stack(transformed['masks'], axis=0)
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                masks = np.zeros((0, self.image_size, self.image_size))
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)
        else:
            # Even with no boxes, must provide empty lists for bbox_params fields
            transformed = self.transforms(
                image=image,
                masks=[],
                bboxes=[],
                class_labels=[]
            )
            image = transformed['image']
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Compute areas
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros(0)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx if isinstance(image_id, int) else idx]),
            'area': area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, dim=0)
        
        return images, targets


class TreeCrownDatasetWithDepth(TreeCrownDataset):
    """
    Dataset that also loads pre-computed depth maps.
    Depth maps are loaded BEFORE augmentation so geometric transforms
    (flip, rotate) are applied consistently to both RGB and depth.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 1024,
        transforms: Optional[Callable] = None,
        use_augmentation: bool = True,
        depth_dir: Optional[str] = None
    ):
        # Store depth_dir before super().__init__ calls _get_default_transforms
        self._depth_dir_override = depth_dir
        super().__init__(data_root, split, image_size, transforms, use_augmentation)

        # Depth directory
        if depth_dir is None:
            self.depth_dir = os.path.join(data_root, 'depth', split)
        else:
            self.depth_dir = depth_dir

    def _get_default_transforms(self) -> A.Compose:
        """Override to register depth as additional target.
        Using 'mask' type ensures only geometric transforms (flip, rotate, resize)
        are applied to depth, not pixel-level transforms (color jitter, noise).
        """
        if self.use_augmentation:
            return A.Compose([
                # Always resize to target size first, then apply augmentations
                A.Resize(self.image_size, self.image_size),
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                # Color augmentations (RGB only, depth registered as 'mask' type)
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ], p=0.2),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ), additional_targets={'depth': 'mask'})
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ), additional_targets={'depth': 'mask'})

    def _load_depth(self, image_id) -> Optional[np.ndarray]:
        """Load pre-computed depth map"""
        if self.coco is not None:
            img_info = self.coco.loadImgs(image_id)[0]
            depth_name = os.path.splitext(img_info['file_name'])[0] + '_depth.npy'
        else:
            depth_name = f'{image_id}_depth.npy'

        depth_path = os.path.join(self.depth_dir, depth_name)

        if os.path.exists(depth_path):
            return np.load(depth_path).astype(np.float32)
        else:
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get sample with synchronized depth augmentation.
        Depth is loaded BEFORE augmentation so geometric transforms
        are applied consistently to both image and depth.
        """
        image_id = self.image_ids[idx]

        # Load image
        image = self._load_image(image_id)

        # Load annotations
        boxes, masks, labels = self._load_annotations(image_id)

        # Load depth BEFORE augmentation
        depth = self._load_depth(image_id)
        has_depth = depth is not None

        if has_depth:
            # Ensure depth is 2D (H, W)
            if depth.ndim > 2:
                depth = depth.squeeze()
            # Resize depth to match raw image size before augmentation
            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            # Dummy depth — model will fall back to DepthGenerator
            depth = np.zeros(image.shape[:2], dtype=np.float32)

        # Apply transforms with synchronized depth
        if len(boxes) > 0:
            mask_list = [masks[i] for i in range(masks.shape[0])]

            transformed = self.transforms(
                image=image,
                depth=depth,
                masks=mask_list,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist()
            )

            image = transformed['image']
            depth = transformed['depth']

            if len(transformed['masks']) > 0:
                masks = np.stack(transformed['masks'], axis=0)
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                masks = np.zeros((0, self.image_size, self.image_size))
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)
        else:
            transformed = self.transforms(
                image=image,
                depth=depth,
                masks=[],
                bboxes=[],
                class_labels=[]
            )
            image = transformed['image']
            depth = transformed['depth']

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Handle depth tensor — ToTensorV2 with 'mask' type returns (H, W)
        if isinstance(depth, torch.Tensor):
            depth = depth.float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)  # [1, H, W]
        else:
            depth = torch.as_tensor(depth, dtype=torch.float32)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)

        # Normalize depth to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)

        # Compute areas
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros(0)

        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx if isinstance(image_id, int) else idx]),
            'area': area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }

        if has_depth:
            target['depth'] = depth

        return image, target


def create_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: int = 1024,
    use_augmentation: bool = True,
    with_depth: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for training/validation
    
    Args:
        data_root: Root directory of dataset
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        use_augmentation: Whether to use data augmentation
        with_depth: Whether to load pre-computed depth maps
        
    Returns:
        DataLoader instance
    """
    if with_depth:
        dataset = TreeCrownDatasetWithDepth(
            data_root=data_root,
            split=split,
            image_size=image_size,
            use_augmentation=use_augmentation
        )
    else:
        dataset = TreeCrownDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            use_augmentation=use_augmentation
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=TreeCrownDataset.collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader