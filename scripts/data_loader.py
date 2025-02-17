# scripts/data_loader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def get_transforms():
    """
    Returns a torchvision transformation pipeline that resizes images to 256x256,
    converts them to tensors, and normalizes the image.
    """
    return T.Compose([
        T.Resize((256, 256)),  # Resize images to 256x256
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_all_files(root_dir, suffix):
    """
    Recursively finds all files in root_dir that end with the given suffix.
    Returns a dictionary mapping base filename (without the suffix) to the full file path.
    """
    files_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(suffix):
                base = fname.replace(suffix, "")
                files_dict[base] = os.path.join(dirpath, fname)
    return files_dict

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=20):
        """
        image_dir: Root directory for leftImg8bit images (e.g., dataset/leftImg8bit_trainvaltest/leftImg8bit/train)
        mask_dir: Root directory for gtFine masks (e.g., dataset/gtFine_trainvaltest/gtFine/train)
        transform: Transformation pipeline for images.
        num_classes: Number of valid classes (e.g., 20). Any mask value >= num_classes will be set to 255 (ignore index).
        """
        self.transform = transform if transform is not None else get_transforms()
        self.num_classes = num_classes

        # Get all image files and mask files recursively
        image_dict = get_all_files(image_dir, "_leftImg8bit.png")
        mask_dict = get_all_files(mask_dir, "_gtFine_labelIds.png")

        # Find common keys (base filenames present in both)
        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        if len(common_keys) == 0:
            raise ValueError("No matching image-mask pairs found! Check your dataset paths.")

        self.image_paths = [image_dict[k] for k in common_keys]
        self.mask_paths = [mask_dict[k] for k in common_keys]

        print(f"Found {len(self.image_paths)} image-mask pairs in the dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        # Load mask (as grayscale)
        mask = Image.open(mask_path)

        # Resize both image and mask to 256x256
        image = image.resize((256, 256), resample=Image.BILINEAR)
        mask = mask.resize((256, 256), resample=Image.NEAREST)

        # Apply transformations to image
        image = self.transform(image)

        # Convert mask to NumPy array and then to tensor.
        mask = np.array(mask, dtype=np.int64)
        # Set any value >= num_classes to 255 (ignore index)
        mask[mask >= self.num_classes] = 255
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
