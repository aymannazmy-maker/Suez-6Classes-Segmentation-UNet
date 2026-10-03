import os, glob, rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

# 6 Classes: 0-Urban, 1-Water, 2-Veg, 3-BareSoil, 4-Desert, 5-Cloud
class SuezDataset(Dataset):
    def __init__(self, img_dir="data/images", mask_dir="data/masks"):
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.tif")) + glob.glob(os.path.join(img_dir, "*.jp2")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*.tif")) + glob.glob(os.path.join(mask_dir, "*.png")))
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        with rasterio.open(self.images[idx]) as src:
            img = src.read([1,2,3,4]).astype(np.float32) / 10000.0
            img = np.clip(img, 0, 1)
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1).astype(np.int64)
        # resize to 256 if needed handled in model
        return torch.from_numpy(img), torch.from_numpy(mask)
