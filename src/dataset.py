import numpy as np
import torch
from PIL import Image, ImageFile


class NinjaDataset:
    def __init__(self, image_paths, target_1, target_2, resize=None, transform=None):
        self.image_paths = image_paths
        self.target_1 = target_1
        self.target_2 = target_2
        self.resize = resize
        self.augmentation = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        target_1 = self.target_1[idx]
        target_2 = self.target_2[idx]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "Target 1": torch.tensor(target_1, dtype=torch.long),
            "Target 2": torch.tensor(target_2, dtype=torch.long),
        }
