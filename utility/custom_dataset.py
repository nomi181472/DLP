
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Ensure image is in a consistent mode (e.g., RGB)
        img = img.convert("RGB")

        # Convert to a NumPy array
        img = np.array(img)

        # Convert to a PyTorch tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # Ensure it has the correct shape

        if self.transform:
            img = self.transform(img)

        label = ...  # Load your label here
        return img, label
