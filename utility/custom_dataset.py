
import torch
from torchvision import datasets, transforms

class CustomDataset:
    def __init__(self, root_dir, trn_func=[
            #transforms.Resize((200,200)),  # Resize images
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize
        ]):
        # Define the transformations
        self.transform = transforms.Compose(trn_func)
        
        # Load the dataset using the transformations
        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=self.transform
        )

    def get_dataset(self):
        return self.dataset

    def get_classes(self):
        return self.dataset.classes