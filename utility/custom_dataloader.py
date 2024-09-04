import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class CustomDataLoader:
    def __init__(self, dataset, train_ratio=0.9, batch_size=8, shuffle=True):
        """
        Initializes the CustomDataLoader with the dataset and parameters for splitting and DataLoader creation.

        Args:
        - dataset: An instance of torchvision.datasets.ImageFolder or any PyTorch dataset object.
        - train_ratio (float): The ratio of data to be used for training. Default is 0.9.
        - batch_size (int): The batch size for DataLoader. Default is 8.
        - shuffle (bool): Whether to shuffle the dataset. Default is True.
        """
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate the sizes for training and testing
        train_size = int(self.train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size

        # Split the dataset into training and testing sets
        self.trainset, self.testset = random_split(self.dataset, [train_size, test_size])

        # Create DataLoaders for training and testing
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return self.testloader

    def get_dataset_sizes(self):
        return len(self.trainset), len(self.testset)