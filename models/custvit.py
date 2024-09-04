from vit_pytorch import SimpleViT
import torch.nn as nn


class CustVIT(nn.Module):
    def __init__(self,num_classes,image_size=200,channels=3,patch_size=20,dim=512,depth=6,heads=16,mlp_dim=1024):
        super(CustVIT, self).__init__()

        self.model = SimpleViT(
                image_size = image_size,
                channels=channels,
                patch_size = 20,
                num_classes = num_classes,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim
            )
    def fit(self,x):
        return self.model(x)