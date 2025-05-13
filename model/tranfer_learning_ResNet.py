import torch.nn as nn
from numpy.ma.core import argmax
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.transforms import Compose, ToTensor, Resize
from model.dataset import Animal
import torch

class Model_Tranfer_Resnet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = efficientnet_v2_m(weights = EfficientNet_V2_M_Weights.DEFAULT )
            # print(self.model)
            self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=1280, out_features=512),
                nn.LeakyReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(),
                nn.Linear(in_features=512, out_features=256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=120)
            )
            # print(self.model)

        def forward(self, x):
            x= self.model(x)
            return x

if __name__ == '__main__':
    model = Model_Tranfer_Resnet50()
    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #
    for name, param in model.named_parameters():
        if 'features.6' in name or 'features.7' in name or 'features.8' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name, param.requires_grad)


