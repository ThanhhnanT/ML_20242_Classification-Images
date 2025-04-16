import torch.nn as nn
from numpy.ma.core import argmax
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Resize
from dataset import Animal
import torch

class Model_Tranfer_Resnet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features = 2048, out_features = 1024),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(in_features = 1024, out_features = 90)
            )

        def _forward_impl(self, x):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x

        def forward(self, x):
            return self._forward_impl(x)

if __name__ == '__main__':
    model = Model_Tranfer_Resnet50()

    for name, param in model.named_parameters():
        if 'fc' in name or 'layer4' in name:
            pass
        else:
            param.requires_grad =False
        print(name, param.requires_grad)
