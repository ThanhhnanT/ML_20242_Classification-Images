import torch.nn as nn
from numpy.ma.core import argmax
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Compose, ToTensor, Resize
import torch

class Model_Tranfer_Resnet50(nn.Module):
        def __init__(self, model ='efficientnet'):
            super().__init__()
            self.model_type = model.lower()

            # Efficenet
            if self.model_type == 'efficientnet':
                base_model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
                base_model.classifier = self.output(1280)

            # Resnet
            elif self.model_type == 'resnet':
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
                base_model.fc = self.output(2048)

            # ViT
            elif self.model_type == 'vit':
                base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
                base_model.heads = self.output(768)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model = base_model
            # print(self.modelViT.heads)

        def output(self, input=1280):
            last_layer = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=input, out_features=512),
                nn.LeakyReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(),
                nn.Linear(in_features=512, out_features=256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=120)
            )
            return last_layer

        def forward(self, x):
            return self.model(x)

if __name__ == '__main__':
    model = Model_Tranfer_Resnet50(model = 'vit')
    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #
    for name, param in model.named_parameters():
        if 'heads' in name or 'encoder.ln' in name or 'encoder_layer_11' in name or 'encoder_layer_10' in name or 'encoder_layer_9' in name or 'encoder_layer_8' in name or 'encoder_layer_7' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name, param.requires_grad)


