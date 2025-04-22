import cv2
import torch
import torch.nn as nn
from jinja2.optimizer import optimize
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.v2 import ColorJitter
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from torch.utils.data import DataLoader
from dataset import Animal
from tranfer_learning_ResNet import Model_Tranfer_Resnet50
from torch import transpose
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

def train():
    writer = SummaryWriter()
    root = 'animals'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Model_Tranfer_Resnet50().to(device)
    for name, param in model.named_parameters():
        if 'fc' in name or 'layer4' in name :
            pass
        else:
            param.requires_grad =False
    train_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        ColorJitter(brightness=0.5, contrast=0.9, saturation=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])

    ])

    data_train = Animal(root=root, train=True, transform=train_transform)
    data_test = Animal(root=root, train= False, transform=test_transform)

    train_dataloader = DataLoader(
        data_train,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    test_dataloader = DataLoader(
        data_test,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )
    loss_function = nn.CrossEntropyLoss()
    optimize_function = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs =100
    max = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimize_function.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimize_function.step()
            writer.add_scalar("Loss/train", loss, iter + epoch*len(train_dataloader))
            progress_bar.set_description(f"epoch_{epoch + 1}/{epochs}, Iteration_{iter + 1}/{len(train_dataloader)}, Loss_{loss.item()}")
        model.eval()
        all_labels = []
        all_predic = []
        i =1
        for images, labels in test_dataloader:
            images = images.to(device)
            with torch.no_grad():
                output = model(images)
                predict = torch.argmax(output, dim =1)
            all_predic.extend(predict.cpu().numpy())
            all_labels.extend(labels.numpy())
        result = accuracy_score(all_labels, all_predic)
        writer.add_scalar('Accuracy', result, epoch)
        print('Accuracy: ',result)
        checkpoint= {
            'epoch': epoch +1,
            'model': model.state_dict(),
            'optimizer': optimize_function.state_dict(),
            'accuracy': result
        }
        torch.save(checkpoint, os.path.join('save_model', "last_point.pt"))
        if result > max:
            torch.save(checkpoint, os.path.join('save_model', "best_point.pt"))
            max = result


if __name__ == '__main__':
    train()


