import cv2
import torch
import torch.nn as nn
from torchvision.transforms.v2 import ColorJitter
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from torch.utils.data import DataLoader
from dataset import Animal
from tranfer_learning_ResNet import Model_Tranfer_Resnet50
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def Arg():
    parser = ArgumentParser(description="Train Model")
    parser.add_argument("--batch-size", '-b', type=int, default=64, help='batch-size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--root', '-r', type=str, default='../animals', help='root path')
    parser.add_argument('--checkpoint', '-c', type=str, default='../save_model/efficientnet/last_point.pt',help='last save model')
    parser.add_argument('--best_point', '-bp', type=str, default='../save_model/efficientnet/best_point.pt')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--image_size', '-imgs', type=int, default=224, help='size image')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--tensorboard', '-ts', type=str, default= '../runs/efficientnet', choices=['efficientnet', 'resnet', 'vit'],
                        help='tensor path')
    parser.add_argument('--model_type', '-m', type=str, default='efficientnet', choices=['efficientnet', 'resnet', 'vit'],
                        help='Choose model type: efficientnet | resnet | vit')
    arg = parser.parse_args()
    return arg

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(40, 40))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train():
    arg = Arg()
    root = arg.root

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load model
    model = Model_Tranfer_Resnet50(model = arg.model_type).to(device)
    # Loss and optimize function
    loss_function = nn.CrossEntropyLoss()
    optimize_function = torch.optim.AdamW(model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)
    if arg.model_type == 'efficientnet':
        arg.tensorboard = '../runs/efficientnet'
        arg.checkpoint = '../save_model/efficientnet/last_point.pt'
        arg.best_point = '../save_model/efficientnet/best_point.pt'
    elif arg.model_type == 'resnet':
        arg.tensorboard = '../runs/resnet'
        arg.checkpoint = '../save_model/resnet/last_point.pt'
        arg.best_point = '../save_model/resnet/best_point.pt'
    elif arg.model_type == 'vit':
        arg.tensorboard = '../runs/vit'
        arg.checkpoint = '../save_model/vit/last_point.pt'
        arg.best_point = '../save_model/vit/best_point.pt'

    writer = SummaryWriter(arg.tensorboard)
    start =0
    if os.path.exists(arg.checkpoint):
        checkpoint = torch.load(arg.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimize_function.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch']
        print(checkpoint['accuracy'])

    # Freeze model
    for name, param in model.named_parameters():
        if arg.model_type == 'efficientnet':
            if 'features.7' in name or 'features.8' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif arg.model_type == 'resnet':
            for name, param in model.named_parameters():
                if 'layer4' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif arg.model_type == 'vit':
            for name, param in model.named_parameters():
                if 'heads' in name or 'encoder.ln' in name or 'encoder_layer_11' in name or 'encoder_layer_10' in name or 'encoder_layer_9' in name or 'encoder_layer_8' in name or 'encoder_layer_7' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    # Transform image
    train_transform = Compose([
        ToTensor(),
        Resize((arg.image_size, arg.image_size)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        ColorJitter(brightness=0.5, contrast=0.9, saturation=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        ToTensor(),
        Resize((arg.image_size, arg.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])

    ])

    # Load data
    data_train = Animal(root=root, train=True, transform=train_transform)
    data_test = Animal(root=root, train= False, transform=test_transform)

    train_dataloader = DataLoader(
        data_train,
        batch_size=arg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6
    )

    test_dataloader = DataLoader(
        data_test,
        batch_size=arg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6
    )

    # Epoch
    epochs = arg.epochs
    max = 0
    if os.path.exists(arg.best_point):
        best_point = torch.load(arg.best_point)
        max = best_point['accuracy']
    print('max', max)

    # Train model
    for epoch in range(start, epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        total_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimize_function.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)

            # Backward
            loss.backward()
            optimize_function.step()
            total_loss.append(loss.item())
            writer.add_scalar("Loss/train", np.mean(total_loss), iter + epoch*len(train_dataloader))
            progress_bar.set_description(f"epoch_{epoch + 1}/{epochs}, Iteration_{iter + 1}/{len(train_dataloader)}, Loss_{loss.item()}")

        # Evaluation
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

        # Compute accuracy
        result = accuracy_score(all_labels, all_predic)
        writer.add_scalar('Accuracy', result, epoch)
        print('Accuracy: ',result)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predic),
                              class_names= data_test.categories, epoch=epoch)

        # Create checkpoint
        checkpoint= {
            'epoch': epoch +1,
            'model': model.state_dict(),
            'optimizer': optimize_function.state_dict(),
            'accuracy': result
        }
        torch.save(checkpoint, arg.checkpoint)
        print("Successful save last point")
        if result > max:
            torch.save(checkpoint, arg.best_point)
            max = result
            print("Successful save max point")


if __name__ == '__main__':
    train()