import cv2
from torch.utils.data import Dataset
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor

class Animal(Dataset):
    def __init__(self, root = None, train = True, transform = None):
        self.data = []
        self.categories = os.listdir(root)
        self.labels = []
        self.transform = transform
        # print(len(self.categories))
        for i,(item) in enumerate(self.categories):
            path = os.path.join(root, item)
            # print(len(os.listdir(path)))
            data_train, data_test = train_test_split(os.listdir(path), train_size = 0.8, random_state = 42)
            if train == True:
                for path_image in data_train:
                    self.data.append(os.path.join(path, path_image))
                    self.labels.append(i)

            else:
                for path_image in data_test:
                    self.data.append(os.path.join(path, path_image))
                    self.labels.append(i)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(self.data[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = self.labels[index]
        return image, label

if __name__ == '__main__':
    root = 'animals'
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    data = Animal(root = root, train =True, transform = transform)
    # image, label = data.__getitem__(3)
    # print(image, data.categories[label])

    # print(data.__len__())