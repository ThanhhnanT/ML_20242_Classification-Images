import cv2
from torch.utils.data import Dataset
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor

class Animal(Dataset):
    def __init__(self, root = None, train = True, transform = None):
        self.data = []
        # Sắp xếp categories để đảm bảo thứ tự luôn giống nhau
        self.categories = os.listdir(root)
        # print("Categories order:", self.categories)  # In ra để debug
        self.labels = []
        self.transform = transform
        self.all = []
        for i, (item) in enumerate(self.categories):
            path = os.path.join(root, item)
            data_train, data_test = train_test_split(os.listdir(path), train_size = 0.8, random_state = 42)
            # for _ in os.listdir(path):
            #     self.all.append(i)
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
    root = '../animals'
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    data = Animal(root = root, train = True, transform = transform)
    print("Number of training samples:", len(data), len(data.all))
    print("Number of categories:", len(data.categories))

    # label_counts = Counter(data.all)
    # class_names = [data.categories[i] for i in label_counts.keys()]
    # counts = [label_counts[i] for i in label_counts.keys()]
    #
    # # Sắp xếp theo tên class
    # sorted_items = sorted(zip(class_names, counts), key=lambda x: x[0])
    # class_names, counts = zip(*sorted_items)
    #
    # num_classes = len(class_names)
    # classes_per_fig = 30
    # num_figs = math.ceil(num_classes / classes_per_fig)
    #
    # # Tạo folder để lưu ảnh (nếu cần)
    # os.makedirs("../result/charts", exist_ok=True)
    #
    # for i in range(num_figs):
    #     start = i * classes_per_fig
    #     end = min((i + 1) * classes_per_fig, num_classes)
    #
    #     plt.figure(figsize=(14, 6))
    #     plt.bar(class_names[start:end], counts[start:end], color='skyblue')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.xlabel("Class")
    #     plt.ylabel("Number of Images")
    #     plt.title(f"Class Distribution {start + 1} to {end}")
    #     plt.tight_layout()
    #
    #     filename = f"../result/charts/class_distribution_{i + 1}.png"
    #     plt.savefig(filename)
    #     print(f"Saved: {filename}")
    #     plt.close()