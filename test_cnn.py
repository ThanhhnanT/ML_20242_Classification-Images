from tranfer_learning_ResNet import Model_Tranfer_Resnet50
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import os

def test():
    root = 'animals'
    ## Lấy categories
    categories = os.listdir(root)
    print(len(categories))

    # Thiết lập GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load checkpoint model
    model = Model_Tranfer_Resnet50().to(device)
    checkpoint = torch.load("save_model/best_point.pt", weights_only = False)
    model.load_state_dict(checkpoint['model'])
    print(checkpoint['accuracy'])

    # Lấy ảnh
    ori_image = cv2.imread('download.jpeg')

    # Chuyển đổi ảnh phù hợp với mô hình
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    tranform = Compose([
        ToTensor(),
        Resize((224,224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    image = tranform(image).to(device)
    image = torch.unsqueeze(image, dim=0)
    print(image.shape)

    # Dự đoán
    model.eval()
    with torch.no_grad():
        predict = model(image)
        index_label = torch.argmax(predict)
        label = categories[index_label]
    cv2.imshow(label, ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    test()