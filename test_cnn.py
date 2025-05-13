from tranfer_learning_ResNet import Model_Tranfer_Resnet50
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import os
from dataset import Animal  # Import class Animal để lấy categories

def test():
    root = 'animals'
    # Lấy categories từ dataset để đảm bảo thứ tự giống với training
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    dataset = Animal(root=root, train=True, transform=transform)
    categories = os.listdir(root)
    print("Categories order:", categories)
    print("Number of categories:", len(categories))

    # Thiết lập GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load checkpoint model
    model = Model_Tranfer_Resnet50().to(device)
    checkpoint = torch.load("save_model/best_point.pt", weights_only = False, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Model accuracy:", checkpoint['accuracy'])

    # Lấy ảnh

    ori_image = cv2.imread('images.jpeg')
    ori_image = cv2.imread('download.jpeg')
    if ori_image is None:
        print("Error: Could not read image 'download.jpeg'")
        return
    # Chuyển đổi ảnh phù hợp với mô hình
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    transform = Compose([
        ToTensor(),
        Resize((224,224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).to(device)
    image = torch.unsqueeze(image, dim=0)
    print("Input image shape:", image.shape)

    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        # Lấy top 3 dự đoán
        top3_prob, top3_indices = torch.topk(torch.softmax(outputs, dim=1), 3)
        
        # In ra debug information
        print("\nTop 3 predictions:")
        for prob, idx in zip(top3_prob[0], top3_indices[0]):
            print(f"{categories[idx]}: {prob.item()*100:.2f}%")
        
        # Lấy kết quả cao nhất
        index_label = top3_indices[0][0].item()
        confidence = top3_prob[0][0].item()
        label = categories[index_label]
    
    print(f"\nFinal prediction: {label} with confidence {confidence*100:.2f}%")
    cv2.imshow(label, ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    test()