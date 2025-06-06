from flask import Flask, request, jsonify, send_file
from model.tranfer_learning_ResNet import Model_Tranfer_Resnet50
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Load model 1 lần khi server start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model_Tranfer_Resnet50().to(device)
checkpoint = torch.load("save_model/efficientnet/best_point.pt", weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Lấy categories từ dataset để đảm bảo thứ tự giống với training
root = 'animals'

categories = os.listdir(root)
print("Categories order:", categories)

# Hàm tiền xử lý ảnh
transform = Compose([
    ToTensor(),
    Resize((224,224)),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global selected_model_name, model
    data = request.get_json()
    print(data)
    model_name = data['model']
    print(model_name)

    if model_name not in ['ResNet', 'EfficientNet', 'ViT']:
        return jsonify({'error': 'Invalid model name'}), 400

    selected_model_name = model_name

    # Load mô hình tương ứng
    if model_name == 'ResNet':
        from model.tranfer_learning_ResNet import Model_Tranfer_Resnet50
        model = Model_Tranfer_Resnet50(model='resnet').to(device)
        checkpoint = torch.load("./save_model/resnet/best_point.pt", map_location=device)
    elif model_name == 'EfficientNet':
        from model.tranfer_learning_ResNet import Model_Tranfer_Resnet50
        model = Model_Tranfer_Resnet50(model='efficientnet').to(device)
        checkpoint = torch.load("./save_model/efficientnet/best_point.pt", map_location=device)
    elif model_name == 'ViT':
        from model.tranfer_learning_ResNet import Model_Tranfer_Resnet50
        model = Model_Tranfer_Resnet50(model='vit').to(device)
        checkpoint = torch.load("./save_model/vit/best_point.pt", map_location=device)

    model.load_state_dict(checkpoint['model'])
    print(checkpoint['accuracy'])
    model.eval()

    print(f"✅ Model switched to: {model_name}")
    return jsonify({'message': f'Model changed to {model_name}'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    in_memory_file = np.frombuffer(file.read(), np.uint8)
    ori_image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    if ori_image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    
    image = transform(image).to(device)
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        outputs = model(image)
        top3_prob, top3_indices = torch.topk(torch.softmax(outputs, dim=1), 3)
        
        print("\nTop 3 predictions:")
        for prob, idx in zip(top3_prob[0], top3_indices[0]):
            print(f"{categories[idx]}: {prob.item()*100:.2f}%")
        
        # Lấy kết quả cao nhất
        index_label = top3_indices[0][0].item()
        confidence = top3_prob[0][0].item()
        label = categories[index_label]
    
    return jsonify({
        'label': label,
        'confidence': f"{confidence*100:.2f}%",
        'top3': [
            {
                'label': categories[idx.item()],
                'confidence': f"{prob.item()*100:.2f}%"
            }
            for prob, idx in zip(top3_prob[0], top3_indices[0])
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
