import os
import shutil

# Đường dẫn đến thư mục chứa dữ liệu
root = os.path.join("split_ttv_dataset_type_of_plants", "Train_Set_Folder")
train_split = os.listdir(root)

# Tạo thư mục animals nếu chưa tồn tại
if not os.path.exists("animals"):
    os.makedirs("animals")

# Copy các thư mục từ Train_Set_Folder vào animals
for cate in train_split:
    old_path = os.path.join(root, cate)
    new_path = os.path.join("animals", cate)
    if os.path.isdir(old_path):
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
