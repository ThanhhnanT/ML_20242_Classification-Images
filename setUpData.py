import os


root ="./split_ttv_dataset_type_of_plants"
train_split = os.listdir(os.path.join(root, "Test_Set_Folder"))
test_split = os.listdir(os.path.join(root, "Train_Set_Folder"))
val_split = os.listdir(os.path.join(root, "Validation_Set_Folder"))

for cate in train_split:
    old_path = os.path.join(root, "Test_Set_Folder", cate)
    new_path = os.path.join("animals", cate)
    os.replace(old_path, new_path)
