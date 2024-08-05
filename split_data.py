import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "./resized_mel_specs/"
train_dir = "./split_data/train/"
test_dir = "./split_data/test/"

for class_folder in os.listdir(data_dir):
    files = []
    class_path = os.path.join(data_dir, class_folder)
    for file in os.listdir(class_path):
        files.append(os.path.join(class_path, file))
    
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, class_folder, os.path.basename(file)))
    for file in test_files:
        shutil.copy(file, os.path.join(test_dir, class_folder, os.path.basename(file)))