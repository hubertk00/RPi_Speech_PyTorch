import os
from pathlib import Path

def load_wake_word(root_path, wake_word="Hugo"):
    root_dir = Path(root_path)
    file_paths = []
    labels = []

    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir(): continue

        if class_dir.name == wake_word:
            label = 1
        else: label = 0

        files = list(class_dir.glob('*.wav'))
        
        file_paths.extend([str(f) for f in files])
        labels.extend([label] * len(files))
        
    return file_paths, labels

def load_multiclass_data(root_path, commands):
    root_dir = Path(root_path)
    class_names = [cmd for cmd in commands if (root_dir / cmd).is_dir()]    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    file_paths = []
    labels = []

    for class_name in class_names:
        class_dir = root_dir / class_name
        label = class_to_idx[class_name]
        files = list(class_dir.glob('*.wav'))
        
        file_paths.extend([str(f) for f in files])
        labels.extend([label] * len(files))
        
    return file_paths, labels, class_to_idx