import torch
import torchvision
import sys
import os

count = 0


def iterate_files(dir):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('detection_r'):
            for item in os.listdir(file_path):
                if item.lower().endswith('.npy'):
                    npy_path = os.path.join(file_path, item)
                    arr = np.load(npy_path)
                    
        elif os.path.isdir(file_path):
            iterate_files(file_path)


if __name__ == "main":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    iterate_files(root_dir)

