import torch
import torchvision
import sys
import os
import subprocess

count = 0


def iterate_files(dir):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            for item in os.listdir(file_path):
                if item.lower().startswith('frames') and item.lower().endswith('.npy'):
                    npy_path = file_path
                    npy_name = item
                    subprocess.Popen("./ExtractFramesFromNpy.py " + npy_path + " " + npy_name)
        elif os.path.isdir(file_path):
            iterate_files(file_path)


if __name__ == "main":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    iterate_files(root_dir)
