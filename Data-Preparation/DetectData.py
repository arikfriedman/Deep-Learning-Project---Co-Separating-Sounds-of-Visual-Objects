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
                if item.lower().startswith('images'):
                    images_path = os.path.join(file_path, item.join('/'))
                    subprocess.Popen("./getDetectionResults.py --cfg /home/dsi/ravivme/fasterRCNN/faster-rcnn.pytorch/cfgs/res101_ls.yml " +
                                     "--load_dir /home/dsi/ravivme/fasterRCNN/faster-rcnn.pytorch/data/pretrained_model --net res101 --checksession 1 " +
                                     "--checkepoch 1 --checkpoint 1 --image_dir " + images_path)
        elif os.path.isdir(file_path):
            iterate_files(file_path)


if __name__ == "main":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    iterate_files(root_dir)

