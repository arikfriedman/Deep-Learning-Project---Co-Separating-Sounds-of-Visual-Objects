import torch
import torchvision
import sys
import os
import numpy as np

count = 0


def iterate_files(dir):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('detection_r'):
            for item in os.listdir(file_path):
                if item.lower().endswith('.npy'):
                    npy_path = os.path.join(file_path, item)
                    arr = np.load(npy_path)
                    arr = np.asarray(arr)
                    adj = 0
                    if arr[0].shape == 8:
                        adj = 1
                    cls1 = arr[0][1+adj]
                    conf1 = arr[0][2+adj]
                    index1 = 0

                    cls2 = -1
                    conf2 = 0
                    index2 = 0

                    for idx, tup in enumerate(arr):
                        cls = arr[idx][1+adj]
                        conf = arr[idx][2+adj]

                        if cls == cls1:
                            if conf > conf1:
                                conf1 = conf
                                index1 = idx
                        elif cls2 == -1:
                            cls2 = cls
                            conf2 = conf
                        else:
                            if conf > conf2:
                                conf2 = conf
                                index2 = idx

                    #extract bbox to jpeg
                    frame1 = arr[index1][adj]
                    image_path = os.path.join(dir, "images")
                    image_path = os.path.join(image_path, str(frame1) + ".jpg")
                    bbox = arr[index1][3 + adj:]
                    cropped_image = crop(image_path, bbox)
                    dir_path = os.path.join(dir, str(count))
                    os.mkfir(dir_path)
                    cropped_image.save()
                    if cls2 != -1:
                        frame2 = arr[index2][adj]
                        image_path = os.path.join(dir, "images")
                        image_path = os.path.join(image_path, str(frame2) + ".jpg")
                        bbox = arr[index2][3 + adj:]
                        cropped_image = crop(image_path, bbox)
                        cropped_image.save(dir_path)

                    os.copyDir()...
                    count += 1

        elif os.path.isdir(file_path):
            iterate_files(file_path)


if __name__ == "main":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    iterate_files(root_dir)

