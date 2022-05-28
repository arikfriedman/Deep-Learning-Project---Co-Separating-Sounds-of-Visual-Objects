import torch
import torchvision
import sys
import os
import numpy as np
from PIL import Image

def cropAndResize(image_path, bbox, log):
    try:
        im = Image.open(image_path)
    except OSError:
        log.write("-->> " + image_path + " does not exist!!\n")
        return None

    im = im.crop(bbox)
    im = im.resize((224, 224))
    return im

def hasDirs(path):
    f1, f2, f3, f4, f5, f6 = False, False, False, False, False, False

    for file in os.listdir(path):
        if file.endswith('.npy'):
            f1 = True
        if file == 'images':
            f2 = True
        if file == 'detection_results':
            f3 = True
        if file.startswith('wav'):
            f4 = True
    if f3:
        for file in os.listdir(os.path.join(path, 'detection_results')):
            if file == '.npy':
                f5 = True
    if f2:
        f6 = len(os.listdir(os.path.join(path, 'images'))) > 0

    res = f1 and f2 and f3 and f4 and f5 and f6
    #print(res)

    return res

def iterate_files(root, count, log, parent):

    for dir in os.listdir(root):
        dir_path = os.path.join(root, dir)

        if dir == 'Duet':
            parent = 'Duet'
        if dir == 'Solo':
            parent = 'Solo'

        if dir.lower().startswith('chunk'):
            print(count)
            count[0] += 1
            if not hasDirs(dir_path):
                log.write("[" + str(count[1]) + "] --->>> " + dir_path + " is not a valid chunk\n")
                count[1] += 1
                continue

            detect_path = os.path.join(dir_path, 'detection_results')
            npy_path = os.path.join(detect_path, '.npy')
            arr = np.load(npy_path)
            arr = np.asarray(arr)

            if len(arr) == 0:
                continue

            adj = 0
            if arr[0].shape == (8, ):
                adj = 1

            conf1 = 0
            for idx, tup in enumerate(arr):
                if arr[idx][2+adj] > conf1:
                    conf1 = arr[idx][2+adj]
                    index1 = idx
                    cls1 = arr[idx][1+adj]

            #cls1 = arr[0][1+adj]
            #conf1 = arr[0][2+adj]
            #index1 = 0

            cls2 = -1
            conf2 = 0
            index2 = 0

            for idx, tup in enumerate(arr):
                cls = arr[idx][1+adj]
                conf = arr[idx][2+adj]

                if cls != cls1:
                    if cls2 == -1:
                        cls2 = cls
                        conf2 = conf
                        index2 = idx
                    elif conf > conf2:
                        cls2 = cls
                        conf2 = conf
                        index2 = idx


            cls1 = int(cls1)
            cls2 = int(cls2)

            #extract bbox to jpeg
            frame1 = int(arr[index1][adj])
            #dir ends with chunk_
            image_path = os.path.join(dir_path, "images")
            image_path = os.path.join(image_path, str(frame1) + ".jpg")
            bbox = arr[index1][3 + adj:]

            cropped_image = cropAndResize(image_path, tuple(bbox), log)
            if cropped_image is None:
                continue

            vid_num = dir_path.split('/')[-2]
            crop_path = os.path.join(dir_path, "cropped_" + vid_num)
            try:
                os.mkdir(crop_path)
            except OSError:
                pass

            cropped_image.save(os.path.join(crop_path, str(cls1) + ".jpg"))

            if parent == 'Duet' and cls2 != -1:
                frame2 = int(arr[index2][adj])
                image_path = os.path.join(dir_path, "images")
                image_path = os.path.join(image_path, str(frame2) + ".jpg")
                bbox = arr[index2][3 + adj:]

                cropped_image = cropAndResize(image_path, tuple(bbox), log)
                if cropped_image is None:
                    continue

                cropped_image.save(os.path.join(crop_path, str(cls2) + ".jpg"))

        elif os.path.isdir(dir_path):
            iterate_files(dir_path, count, log, parent)


if __name__ == "__main__":
    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/FilterErrorsLog.txt", "x")
    except:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/FilterErrorsLog.txt", "w")

    log.write("\nFilter Errors : \n")

    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    iterate_files(root_dir, [0, 0], log, '')
    print("Done")

