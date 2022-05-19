import torch
import torchvision
import sys
import h5py
import cv2

#pick for each video random other video and combine them together and normalize the audio

if __name__ == "__main__":
    # argument 1 is the root directory of the data
#    root_dir = sys.argv[1]
#    dataset_dir = sys.argv[2]
    count = [0]
    f = h5py.File(r'C:/Users/user/Desktop/try.h5', 'a')
    im = cv2.imread(r'C:/Users/user/Desktop/0.jpg')

    #print(im.shape)
    f.create_dataset(name="image", data=im)
    print(list(f.keys()))
    print(f['image'])
    #print(f['image'])