import numpy as np
import sys
import cv2
import os

path = sys.argv[1]  # "C:\\Users\\user\\Desktop\\frames"#
fileName = sys.argv[2]  # "\\frames.npy"
arr = np.load(os.path.join(path, fileName))
# print(len(arr))
fullPath = os.path.join(path, "images")
try:
    os.mkdir(fullPath)
except OSError:
    pass
for i, img in enumerate(arr):
    # print(img)
    cv2.imwrite(os.path.join(fullPath, str(i) + ".jpg"), img)
