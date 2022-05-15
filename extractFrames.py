import numpy as np
import sys
import cv2
import os

path = sys.argv[1]  #"C:\\Users\\user\\Desktop\\frames"#
fileName = sys.argv[2]  #"\\frames.npy"
arr = np.load(path) #+fileName)
#print(len(arr))
fullPath = path + "\\images"
try:
    os.mkdir(fullPath)
except:
    pass
for i, img in enumerate(arr):
    #print(img)
    cv2.imwrite(fullPath+"\\"+str(i)+".jpg", img)


