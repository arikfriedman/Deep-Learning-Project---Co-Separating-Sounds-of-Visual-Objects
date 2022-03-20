'''
  @InProceedings{zhao2018sound,
        author = {Zhao, Hang and Gan, Chuang and Rouditchenko, Andrew and Vondrick, Carl and McDermott, Josh and Torralba, Antonio},
        title = {The Sound of Pixels},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {September},
        year = {2018}
  }
'''

#!pip install pytube

#import pytube
import json
from pytube import YouTube
import os
import numpy as np
#import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt


class DataDownloader():
  def __init__(self):
    pass
#if __name__ == 'main':

path = "/dsi/gannot-lab/datasets/Music/MUSIC_arme/Data/Duet/saxophone acoustic_guitar/Hallelujah (Rufus Wainwright) - Fabián Rivero- Sax and Guitar Cover/chunk_1/frames_1.txt.npy"
path = "frames_1.txt.npy"
file = np.load(path)
plt.imshow(file[50])
plt.show()

if 1 != 1:

  path = "/dsi/gannot-lab/datasets/Music_arme"


  link = "https://www.youtube.com/watch?v=8DHG_hVSw1o"
  try:
    yt = YouTube(link)
  except:
    print("Connection Error with url '" + link + "'")

  for i in range(15):
    try:
      yt.streams.get_by_itag(18).download(path)
      print(i)
    except:
       print("err")
