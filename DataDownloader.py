
#!pip install pytube
import json
#from google.colab import files
from pytube import YouTube

class DataDownloader():
  def __init__(self):
    pass

  def downloadSingleVideo(self, path, videoCode, errorLog):
    link = "https://www.youtube.com/watch?v=" + videoCode
    try:
      yt = YouTube(link)
    except:
      print("Connection Error with url '" + link + "'")

    #print("Title: ",yt.title)
      
    # filters out all the files with "mp4" extension 
    #mp4files = yt.streams.filter(file_extension='mp4')
    vid = None
    try:
      vid = yt.streams.get_by_itag(18).download(path)
    except:
      errorLog = errorLog + ["Error occured downloading url '" + link + "'"]
    #print()

    return vid

    #vid = downloadSingleVideo(path, "kw-OQpF9N4E")

  def downloadDataFromJSON(self, path, jsonObj):
    i = 1
    n = 149
    errorLog = []
    for node in jsonObj:
      if node == "videos":
        for key in jsonObj[node]:
          urls = jsonObj[node][key]
          label = key
          for url in urls:
            #if i >= 1:
            #print("downloading video ", i, " out of ",n)
            vid = self.downloadSingleVideo(path + key, url, errorLog)
            i = i + 1
            #print(vid)

    return errorLog

if __name__ == 'main':
  path = "Data"
  jsonObj = json.load(open('MUSIC.json'))
  
  #uploaded = files.upload()
  #jsonObj = json.load("MUSIC.json")
  DD = DataDownloader()
  errorLog = DD.downloadDataFromJSON(path, jsonObj)
  #errorLog = errorLog + ["none"]
  print("errors : ")
  print(errorLog)

  print('Done!')