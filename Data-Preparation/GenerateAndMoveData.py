#import torch
#import torchvision
import sys
import h5py
#import cv2
from PIL import Image
import wave
import librosa
import random
import numpy as np
import soundfile as sf
import os

def sample_wav(wav, size=65535):
    # we expand the audio if its too short (with tile)
    if wav.shape[0] < size:
        n = int(size / wav.shape[0]) + 1
        wav = np.tile(wav, n)
    #start the sampling somewhere randomly
    start = random.randrange(0, wav.shape[0] - size + 1)
    #we get the audio values with the window size
    sample = wav[start:start+size]
    return sample


def create_spectrogram(wav, frame=1022, hop=256):
    trans = librosa.core.stft(wav, n_fft=frame, hop_length=hop, center=True)
    mag, phase = librosa.core.magphase(trans)
    #we treat it as array of arrays:
    mag = np.expand_dims(mag, axis=0)
    phase = np.expand_dims(np.angle(phase), axis=0)
    return mag, phase

def validChunk(path):
    f1, f2, f3, f4, f5, f6, f7, f8 = False, False, False, False, False, False, False, False

    crop_name = ""

    for file in os.listdir(path):
        if file.endswith('.npy'):
            f1 = True
        if file == 'image':
            f2 = True
        if file == 'detection_results':
            f3 = True
        if file.startswith('wav'):
            f4 = True
        if file.startswith('cropped'):
            f7 = True
            crop_name = file
    if f3:
        for file in os.listdir(os.path.join(path, 'detection_results')):
            if file == '.npy':
                f5 = True
    if f2:
        f6 = len(os.listdir(os.path.join(path, 'image'))) > 0

    if f7:
        f8 = len(os.listdir(os.path.join(path, crop_name))) > 0

    res = f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8
    #print( "-->> " + path + " : " + str(res))

    return res

'''
Retrieves a dictionary for a single clip with the following structure:
{
'id' : video id number
'audio' : { 'wave' : (wave, sr), 'stft' : (mags, phases) }
'images' : [(class_id1, image), (class_id2, image),...] -> just 1 or 2 cropped images
}
'''
def pickItems(path):
    if path is None:
        return None
    if not validChunk(path):
        return None
    print("-->> " + path + " : " + str(True))

    obj_dict = {}
    wav_name = ""
    crop_name = ""

    for item in os.listdir(path):
        if item.lower().startswith("cropped_"):
            crop_name = item
        if item.lower().startswith("wav"):
            wav_name = item

    #a tuple of audio, sr
    audio, sr = librosa.load(os.path.join(path, wav_name), sr=11025)
    sample = sample_wav(audio)
    mags, phases = create_spectrogram(sample)
    obj_dict['audio'] = {'wave': (audio, sr), 'stft': (mags, phases)}

    obj_dict['images'] = []
    crop_path = os.path.join(path, crop_name)
    for bbox in os.listdir(crop_path):
        obj_dict['images'] += [(bbox.split('.')[0], Image.open(os.path.join(crop_path, bbox)).resize((224, 224)))]

    vid_id = crop_name.split('_')[-1]
    obj_dict['id'] = vid_id

    return obj_dict

#retrieves a random different clip from the video class
def pick_rand_clip(vid_class, vid_id, base_path):
    path = base_path
    if random.randrange(0, 2) == 0:
        path = os.path.join(path, 'Solo')
    else:
        path = os.path.join(path, 'Duet')

    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_cls = dirs[idx]
    while len(dirs) > 1 and (dir_cls == vid_class or dir_cls == '99'):
        dirs.remove(dir_cls)
        idx = random.randrange(0, len(dirs))
        dir_cls = dirs[idx]
    path = os.path.join(path, dir_cls)


    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_id = dirs[idx]
    while len(dirs) > 1 and dir_id == vid_id:
        dirs.remove(dir_id)
        idx = random.randrange(0, len(dirs))
        dir_id = dirs[idx]
    path = os.path.join(path, dir_id)

    # chunk select
    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_chunk = dirs[idx]
    while len(dirs) > 1 and not validChunk(os.path.join(path, dir_chunk)):
        dirs.remove(dir_chunk)
        idx = random.randrange(0, len(dirs))
        dir_chunk = dirs[idx]
        print(os.path.join(path, dir_chunk))

    chunk_path = None
    if validChunk(os.path.join(path, dir_chunk)):
        chunk_path = os.path.join(path, dir_chunk)
    return chunk_path

def iterate_files(dir, count, log, source, target='/dsi/gannot-lab/datasets/Music/Batches'):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)

        #Erhu detection performed really worse and is not considered a genuine musical instrument detection
        if file == "99":
            continue

        if file.lower().startswith('chunk'):
            if not validChunk(file_path):
                continue

            print(count)
            count[0] += 1

            obj1 = pickItems(file_path)
            if obj1 is None:
                log.write("-->> obj-1 : could not pick items for " + file_path + "\n")
                continue

            random_clip_path = None
            #random_clip_path = file_path

            c = 0
            vid_id = file_path.split('/')[-2]
            vid_class = file_path.split('/')[-3]
            while random_clip_path == None and c < 50:
                random_clip_path = pick_rand_clip(vid_class, vid_id, source)
                c += 1
            obj2 = pickItems(random_clip_path)
            if obj2 is None:
                log.write("-->> obj-2 : could not pick items for " + str(random_clip_path) + "\n")
                continue
            mix_stft = (obj1['audio']['wave'][0] + obj2['audio']['wave'][0]) / 2
            mix_stft = librosa.util.normalize(mix_stft)
            sample = sample_wav(mix_stft)
            mix_mags, mix_phases = create_spectrogram(sample)

            #assume target exists
            t_path = os.path.join(target, str(count[1]).zfill(6))
            count[1] += 1

            f5 = h5py.File(t_path, 'w')
            f5.create_group('/obj1')
            f5['obj1']['id'] = obj1['id']
            #f5.create_group('/obj1/audio')
            f5['obj1']['audio'] = obj1['audio']
            #f5.create_group('/obj1/images')
            f5['obj1']['images'] = obj1['images']
            #f5.create_dataset(name='obj1', data=obj1)
            #f5.create_dataset(name='obj2', data=obj2)
            f5.create_dataset(name='mix', data=(mix_mags, mix_phases))

        elif os.path.isdir(file_path):
            iterate_files(file_path, count, log, source, target)


#pick for each video random other video and combine them together and normalize the audio

if __name__ == "__main__":

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/GeneratorErrorsLog.txt", "x")
    except:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/GeneratorErrorsLog.txt", "w")

        log.write("\nGenerator Errors : \n")

    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    count = [0, 0]
    source = '/dsi/gannot-lab/datasets/Music/Try/'
    iterate_files(root_dir, count, log, source)


'''
    #f = h5py.File(r'C:/Users/user/Desktop/try.h5', 'a')
    #im = cv2.imread(r'C:/Users/user/Desktop/0.jpg')
    #im = Image.open(r'C:/Users/user/Desktop/0.jpg')
    #im = im.crop((0, 0, 300, 300))

    #im.show()

    #im = im.resize((224, 224))
    #print(im.size)
    #im.show()

    #im = im.resize((224, 224))
    #print(im.size)
    #im.show()

    wav, rate = librosa.load('C:/Users/user/Desktop/wav_12.wav', sr=11025)
    wav2, rate2 = librosa.load('C:/Users/user/Desktop/wav_4.wav', sr=11025)
    wav3 = (librosa.util.normalize(wav) + librosa.util.normalize(wav2))
    wav4 = librosa.util.normalize(wav3)
    print(min(wav4))
    wav5 = librosa.util.normalize(wav2)
    print(min(wav5))
    sf.write('C:/Users/user/Desktop/wav_13.wav', wav4, rate)
    sf.write('C:/Users/user/Desktop/wav_5.wav', wav5, rate)

    a = 1
    if a == 1:
        exit(0)

    #wav = Wave('C:/Users/user/Desktop/wav_1.wav')
    #wav.start()
    #wav.overlay(wav2)
    wav3 = (wav + wav2) / 2

#    sf.write('C:/Users/user/Desktop/wav_11.wav', wav3, rate)
    #audio.write_audiofile(filename=chunkPath + '/wav_' + str(index) + '.wav', codec='pcm_s32le')
    #wav = librosa.stft(wav)
    print("rate = " + str(rate))
    print(wav.shape)
    size = 65535
    samp = sample_wav(wav, size)

    print("Length = ", end='')
    print(len(samp))
    frame = 1022
    hop = 256
    mag, phase = create_spectrogram(samp, frame, hop)
    print(mag.shape)
    print(phase.shape)




    #print(im.shape)
    #f.create_dataset(name="image", data=im)
    #f.create_dataset(name="audio_mag", data=mag)

    print(list(f.keys()))
    print(f['audio_mag'][0])
    #print(f['image'])'''