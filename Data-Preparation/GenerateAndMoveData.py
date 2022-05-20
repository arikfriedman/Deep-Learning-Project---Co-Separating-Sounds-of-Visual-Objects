import torch
import torchvision
import sys
import h5py
import cv2
from PIL import Image
import wave
import librosa
import random
import numpy as np
import soundfile as sf

def sample_wav(wav, size):
    # we expand the audio if its too short (with tile)
    if wav.shape[0] < size:
        n = int(size / wav.shape[0]) + 1
        wav = np.tile(wav, n)
    #start the sampling somewhere randomly
    start = random.randrange(0, wav.shape[0] - size + 1)
    #we get the audio values with the window size
    sample = wav[start:start+size]
    return sample


def create_spectrogram(wav, frame, hop):
    trans = librosa.core.stft(wav, n_fft=frame, hop_length=hop, center=True)
    mag, phase = librosa.core.magphase(trans)
    #we treat it as array of arrays:
    mag = np.expand_dims(mag, axis=0)
    phase = np.expand_dims(np.angle(phase), axis=0)
    return mag, phase



#pick for each video random other video and combine them together and normalize the audio

if __name__ == "__main__":
    # argument 1 is the root directory of the data
#    root_dir = sys.argv[1]
#    dataset_dir = sys.argv[2]
    count = [0]
    f = h5py.File(r'C:/Users/user/Desktop/try.h5', 'w')
    #im = cv2.imread(r'C:/Users/user/Desktop/0.jpg')
    im = Image.open(r'C:/Users/user/Desktop/0.jpg')
    im = im.crop((0, 0, 300, 300))
    #im.show()
    im = im.resize((224, 224))
    #im.show()

    wav, rate = librosa.load('C:/Users/user/Desktop/wav_12.wav', sr=11025)
    wav2, rate2 = librosa.load('C:/Users/user/Desktop/wav_4.wav', sr=11025)
    #wav = Wave('C:/Users/user/Desktop/wav_1.wav')
    #wav.start()
    #wav.overlay(wav2)
    wav3 = (wav + wav2) / 2

    sf.write('C:/Users/user/Desktop/wav_11.wav', wav3, rate)
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
    #f.create_dataset(name="audio_mags", data=mag)

    print(list(f.keys()))
#    print(f['audio_mags'])
    #print(f['image'])