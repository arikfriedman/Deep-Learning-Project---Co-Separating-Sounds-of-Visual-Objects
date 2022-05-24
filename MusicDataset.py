from torch.utils.data import Dataset
import os
import random
import librosa
from torchvision.io import read_image

# the files in obj_path will be enumerated from 00000_classx_classy to 09876_classz_class_w for instance
class MusicDataset(Dataset):

    def __init__(self, obj_path):
        self.path = obj_path
        try:
            self.files = os.listdir(self.path)
            self.files.sort()
        except OSError:
            print("not a valid path")

        self.size = len(self.files)

    def __len__(self):
        return self.size

    # returns a list of objects with their labels - between 2 to 4 objects
    def __getitem__(self, index):
        mix_dir = os.path.join(self.path, self.files[index])
        return





        idx1 = index
        idx2 = random.randint(0, self.size-1)
        if idx2 == idx1:
            idx2 = (idx2 + 1) % self.size

        objs = []
        labels = []
        audios = []
self.path
        path1 = self.path + ... self.files[idx1]
        path2 = self.files[idx2]

        for file in os.listdir(path1):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = read_image(os.path.join(path1, file))
                label = file.split('.')[0]
                objs.append(img)
                labels.append(label)
            elif file.lower().endswith(('.wav', '.mp3')):
                audio = librosa.load(os.path.join(path1, file))
                audios.append(audio)

        for file in os.listdir(path2):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = read_image(os.path.join(path1, file))
                label = file.split('.')[0]
                objs.append(img)
                labels.append(label)
            elif file.lower().endswith(('.wav', '.mp3')):
                audio, sr = librosa.load(os.path.join(path1, file))
                audios.append(audio)
2 lists
separate for valid
    normalize audio
    stft in batch
    check if in train we have at leasts 1 object for each viedo
add confidence bar when analizing objects
leave erhu and not choose it, give it number 16

->
normalize all data
check if json works




        audio = MusicDataset.mix_audios(audios)
        return objs, labels, audio

    @staticmethod
    def mix_audios(audios):
        if len(audios) == 1:
            return librosa.stft(audios[0])
        mixed = audios[0]
        for audio in audios[1:]:
            mixed += audio

        return librosa.stft(mixed)