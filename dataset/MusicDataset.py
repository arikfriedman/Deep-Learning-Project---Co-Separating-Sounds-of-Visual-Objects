from torch.utils.data import Dataset
import os
import random
import librosa
from torchvision.io import read_image

# the files in data_dir will be enumerated from 000000, and contain pickles objects
class MusicDataset(Dataset):

    def __init__(self, data_dir, transform, log, train=True):
        self.log = log
        self.dir_path = data_dir
        self.size = 0
        try:
            self.size = len(os.listdir(self.dir_path))
        except OSError:
            log.write(" -->> " + self.dir_path + " is not a valid path\n")

    def __len__(self):
        return self.size

    '''
    ->  each object has 2 dictionaries, each for a single clip with the following structure:
        {
            'id'     :   video id number
            'audio'  :   { 'wave' : (wave, sr), 'stft' : (mags, phases) }
            'images' :   [(class_id1, image), (class_id2, image),...] -> just 1 or 2 cropped images
        }
        
    ->  and total:
        
        {
            "obj1": obj1,
            "obj2": obj2,
            "mix": (mix_mags, mix_phases)
        }
    '''
    # returns a list of objects with their labels - between 2 to 4 objects
    def __getitem__(self, index):
        pickle_idx = str(index).zfill(6) + '.pickle'
        file_path = os.path.join(self.dir_path, pickle_idx)
        try:
            mix_file = open(file_path, 'rb')
            pick = pickle.load(mix_file)
            mix_file.close()
        except OSError:
            self.log.write("-->> Error with file " + file_path)
            pick = None

        return pick


'''

separate for validation

find jpgs mean and std
apply trsfm in music dataset
normalize:  check for audio when to norm

'''