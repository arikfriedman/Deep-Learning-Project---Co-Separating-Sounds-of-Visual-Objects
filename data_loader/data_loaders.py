from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import MusicDataset

class MusicDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,))  #????
        ])
        self.data_dir = data_dir
        self.dataset = MusicDataset(self.data_dir, transform=trsfm, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)