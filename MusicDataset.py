from torch.utils.data import Dataset
import torch

class MusicDataset(Dataset):
    def __init__(self, fps, mode, batchSize):
        self.fps = fps


        if mode == "train":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batchSize,
                shuffle=True)
        elif mode == 'test':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batchSize,
                shuffle=False)   
                
    def __len__(self):
        pass

    def __getitem__(self):
        pass
    