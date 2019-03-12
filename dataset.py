import numpy as np
import pickle
import torch

from torch.utils.data import Dataset 


class EarthquakeDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        with open(src_file, 'rb') as f:
            self.signals = pickle.load(f)     

        with open(tgt_file, 'rb') as f:
            self.targets = pickle.load(f)

    def __getitem__(self, index):
        return {
            'id': index,
            'signals': torch.cuda.FloatTensor(self.signals[index]),
            'target': torch.cuda.FloatTensor([self.targets[index]]),
        } 

    def __len__(self):
        return len(self.signals)

