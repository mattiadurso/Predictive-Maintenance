import torch
import numpy as np
from torch.utils.data import Dataset

# Sliding window dataset definition. If in the range i:i+window:size there is a fail, the last safe window is repeated
class SimpleWindowDataset(Dataset):
    def __init__(self, df, seq_len):
        # Init the class
        self.X = df[df.columns[:-2]] # all columns until last 2
        self.y = df[df.columns[-1]] # last columns, the target variable
        self.len = df.shape[0] - seq_len + 1 # the number of windows
        self.seq_len = seq_len

    def __len__(self):
        return self.len

    def __getitem__(self, index): 
        # Return the couple (window, targets) of corrispondent index as torch float
        return (torch.tensor(self.X.iloc[index:index+self.seq_len].to_numpy().astype(np.float64)).float(),
                torch.tensor(self.y.iloc[index:index+self.seq_len].to_numpy().astype(np.float64)).float())

class CMAPSSWindowDataset(Dataset):
    def __init__(self, df, seq_len):
        self.X = df[df.columns[1:-1]]
        self.y = df[df.columns[-1]]
        self.len = df.shape[0] - seq_len + 1
        self.seq_len = seq_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index:index+self.seq_len].to_numpy().astype(np.float64)).float(),
                torch.tensor(self.y.iloc[index:index+self.seq_len].to_numpy().astype(np.float64)).float())
