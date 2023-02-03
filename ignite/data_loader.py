import torch

from torch.utils.data import Dataset,DataLoader

class MNISTDataset(Dataset):
    def __init__(self,data,labels,flatten=True):
        self.data=data
        self.labels=labels
        self.flatten=flatten

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            x = x.reshape(-1)

        return x , y
    