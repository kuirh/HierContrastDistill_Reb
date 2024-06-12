import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

from data_processing import Data_Loader

from sklearn.model_selection import train_test_split

def build_dataset(device, filename, batch_size=32):
    class CustomDataset(Dataset):
        def __init__(self, features_x, labels):
            self.features_x = features_x
            self.labels = labels

        def __len__(self):
            return len(self.features_x)

        def __getitem__(self, idx):
            return self.features_x[idx], self.labels[idx]

    data_loader = Data_Loader(filename)  # folder name -> Train.csv, Test.csv
    dataseed=random.randint(0,10000)
    print(dataseed)
    train_x, test_x, train_y, test_y = train_test_split(data_loader.scaled_x,data_loader.scaled_y, test_size=0.2, random_state=dataseed,shuffle=True )
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.155, random_state=dataseed,shuffle=True)

    train_x, val_x, test_x, train_y, val_y, test_y = \
        torch.from_numpy(train_x), torch.from_numpy(val_x), torch.from_numpy(test_x), \
        torch.from_numpy(train_y), torch.from_numpy(val_y), torch.from_numpy(test_y)

    train_x, val_x, test_x, train_y, val_y, test_y = \
        train_x.to(device), val_x.to(device), test_x.to(device), \
        train_y.to(device), val_y.to(device), test_y.to(device)

    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)
    test_dataset = CustomDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, data_loader




