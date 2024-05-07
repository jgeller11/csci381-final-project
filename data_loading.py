import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from info import DEVICE

class MNISTDataset(Dataset):
    def __init__(self):
        mnist_data = torchvision.datasets.MNIST('data/mnist', 
                                        download=True)
        self.data = mnist_data.data.to(dtype=torch.float32, device=DEVICE) / 255
        self.data = self.data.flatten(1)#self.data.unsqueeze(dim = 1)
        print(self.data.shape)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

class DataManager:
    
    def __init__(self, dataset, training_percent=0.7, validation_percent=0.1):
        """
            Splits dataset into train, validation, and testing splits
            Note, testing percent is calculated implicitly based on 
            training and validation percents.
        """
        train_size = int(training_percent * len(dataset))
        val_size = int(validation_percent * len(dataset))
        test_size = len(dataset) - train_size - val_size
        training_data, validation_data, testing_data = torch.utils.data.random_split(dataset, 
                                            [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(0))
        self.train_set = training_data
        self.val_set = validation_data
        self.test_set = testing_data
    
    def train(self, batch_size):
        return DataLoader(self.train_set, batch_size=batch_size,
                                  sampler=RandomSampler(self.train_set))
    
    def val(self, batch_size):
        return DataLoader(self.val_set, batch_size=batch_size, 
                          sampler=SequentialSampler(self.val_set))
    
    def test(self, batch_size):
        return DataLoader(self.test_set, batch_size=batch_size, 
                          sampler=SequentialSampler(self.test_set))