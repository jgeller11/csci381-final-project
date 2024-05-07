import torchvision
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from info import DEVICE
from torch.utils.data import Dataset
from img_plot import display_image



# data_loader = torch.utils.data.DataLoader(training_data,
#                                           batch_size=4,
#                                           shuffle=False)
class MNISTDataset(Dataset):
    def __init__(self):
        mnist_data = torchvision.datasets.MNIST('data/mnist', 
                                        download=True)
        self.data = mnist_data.data.to(dtype=torch.float32, device=DEVICE) / 255
        self.data = self.data.reshape((self.data.shape[0], -1))
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]


# Code from uncool.py of kernels lab
class DataManager:
    
    def __init__(self, train_partition, val_partition, test_partition, 
                 feature_key = 'image', response_key='category'):
        """
        Creates a DataManager from a JSON configuration file. The job
        of a DataManager is to manage the data needed to train and
        evaluate a neural network.
        
        - train_partition is the DataPartition for the training data.
        - test_partition is the DataPartition for the test data.
        - feature_key is the key associated with the feature in each
          datum of the data partitions, i.e. train_partition[i][feature_key]
          should be the ith feature tensor in the training partition.
        - response_key is the key associated with the response in each
          datum of the data partitions, i.e. train_partition[i][response_key]
          should be the ith response tensor in the training partition.
        
        """
        self.train_set = train_partition
        self.val_set = val_partition
        self.test_set = test_partition
        self.feature_key = feature_key
        self.response_key = response_key
        try:
            self.categories = sorted(list(set(train_partition.categories()) |
                                          set(val_partition.categories()) |
                                          set(test_partition.categories())))
        except AttributeError:
            pass
    
    def train(self, batch_size):
        """
        Returns a torch.DataLoader for the training examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
        
        - batch_size is the number of desired training examples per batch
        
        """
        return DataLoader(self.train_set, batch_size=batch_size,
                                  sampler=RandomSampler(self.train_set))
    
    def val(self):
        """
        Returns a torch.DataLoader for the val examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
                
        """
        return DataLoader(self.val_set, batch_size=4, 
                          sampler=SequentialSampler(self.val_set))
    
    def test(self):
        """
        Returns a torch.DataLoader for the test examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
                
        """
        return DataLoader(self.test_set, batch_size=4, 
                          sampler=SequentialSampler(self.test_set))

    
    def features_and_response(self, batch):
        """
        Converts a batch obtained from either the train or test DataLoader
        into an feature tensor and a response tensor.
        
        The feature tensor returned is just batch[self.feature_key].
        
        To build the response tensor, one starts with batch[self.response_key],
        where each element is a "response value". Each of these response
        values is then mapped to the index of that response in the sorted set of
        all possible response values. The resulting tensor should be
        a LongTensor.

        The return value of this function is:
            feature_tensor, response_tensor
        
        See the unit tests in test.py for example usages.
        
        """
        def category_index(category):
            return self.categories.index(category)
        inputs = batch[self.feature_key].float()
        labels = torch.Tensor([category_index(c) for c 
                               in batch[self.response_key]]).long()
        return inputs, labels
