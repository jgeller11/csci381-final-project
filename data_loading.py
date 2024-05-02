import torchvision
from torchvision.transforms import ToTensor
import torch

mnist_data = torchvision.datasets.MNIST('data/mnist', 
                                        download=True,
                                        transform=ToTensor())
data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=4,
                                          shuffle=False)
for batch in data_loader:
    print(batch[0].shape)
    print(batch[0][0])
    print(torch.mean(batch[0]))
    break