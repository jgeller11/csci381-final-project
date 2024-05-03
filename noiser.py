import torch


class Noiser():

    # L: number of noise values to return per sample
    # batch_size: number of samples to return
    # seed: set to True to get controlled noise
    def __init__(self, L, batch_size):
        self.L = L
        self.batch_size = batch_size
        
    # returns: uniform random values 0 to 1 in specified shape
    def noise(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = torch.empty(batch_size, self.L)
        torch.nn.init.uniform_(samples, a=0, b=1, generator=None)
        return samples
    

