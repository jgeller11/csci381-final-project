import tqdm
import torch
import torch.nn.init as init
from torch.nn import Parameter, Sequential
from data_loading import mnist_data
from noiser import Noiser

class Dense(torch.nn.Module):
    """
        Dense Module code copied from lab "kernels"
    """
    def __init__(self, input_size, output_size):
        """
        A Module that multiplies an feature matrix (with input_size
        feature vectors) with a weight vector to produce output_size
        output vectors per feature vector. See the unit tests in test.py
        for specific examples.
    
        - An offset variable is added to each feature vector. e.g. Suppose 
          the feature tensor (a batch of feature vectors) is 
          [[6.2, 127.0], [5.4, 129.0]]. The forward method will first
          prepend offset variables to each feature vector, i.e. it 
          becomes [[1.0, 6.2, 127.0], [1.0, 5.4, 129.0]]. Then, it takes the
          dot product of the weight vector with each of the new feature
          vectors.
        
        """
        super(Dense, self).__init__()
        self.weight = Parameter(torch.empty(output_size, 1+input_size))
        init.kaiming_uniform_(self.weight, nonlinearity="relu")
        
    def unit_weights(self):
        """Resets all weights to 1. For testing purposes."""
        init.ones_(self.weight)
        
    def forward(self, x):
        """
        Computes the linear combination of the weight vector with
        each of the feature vectors in the feature matrix x.
        
        Note that this function will add an offset variable to each
        feature vector. e.g. Suppose the feature matrix (a batch of
        feature vectors) is [[6.2, 127.0], [5.4, 129.0]]. The forward
        method will first prepend offset variables to each feature vector,
        i.e. the feature matrix becomes 
        [[1.0, 6.2, 127.0], [1.0, 5.4, 129.0]]. Then, it takes the dot 
        product of the weight vector with each of the new feature vectors.
        
        """        
        x2 = torch.cat([torch.ones(x.shape[0],1),x], dim=1)
        return torch.matmul(self.weight,x2.t()).t()

class ReLU(torch.nn.Module):
    """
    Implements a rectified linear unit. The ```forward``` method takes
    a torch.Tensor as its argument, and returns a torch.Tensor of the
    same shape, where all negative entries are replaced by 0. 
    For instance:

        t = torch.tensor([[-3., 0., 3.2], 
                          [2., -3.5, 1.]])
        relu = ReLU()
        relu.forward(t)

    should return the tensor:
        
        torch.tensor([[0., 0., 3.2], 
                      [2., 0., 1.]])
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(min=0) 

class Clamp(torch.nn.Module):
    """
    Clamps the values of the input tensor between 0 and 1.    
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(min=0, max=1) 

def make_dense_network(input_size, output_size, num_hidden_layers=10, hidden_layer_size=100):
    layers = Sequential()

    layers.add_module("dense1", Dense(input_size, hidden_layer_size))
    layers.add_module("relu1", ReLU())

    for i in range(num_hidden_layers - 1):
        layers.add_module(f"dense{i+2}", Dense(hidden_layer_size, hidden_layer_size))
        layers.add_module(f"relu{i+2}", ReLU())

    layers.add_module(f"dense{num_hidden_layers+1}", Dense(hidden_layer_size, output_size))

    return layers

def create_generator_and_discriminator(noise_size, image_size, num_hidden_layers = 10, hidden_layer_size = 100):
    generator = make_dense_network(noise_size, image_size, num_hidden_layers, hidden_layer_size)
    generator.add_module("clamp", Clamp())

    discriminator = make_dense_network(image_size, 1, num_hidden_layers, hidden_layer_size)
    return generator, discriminator


def train(generator, discriminator, 
          generator_noise_gen, discriminator_noise_gen, image_dataloader,
          num_iterations=10, discrim_sub_iterations=1):
    for _ in range(num_epochs):
        # Generator training loop--go through full dataset
        for X, y in image_dataloader:
            # Discriminator training loop
            for _ in range(discrim_sub_iterations):
                ## Getting Inputs
                # Sample noise
                noise = discriminator_noise_gen.noise()
                # Use generator to get minibatched input for discriminator
                generated_images = generator(noise)
                # Get minibatch of real examples from training data

                ## Updating parameters
                # Use combined inputs to calculate gradients and take a step
                pass
            # Sample minibatch of noise
            # Pass noise through generator
            #  -> pass result through discriminator
            # calculate gradients and update generator (NOT discriminator)


if __name__ == "__main__":
    batch_size = 4
    image_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=batch_size,
                                          shuffle=False)

    generator, discriminator = create_generator_and_discriminator(2, 3, 4, 5)
    train(generator, discriminator, Noiser(30, batch_size), Noiser(30, batch_size), image_loader)


    # for param in gen.parameters():
    #     print(param)
    # print("Dis:")
    # for param in dis.parameters():
    #     print(param)
