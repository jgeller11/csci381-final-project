import tqdm
import torch
import torch.nn.init as init
from torch.nn import Parameter, Sequential
from noiser import Noiser
import torchvision
import torchvision.transforms as T
from data_loading import DataManager

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
        print(f"Theta Shape: {self.weight.shape}")
        print(f"x Shape: {x.shape}")
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
    """
        Creates a Sequential of Dense and ReLU layers with given sizes
    """
    layers = Sequential()

    layers.add_module("dense1", Dense(input_size, hidden_layer_size))
    layers.add_module("relu1", ReLU())

    for i in range(num_hidden_layers - 1):
        layers.add_module(f"dense{i+2}", Dense(hidden_layer_size, hidden_layer_size))
        layers.add_module(f"relu{i+2}", ReLU())

    layers.add_module(f"dense{num_hidden_layers+1}", Dense(hidden_layer_size, output_size))

    return layers

class GAN():
    def __init__(self, noise_size, image_width, 
                 generator_hidden_layers=10, generator_layer_size=100,
                 discriminator_hidden_layers=10, discriminator_layer_size=100):
        
        self.noise_size = noise_size
        # We assume square images
        self.flattened_image_size = image_width * image_width

        # Clamp the output of the generator, so it's a valid image
        self.generator = make_dense_network(noise_size, self.flattened_image_size, generator_hidden_layers, generator_layer_size)
        self.generator.add_module("clamp", Clamp())

        # Output of discriminator is prediction of whether or not it is real 
        # (note it still needs to be passed through sigmoid to be normalized)
        self.discriminator = make_dense_network(image_size, 1, discriminator_hidden_layers, discriminator_layer_size)

    def gen_noise(self, batch_size = 1):
        noise = torch.empty(batch_size, self.noise_size)
        torch.nn.init.uniform_(noise, a=0, b=1)
        return noise
    
    def train(train_dataloader, val_dataloader = None, num_epochs=10, discrim_sub_iterations=1):
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr = 0.01, momentum = 0.9)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr = 0.01, momentum = 0.9)

        if val_dataloader is not None:
            evaluator = Evaluator(discriminator, generator, discriminator_noiser)

        # TODO: actually allow for discrim_sub_iterations != 1
        if discrim_sub_iterations != 1:
            raise Exception("discrim_sub_iterations > 1 not yet supported")
        for epoch in range(num_epochs):
            # Evaluate model if validation set is given
            if val_data_loader is not None:
                discriminator.eval()
            
                print(f"After {epoch} training epochs: loss: {evaluator.evaluate(val_data_loader)}")

            # Generator training loop--go through full dataset
            discriminator.train()
            for X, _ in image_dataloader:
                # Discriminator training loop

                for _ in range(discrim_sub_iterations):

                    discriminator_optimizer.zero_grad()

                    ## Getting Inputs
                    # Sample noise
                    noise = discriminator_noiser.noise()
                    # Use generator to get minibatched input for discriminator
                    generated_images = generator(noise)
                    # Combine with minibatch of real examples from training data
                    all_images = torch.cat((generated_images, X), dim = 0)
                    
                    # Make y vector with 1 for training images, 0 for generated images
                    y = torch.cat((torch.ones(X.shape[0]),torch.zeros(generated_images.shape[0])))

                    # Get predictions from model, calculate loss
                    preds = torch.sigmoid(discriminator(all_images))
                    loss = discriminator_loss(preds, y)
                    loss.backward()

                    ## Updating parameters
                    discriminator_optimizer.step()

                # Sample minibatch of noise
                # Pass noise through generator
                #  -> pass result through discriminator
                # calculate gradients and update generator (NOT discriminator)
    

def discriminator_loss(preds, y):
    return torch.sum(torch.log(torch.abs(y-preds)))

def evaluate_discriminator(discriminator, test_images, test_y):
    preds = torch.sigmoid(discriminator(test_images))
    return discriminator_loss(preds, test_y)

class Evaluator():
    def __init__(self, discriminator, generator, noiser):
        self.discriminator = discriminator
        self.generator = generator
        self.noiser = noiser

    # Evaluates discriminator using images from data_loader
    def evaluate(self, data_loader):
        total_samples = 0
        total_loss = 0
        for batch, _ in data_loader:

            # Generate images with generator
            noise_data = self.noiser.noise(len(batch))
            generated_images = self.generator(noise_data)

            print(f"Batch Shape: {batch.shape}")
            print(f"Generated Images Shape: {generated_images.shape}")
            all_images = torch.cat([batch, generated_images])

            y = torch.cat((torch.ones(len(batch)), torch.zeros(len(batch))))

            total_loss += discriminator_loss(all_images, y)
            total_samples += len(batch) * 2
        return total_loss/total_samples



            

        




if __name__ == "__main__":
    # test_net = make_dense_network(3, 5)
    
    # print(test_net(torch.randn((16, 3))))

    mnist_data = torchvision.datasets.MNIST('data/mnist', 
                                        download=True,
                                        transform=T.Compose([T.ToTensor(),torch.flatten]))

    train_size = int(.7 * len(mnist_data))
    val_size = int(.1 * len(mnist_data))
    test_size = len(mnist_data) - train_size - val_size
    training_data, val_data, test_data = torch.utils.data.random_split(mnist_data, 
                                            [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(1))

    data_manager = DataManager(training_data, val_data, test_data)

    batch_size = 8
    noise_size = 30
    image_size = 28

    noiser = Noiser(30, batch_size)
    print(noiser.noise().shape)

    generator, discriminator = create_generator_and_discriminator(noise_size, image_size, 10, 5)
    train(generator, discriminator, Noiser(30, batch_size), Noiser(30, batch_size), data_manager.train(16), val_data_loader=data_manager.val())


    # for param in gen.parameters():
    #     print(param)
    # print("Dis:")
    # for param in dis.parameters():
    #     print(param)
