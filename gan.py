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
        self.discriminator = make_dense_network(self.flattened_image_size, 1, discriminator_hidden_layers, discriminator_layer_size)

    def gen_noise(self, batch_size = 1):
        noise = torch.empty(batch_size, self.noise_size)
        torch.nn.init.uniform_(noise, a=0, b=1)
        return noise
    
    def gen_images(self, batch_size=1):
        """
            Uses the generator to create batch_size number of images
        """
        return self.generator(self.gen_noise(batch_size))
    
    def mix_with_generated_images(self, real_images):
            # Presently the images are in two seperate groups–maybe shuffle them later?
            generated_images = self.gen_images(len(real_images))
            combined_images = torch.cat([real_images, generated_images])
            labels = torch.cat((torch.ones(len(real_images)), torch.zeros(len(generated_images))))

            return combined_images, labels
        
    
    def discriminator_loss(self, preds, labels):
        return torch.sum(torch.log(torch.abs(labels-preds)))
    
    def evaluate(self, dataloader):
        """
            Calculates the model's performance on the given data.
            Presently, only returns the discriminator's loss
            this should maybe return a tuple with the generator's loss too
        """
        total_samples = 0
        total_loss = 0
        for batch, _ in dataloader:
            # Generate images with generator and combine with real images
            images, labels = self.mix_with_generated_images(batch)
            
            # Calculate the discriminator's performance
            preds = self.discriminator(images)
            total_loss += self.discriminator_loss(preds, labels)
            total_samples += len(images)
        return total_loss / total_samples
    
    def train(self, train_dataloader, val_dataloader = None, num_epochs=10, discrim_sub_iterations=1):
        # TODO: actually allow for discrim_sub_iterations != 1
        if discrim_sub_iterations != 1:
            raise Exception("discrim_sub_iterations > 1 not yet supported")

        discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr = 0.01, momentum = 0.9)
        generator_optimizer = torch.optim.SGD(self.generator.parameters(), lr = 0.01, momentum = 0.9)
        
        for epoch in range(num_epochs):
            # Evaluate model if validation set is given
            if val_dataloader is not None:
                self.discriminator.eval()
                print(f"After {epoch} training epochs: loss = {self.evaluate(val_dataloader)}")

            # Generator training loop--go through full dataset
            self.discriminator.train()
            for training_images, _ in train_dataloader:
                batch_size = len(training_images)

                # Discriminator training loop
                for _ in range(discrim_sub_iterations):
                    discriminator_optimizer.zero_grad()

                    # Use generator to get minibatched input for discriminator
                    images, labels = self.mix_with_generated_images(training_images)

                    # Get predictions from model, calculate loss, and update parameters
                    preds = torch.sigmoid(self.discriminator(images))
                    loss = self.discriminator_loss(preds, labels)
                    loss.backward()
                    discriminator_optimizer.step()

                # Sample minibatch of noise
                # Pass noise through generator
                #  -> pass result through discriminator
                # calculate gradients and update generator (NOT discriminator)

if __name__ == "__main__":
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

    noise_size = 30
    image_size = 28

    gan = GAN(noise_size, image_size)
    gan.train(data_manager.train(batch_size=8), data_manager.val())

    # generator, discriminator = create_generator_and_discriminator(noise_size, image_size, 10, 5)
    # train(generator, discriminator, Noiser(30, batch_size), Noiser(30, batch_size), data_manager.train(16), val_data_loader=data_manager.val())


    # for param in gen.parameters():
    #     print(param)
    # print("Dis:")
    # for param in dis.parameters():
    #     print(param)
