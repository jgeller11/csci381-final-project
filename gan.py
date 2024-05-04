import math
import tqdm
import torch
import torch.nn.init as init
from torch.nn import Parameter, Sequential, BatchNorm1d
import torchvision
import torchvision.transforms as T
from data_loading import DataManager
from img_plot import display_image
from info import DEVICE

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
        x2 = torch.cat([torch.ones((x.shape[0],1)), x], dim=1)
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

class HiddenLayer(torch.nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layers = Sequential(
            BatchNorm1d(layer_size),
            Dense(layer_size, layer_size),
            ReLU())

    def forward(self, x):
        return x + self.layers(x)

    


def make_dense_network(input_size, output_size, num_hidden_layers=10, hidden_layer_size=100):
    """
        Creates a Sequential of Dense and ReLU layers with given sizes
    """
    layers = Sequential()

    layers.add_module("batchnorm1", BatchNorm1d(input_size))
    layers.add_module("dense1", Dense(input_size, hidden_layer_size))
    layers.add_module("relu1", ReLU())

    for i in range(num_hidden_layers - 1):
        layers.add_module(f"hiddenlayer{i+2}", HiddenLayer(hidden_layer_size))

    layers.add_module(f"batchnorm{i+2}", BatchNorm1d(hidden_layer_size))
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
        self.generator.add_module("sigmoid", torch.nn.Sigmoid())
        # self.generator.add_module("clamp", Clamp()) # Theory crafting this might be problematic

        # Output of discriminator is prediction of whether or not it is real 
        # (note it still needs to be passed through sigmoid to be normalized)
        self.discriminator = make_dense_network(self.flattened_image_size, 1, discriminator_hidden_layers, discriminator_layer_size)
        self.discriminator.add_module("sigmoid", torch.nn.Sigmoid())

    def gen_noise(self, batch_size = 1):
        return torch.randn(batch_size, self.noise_size)
    
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
        
    
    def discriminator_loss(self, preds, labels, epsilon = 1e-8):
        return -torch.mean(torch.log(1-torch.abs(labels-preds) + epsilon))
    
    def generator_loss(self, preds, labels, epsilon = 1e-8): 
        return -torch.mean(torch.log(torch.abs(labels-preds) + epsilon))
        

    def evaluate(self, dataloader):
        """
            Calculates the model's performance on the given data.
            Presently, only returns the discriminator's loss
            this should maybe return a tuple with the generator's loss too
        """
        total_samples = 0
        correct_on_real_images = 0
        correct_on_generated_images = 0
        for batch, _ in dataloader:
            real_preds = self.discriminator(batch).squeeze(dim=1)
            correct_on_real_images += (real_preds > 0.5).sum()

            generated_preds = self.discriminator(self.gen_images(len(batch))).squeeze(dim=1)
            correct_on_generated_images += (generated_preds < 0.5).sum()
            
            total_samples += len(batch)
        return correct_on_real_images / total_samples, correct_on_generated_images / total_samples
    
    def train(self, train_dataloader, val_dataloader = None, num_epochs=10, discrim_sub_iterations=1):
        # TODO: actually allow for discrim_sub_iterations != 1
        if discrim_sub_iterations != 1:
            raise Exception("discrim_sub_iterations > 1 not yet supported")
        


        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.9))
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.0001, betas=(0.5, 0.9))
        
        for epoch in range(num_epochs):
            # Evaluate model if validation set is given
            self.discriminator.eval()
            self.generator.eval()
            real_img_training_acc, gen_img_training_acc = self.evaluate(train_dataloader)
            print(f"After {epoch} training epochs:")
            print(f"\tReal Image Accuracy:                   {real_img_training_acc}")
            print(f"\tGenerated Image Accuracy:              {gen_img_training_acc}")

            if val_dataloader is not None:
                real_img_validation_acc, gen_img_validation_acc = self.evaluate(val_dataloader)
                print()
                print(f"\tReal Image Accuracy (Validation):      {real_img_validation_acc}")
                print(f"\tGenerated Image Accuracy (Validation): {gen_img_validation_acc}")

            # display_image(self.gen_images().squeeze().clone().detach())
            display_image(self.gen_images().squeeze().clone().detach(), display=False, filename = f"testimgs/{epoch}_1.png")
            display_image(self.gen_images().squeeze().clone().detach(), display=False, filename = f"testimgs/{epoch}_2.png")
            display_image(self.gen_images().squeeze().clone().detach(), display=False, filename = f"testimgs/{epoch}_3.png")
            
            
            # Generator training loop--go through full dataset
            self.discriminator.train()
            for training_images, _ in train_dataloader:
                # Discriminator training loop
                self.discriminator.train()
                for _ in range(discrim_sub_iterations):
                    discriminator_optimizer.zero_grad()

                    # Use generator to get minibatched input for discriminator
                    images, labels = self.mix_with_generated_images(training_images)

                    # Get predictions from model, calculate loss, and update parameters
                    preds = self.discriminator(images).squeeze(dim = 1)
                    loss = self.discriminator_loss(preds, labels)

                    # if loss > 0:
                    assert not math.isnan(loss)
                    loss.backward()
                    discriminator_optimizer.step()
                # Generator training subroutine
                self.discriminator.eval()
                self.generator.train()

                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                # Use generator to get minibatched input for discriminator
                images = self.gen_images(len(training_images)).squeeze(dim = 1)
                labels = torch.zeros(len(training_images))
                # For now, I switched this to more closely match the paper
                #images, labels = self.mix_with_generated_images(training_images)

                # Get predictions from model, calculate loss, and update parameters
                preds = self.discriminator(images).squeeze(dim = 1)
                loss = self.generator_loss(preds, labels)
                #print(loss.item(), end="\r")
                assert not math.isnan(loss)
                loss.backward()
                generator_optimizer.step()
            print()

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

    noise_size = 100
    image_size = 28

    gan = GAN(noise_size, image_size, discriminator_hidden_layers=6, discriminator_layer_size=100, generator_hidden_layers=6, generator_layer_size=100)
    gan.train(data_manager.train(batch_size=32), num_epochs=1000)
    print(f"Discriminator params:")
    print([param for param in gan.discriminator.parameters()])
    print(f"Generator params:")
    print([param for param in gan.generator.parameters()])
