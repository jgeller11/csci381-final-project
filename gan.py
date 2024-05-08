import math
import torch
from torch.nn import Parameter, Sequential, BatchNorm1d, BatchNorm2d, Linear, ReLU, Sigmoid, Tanh, Conv2d, MaxPool2d, Flatten
from data_loading import DataManager, MNISTDataset
from img_plot import display_image
from info import DEVICE


class NormalizedLinearWithResidual(torch.nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layers = Sequential(
            BatchNorm1d(layer_size),
            Linear(layer_size, layer_size),
            ReLU())

    def forward(self, x):
        return x + self.layers(x)

def make_dense_network(input_size, output_size, num_hidden_layers=10, hidden_layer_size=100):
    """
        Creates a Sequential of Dense and ReLU layers with given sizes
    """
    layers = Sequential()

    layers.add_module("input_batchnorm", BatchNorm1d(input_size))
    layers.add_module("input_linear", Linear(input_size, hidden_layer_size))
    layers.add_module("input_relu", ReLU())

    for i in range(num_hidden_layers - 1):
        layers.add_module(f"hiddenlayer{i+1}", NormalizedLinearWithResidual(hidden_layer_size))

    layers.add_module(f"output_batchnorm", BatchNorm1d(hidden_layer_size))
    layers.add_module(f"output_linear", Linear(hidden_layer_size, output_size))

    return layers

class GAN():
    def __init__(self, noise_size, image_width):

        self.noise_size = noise_size
        # We assume square images
        self.image_width = image_width
        self.flattened_image_size = image_width * image_width

        # Clamp the output of the generator, so it's a valid image
        self.generator = Sequential(
            NormalizedLinearWithResidual(49),
            BatchNorm1d(49),
            Linear(49, 98),
            ReLU(),
            NormalizedLinearWithResidual(98),
            BatchNorm1d(98),
            Linear(98, 196),
            ReLU(),
            NormalizedLinearWithResidual(196),
            BatchNorm1d(196),
            Linear(196, 392),
            ReLU(),
            NormalizedLinearWithResidual(392),
            BatchNorm1d(392),
            Linear(392, 784),
            NormalizedLinearWithResidual(784),
            Sigmoid()
        )
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.0001, betas=(0.5, 0.9))
        self.generator.to(device=DEVICE)

        # Output of discriminator is prediction of whether or not it is real 
        # (note it still needs to be passed through sigmoid to be normalized)
        self.discriminator = Sequential(
            NormalizedLinearWithResidual(784),
            BatchNorm1d(784),
            Linear(784, 392),
            ReLU(),
            NormalizedLinearWithResidual(392),
            BatchNorm1d(392),
            Linear(392, 1)
        )
        # self.discriminator = make_dense_network(self.flattened_image_size, 1, discriminator_hidden_layers, discriminator_layer_size)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.9))
        self.discriminator.to(device=DEVICE)

    def discriminator_lossfn(self, logits, labels, real_image=False, epsilon=1e-5):
        return -torch.mean(torch.log(epsilon + 1 - torch.abs(torch.sigmoid(logits) - labels)))
        
    def generator_lossfn(self, logits, epsilon=1e-5):
        return -torch.mean(torch.log(epsilon + torch.sigmoid(logits)))

    def gen_noise(self, batch_size = 1):
        return torch.randn(batch_size, self.noise_size, device=DEVICE)
    
    def gen_images(self, batch_size=1, noise=None):
        """
            Uses the generator to create batch_size number of images
        """
        if noise is not None:
            return self.generator(noise)
        else:
            return self.generator(self.gen_noise(batch_size))#.reshape(batch_size, 1, self.image_width, self.image_width)
    
    def mix_with_generated_images(self, real_images):
        # Presently the images are in two seperate groups–maybe shuffle them later?
        generated_images = self.gen_images(len(real_images))
        combined_images = torch.cat([real_images, generated_images])
        labels = torch.cat((torch.ones(len(real_images), device=DEVICE), 
                            torch.zeros(len(generated_images), device=DEVICE)))

        return combined_images, labels
        
    def evaluate(self, dataloader, print_stats=False):
        """
            Calculates the model's performance on the given data.
            Presently, only returns the discriminator's loss
            this should maybe return a tuple with the generator's loss too
        """
        # Variables for tracking stats as we iterate through the data in batches
        total_samples = 0
        correct_on_real_images = 0
        correct_on_generated_images = 0

        for batch in dataloader:
            # Get number correct on real images
            real_preds = self.discriminator(batch).squeeze(dim=1)
            correct_on_real_images += (real_preds > 0.5).sum()

            # Get number correct on generated images
            generated_preds = self.discriminator(self.gen_images(len(batch))).squeeze(dim=1)
            correct_on_generated_images += (generated_preds < 0.5).sum()
            
            # Track how many images we've seen
            total_samples += len(batch)
        
        # Calculate Accuracies
        real_correct_acc = correct_on_real_images / total_samples
        generated_correct_acc = correct_on_generated_images / total_samples

        if print_stats:
            print(f"Real Image Accuracy:      {real_correct_acc}")
            print(f"Generated Image Accuracy: {generated_correct_acc}")
        return real_correct_acc, generated_correct_acc
    
    def discriminator_train_step(self, real_images):
        self.discriminator_optimizer.zero_grad()

        generated_images = self.gen_images(len(real_images)) # Get images from the generator
        images = torch.cat([real_images, generated_images])

        # Labels for the samples
        labels = torch.cat((torch.ones(len(real_images), device=DEVICE), 
                            torch.zeros(len(generated_images), device=DEVICE)))
        
        # Compute predictions and update the parameters
        preds = self.discriminator(images).squeeze(dim = 1)
        loss = self.discriminator_lossfn(preds, labels)
        loss.backward()
        self.discriminator_optimizer.step()

        # Debugging assert
        assert not math.isnan(loss)
    
    def generator_train_step(self, batch_size):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        # Use generator to get minibatched input for discriminator
        images = self.gen_images(batch_size)

        # Get predictions from model, calculate loss, and update parameters
        preds = self.discriminator(images).squeeze(dim = 1)
        loss = self.generator_lossfn(preds) # Implicitly assumes it's trying to make these seem real
        assert not math.isnan(loss)
        loss.backward()
        self.generator_optimizer.step()
        # print(f"Generator Loss: {loss.item()}")
    
    def train(self, train_dataloader, val_dataloader = None, num_epochs=10, discrim_sub_iterations=1):
        # TODO: actually allow for discrim_sub_iterations != 1
        if discrim_sub_iterations != 1:
            raise Exception("discrim_sub_iterations > 1 not yet supported")
        
        tracked_noise = self.gen_noise(100)
        
        for epoch in range(num_epochs):
            torch.save(self.generator.state_dict(), f'models/g{epoch}.pth')
            torch.save(self.discriminator.state_dict(), f'models/d{epoch}.pth')

            # Evaluate model (also on validation set, if provided)
            self.discriminator.eval()
            self.generator.eval()
            print("-----------------------------------------------")
            print(f"After {epoch} training epochs:")
            self.evaluate(train_dataloader, print_stats=True)
            if val_dataloader is not None:
                print("Validation Performance:")
                self.evaluate(val_dataloader, print_stats=True)

            for i, noise in enumerate(tracked_noise):
                display_image(self.gen_images(noise = noise.unsqueeze(dim=0)).squeeze().clone().detach().to("cpu"), display=False, filename = f"testimgs/tracked_noise/{i}_{str(epoch).zfill(3)}.png")
            
            
            # Generator training loop--go through full dataset
            for training_images in train_dataloader:
                # Discriminator training loop
                self.discriminator.train()
                self.generator.eval()
                for _ in range(discrim_sub_iterations):
                    self.discriminator_train_step(training_images)

                # Generator training subroutine
                self.discriminator.eval()
                self.generator.train()
                self.generator_train_step(len(training_images))

if __name__ == "__main__":
    mnist_data_manager = DataManager(MNISTDataset())
    mnist_gan = GAN(noise_size=49, image_width=28)
    mnist_gan.train(mnist_data_manager.train(batch_size=64), num_epochs=1000)
