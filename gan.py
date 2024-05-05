import math
import torch
from torch.nn import Parameter, Sequential, BatchNorm1d, Linear, ReLU, Sigmoid, Tanh
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
    def __init__(self, noise_size, image_width, 
                 generator_hidden_layers=10, generator_layer_size=100,
                 discriminator_hidden_layers=10, discriminator_layer_size=100):
        print("NOTE: Generator Hidden Layers and Size currently unused!")

        self.noise_size = noise_size
        # We assume square images
        self.flattened_image_size = image_width * image_width

        H1 = self.flattened_image_size // 4
        H2 = 2 * self.flattened_image_size // 4
        H3 = 3 * self.flattened_image_size // 4
        # Clamp the output of the generator, so it's a valid image
        self.generator = Sequential(
            BatchNorm1d(noise_size),
            Linear(noise_size, H1),
            ReLU(),
            NormalizedLinearWithResidual(H1),
            BatchNorm1d(H1),
            Linear(H1, H2),
            ReLU(),
            NormalizedLinearWithResidual(H2),
            BatchNorm1d(H2),
            Linear(H2, H3),
            ReLU(),
            NormalizedLinearWithResidual(H3),
            BatchNorm1d(H3),
            Linear(H3, self.flattened_image_size),
            Sigmoid()
        )
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.0001, betas=(0.5, 0.9))
        self.generator.to(device=DEVICE)

        # Output of discriminator is prediction of whether or not it is real 
        # (note it still needs to be passed through sigmoid to be normalized)
        self.discriminator = make_dense_network(self.flattened_image_size, 1, discriminator_hidden_layers, discriminator_layer_size)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.9))
        self.discriminator.to(device=DEVICE)

        epsilon = 1e-8
        self.discriminator_lossfn = (lambda logit, label : -torch.mean(torch.log(epsilon + 1 - torch.abs(torch.sigmoid(logit) - label))))
        self.generator_lossfn = (lambda logit : -torch.mean(torch.log(epsilon + torch.sigmoid(logit))))

    def gen_noise(self, batch_size = 1):
        return torch.randn(batch_size, self.noise_size, device=DEVICE)
    
    def gen_images(self, batch_size=1):
        """
            Uses the generator to create batch_size number of images
        """
        return self.generator(self.gen_noise(batch_size))
    
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
    
    def discriminator_train_step(self, training_images):
        self.discriminator_optimizer.zero_grad()
        # Use generator to get minibatched input for discriminator
        images, labels = self.mix_with_generated_images(training_images)

        # Get predictions from model, calculate loss, and update parameters
        preds = self.discriminator(images).squeeze(dim = 1)
        loss = self.discriminator_lossfn(preds, labels)
        assert not math.isnan(loss)
        loss.backward()
        self.discriminator_optimizer.step()
        # print(f"Discriminator Loss: {loss.item()}")
    
    def generator_train_step(self, batch_size):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        # Use generator to get minibatched input for discriminator
        images = self.gen_images(batch_size).squeeze(dim = 1)

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
        
        for epoch in range(num_epochs):
            torch.save(self.generator.state_dict(), f'models/generator.pth')
            torch.save(self.discriminator.state_dict(), f'models/discriminator.pth')

            # Evaluate model (also on validation set, if provided)
            self.discriminator.eval()
            self.generator.eval()
            print("-----------------------------------------------")
            print(f"After {epoch} training epochs:")
            self.evaluate(train_dataloader, print_stats=True)
            if val_dataloader is not None:
                print("Validation Performance:")
                self.evaluate(val_dataloader, print_stats=True)

            display_image(self.gen_images().squeeze().clone().detach().to("cpu"), display=False, filename = f"testimgs/{epoch}.png")
            
            
            # Generator training loop--go through full dataset
            self.discriminator.train()
            for training_images in train_dataloader:
                # Discriminator training loop
                self.discriminator.train()
                for _ in range(discrim_sub_iterations):
                    self.discriminator_train_step(training_images)

                # Generator training subroutine
                self.discriminator.eval()
                self.generator.train()
                self.generator_train_step(len(training_images))

if __name__ == "__main__":
    mnist_data_manager = DataManager(MNISTDataset())
    mnist_gan = GAN(noise_size=100, image_width=28, 
              discriminator_hidden_layers=6, discriminator_layer_size=100)
    mnist_gan.train(mnist_data_manager.train(batch_size=32), num_epochs=1000)
