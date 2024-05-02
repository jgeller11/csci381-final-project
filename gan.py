import tqdm

def train(generator, discriminator, 
          generator_noise_gen, discriminator_noise_gen, 
          num_iterations=10, discrim_sub_iterations=1):
    # Generator training loop
    for _ in tqdm(num_iterations):
        # Discriminator training loop
        for _ in range(discrim_sub_iterations):
            ## Getting Inputs
            # Get minibatch of noise inputs
            # Use generator to get minibatched input for discriminator
            # Get minibatch of real examples from training data

            ## Updating parameters
            # Use combined inputs to calculate gradients and take a step
            pass
        # Sample minibatch of noise
        # Pass noise through generator
        #  -> pass result through discriminator
        # calculate gradients and update generator (NOT discriminator)