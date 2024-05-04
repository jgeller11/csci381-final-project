import torch
import matplotlib.pyplot as plt
from math import sqrt

def display_image(img_tensor, filename = None, display = True):
    side_length = sqrt(len(img_tensor))
    if int(side_length) - side_length == 0:
        side_length = int(side_length)
        square_img = img_tensor.float().reshape((side_length,side_length))
        scaled_image_tensor = torch.squeeze(torch.squeeze(
            torch.nn.Upsample(scale_factor=30, mode='nearest')(
            torch.unsqueeze(torch.unsqueeze(square_img, dim = 0), dim = 0)
            ), dim = 0), dim = 0)
        if filename is not None:
            norm = plt.Normalize(vmin=0, vmax=1)
            image = plt.cm.gray(norm(scaled_image_tensor))
            plt.imsave(filename, image, dpi = 10)
        if display:
            plt.imshow(scaled_image_tensor, cmap = "gray", vmax = 1, vmin = 0)
            plt.show()
    else: 
        raise Exception("Tensor cannot be converted to square")

if __name__ == "__main__":
    x = torch.tensor([0,0.5,0.5,0,0.5,0,1,1,0])
    y = torch.tensor([0,0.5,0.5,0,0.5,0,0.5,0.5,0])
    display_image(x, filename="testimgs/0.png", display = False)
    # display_image(x, "test.png")
    # display_image(y, "test.png")