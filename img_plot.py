import torch
import matplotlib.pyplot as plt
from math import sqrt

def display_image(img_tensor, filename = None, display = True):
    side_length = sqrt(len(img_tensor))
    if int(side_length) - side_length == 0:
        side_length = int(side_length)
        if filename is not None:
            norm = plt.Normalize(vmin=0, vmax=1)
            image = plt.cm.gray(norm(img_tensor.reshape((side_length,side_length))))
            plt.imsave(filename, image)
        if display:
            plt.imshow(img_tensor.reshape((side_length,side_length)), cmap = "gray", vmax = 1, vmin = 0)
            plt.show()
    else: 
        raise Exception("Tensor cannot be converted to square")

if __name__ == "__main__":
    x = torch.tensor([0,0.5,0.5,0,0.5,0,1,1,0])
    y = torch.tensor([0,0.5,0.5,0,0.5,0,0.5,0.5,0])
    display_image(x, "test.png")
    display_image(y, "test.png")