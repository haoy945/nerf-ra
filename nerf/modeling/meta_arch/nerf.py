import torch.nn as nn

from ..network import NerfMlp


class NeRF(nn.Module):
    """
    There are about 4 steps to render an image.
    First, we should sample the points and each point is represented by (x,y,z,theta,phi).
    Then we map the vector to a higher dimensional space using high frequency functions.
    Thirdly, we have neural network to operate on the coodinates to get the color and 
    density for every point.
    Last, we use volume rendering to composite these values into an image.

    At each optimization iteration, we randomly samplea batch of camera rays from the 
    set of all pixels in the dataset. Our loss is simply the total squared error 
    betweenthe rendered and true pixel colors.
    """

    def __init__(self, *args):
        super().__init__()

    def forward(self):
        pass

    def inference(self):
        pass
