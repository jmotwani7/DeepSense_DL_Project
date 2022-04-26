import torch.nn as nn
import torch


class FastUpConvolution(nn.Module):
    """
    Fast Convolutions
    """

    def __init__(self, in_channels, out_channels):
        """Initialize the Fast convolution using the in and out channels"""
        super(FastUpConvolution).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2))

    def interleave_conv_output(self, conv33, conv23, conv32, conv22):
        """interleaves the 4 tensors into a single output tensor"""
        interleave1 = torch.stack((conv33, conv32), dim=0).view(2, 4).t().contiguous().view(2, 4)
        interleave2 = torch.stack((conv23, conv22), dim=0).view(2, 4).t().contiguous().view(2, 4)
        return torch.stack((interleave1, interleave2), dim=1).view(4, 2).t().contiguous().view(4, 2)

    def forward(self, input_tensor):
        conved33 = self.conv1(input_tensor)
        conved23 = self.conv2(input_tensor)
        conved32 = self.conv3(input_tensor)
        conved22 = self.conv4(input_tensor)
        return self.interleave_conv_output(conved33, conved23, conved32, conved22)
