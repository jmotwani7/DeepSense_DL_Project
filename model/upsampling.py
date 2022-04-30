import torch.nn as nn
import torch
from torch.nn.functional import pad
import numpy as np


class FastUpConvolution(nn.Module):
    """
    Fast Convolutions
    """

    def __init__(self, in_channels, out_channels):
        """Initialize the Fast convolution using the in and out channels"""
        super(FastUpConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2))

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    # https://github.com/iapatil/depth-semantic-fully-conv
    def prepare_indices(self, before, row, col, after, dims):
        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
        x_0 = torch.from_numpy(x0.reshape([-1]))
        x_1 = torch.from_numpy(x1.reshape([-1]))
        x_2 = torch.from_numpy(x2.reshape([-1]))
        x_3 = torch.from_numpy(x3.reshape([-1]))
        if torch.cuda.is_available():
          x_0 = x_0.cuda()
          x_1 = x_1.cuda()
          x_2 = x_2.cuda()
          x_3 = x_3.cuda()

        linear_indices = x_3 + dims[3] * x_2 + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
        return linear_indices

    # # interleaving operation
    # def interleave_helper(self, tensors, axis):
    #     tensor_shape = list(tensors[0].size())
    #     tensor_shape[axis + 1] *= 2
    #     return torch.stack(tensors, axis + 1).view(tensor_shape)

    # https://github.com/iapatil/depth-semantic-fully-conv
    def interleave_tensors(self, out1, out2, out3, out4, batch_size):
        out1 = out1.permute(0, 2, 3, 1)
        out2 = out2.permute(0, 2, 3, 1)
        out3 = out3.permute(0, 2, 3, 1)
        out4 = out4.permute(0, 2, 3, 1)

        dims = out1.size()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)

        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)
        C_row_indices = range(0, dim1, 2)
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)
        D_col_indices = range(1, dim2, 2)

        all_indices_before = range(int(batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims)
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims)

        A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
        B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
        C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
        D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

        size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0] + C_linear_indices.size()[0] + D_linear_indices.size()[0]

        Y_flat = torch.FloatTensor(size_).zero_()
        if torch.cuda.is_available():
          Y_flat = Y_flat.cuda()

        Y_flat.scatter_(0, A_linear_indices.squeeze(), A_flat.data)
        Y_flat.scatter_(0, B_linear_indices.squeeze(), B_flat.data)
        Y_flat.scatter_(0, C_linear_indices.squeeze(), C_flat.data)
        Y_flat.scatter_(0, D_linear_indices.squeeze(), D_flat.data)

        Y = Y_flat.view(-1, dim1, dim2, dims[3])
        return torch.autograd.Variable(Y.permute(0, 3, 1, 2))

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor : Tensor of shape N, C, H, W

        Returns
        -------
        output_tensor : Returns an output tensor of shape N, C, 2*H, 2*W
        """
        conved33 = self.conv1(pad(input_tensor, (1, 1, 1, 1)))
        conved23 = self.conv2(pad(input_tensor, (1, 1, 1, 0)))
        conved32 = self.conv3(pad(input_tensor, (1, 0, 1, 1)))
        conved22 = self.conv4(pad(input_tensor, (1, 0, 1, 0)))
        out_interleaved = self.interleave_tensors(conved33, conved23, conved32, conved22, input_tensor.shape[0])
        normed = self.batch_norm(out_interleaved)
        return self.relu(normed)
