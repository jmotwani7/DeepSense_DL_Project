import torch
import pytest
from model.upsampling import FastUpConvolution


def test_out_shape():
    input_tensor = torch.rand((1, 3, 4, 4))
    upconv_module = FastUpConvolution(3, 3)

    res = upconv_module.forward(input_tensor)
    assert res.shape[0] == 8
    assert res.shape[1] == 8


def test_interleave_conv_output():
    conv22 = torch.ones((2, 2))
    conv32 = torch.ones((3, 2)) * 2

    conv23 = torch.ones((2, 3)) * 3
    conv33 = torch.ones((3, 3)) * 4
    upconv_module = FastUpConvolution(3, 3)
    res = upconv_module.interleave_tensors(conv22, conv32, axis=0).view(2, 5).t().contiguous().view(5, 2)
    res2 = upconv_module.interleave_tensors(conv23, conv33, axis=0).view(3, 5).t().contiguous().view(5, 3)
    print(res)
    print(res2)


def test_interleaving_dims():
    a = torch.arange(0, 3).unsqueeze(0).unsqueeze(2).expand(2, -1, 4)
    b = torch.arange(3, 6).unsqueeze(0).unsqueeze(2).expand(2, -1, 4)

    c = torch.stack((a, b), dim=2)
    print(c.view(2, 6, 4))
