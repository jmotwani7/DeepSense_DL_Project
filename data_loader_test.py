import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms.functional as tf
import pandas as pd
import torch.nn as nn
from models import resnet32,UNet
from tqdm import tqdm
import glob
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from dataloader import DataGenerator

np.random.seed(100)



root = "./datasets/cityscapes"
augmentations = Compose([Scale(512), RandomRotate(10)])
data_train = DataGenerator(root, split="train", is_transform=True, augmentations = augmentations)
data_test = DataGenerator(root, split="test", is_transform=True, augmentations = augmentations)
batch_size = 32
train_loader = Data.DataLoader(data_train, batch_size, num_workers=0)
test_loader = Data.DataLoader(data_test)



