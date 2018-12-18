from constants import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFilter

class SobelOperator:
    """ Takes in image and performs the sobel operator on the image. Good for edge detection,
        However I found that the width of edges can vary alot, so it was not used for training"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()]
        )

        # Create filters for change in x and y direction
        GX_weights = torch.tensor(([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]]))

        self.GX_filter = nn.Conv2d(1, 1, 3, padding=1, stride = 1)
        self.GX_filter.weight = nn.Parameter(GX_weights.float().unsqueeze(0).unsqueeze(0))

        GY_weights = torch.tensor(([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]]))
        self.GY_filter = nn.Conv2d(1, 1, 3, padding=1, stride = 1)
        self.GY_filter.weight = nn.Parameter(GY_weights.float().unsqueeze(0).unsqueeze(0))

        # Filters are not going to be updated so their computaions do not have to be tracked
        for p in self.GX_filter.parameters():
            p.requires_grad = False
        for p in self.GY_filter.parameters():
            p.requires_grad = False

    def forward(self, image):
        # Convert image to black and white
        image = self.transform(image)
        # Add batch dimension
        image = image.unsqueeze(0)
        # Find change in x and y direction
        GX = self.GX_filter(image)
        GY = self.GY_filter(image)
        # Find gradient magnitude
        output = torch.sqrt(torch.pow(GX, 2) + torch.pow(GY, 2))
        # Remove batch dimension
        return output[0]

class SketchImageDataset(Dataset):
    """ Dataset that produces an image, sketch tuple. Does this
        by extending torch's Dataset class"""
    def  __init__(self):     
        self.data_dir = DATA_DIR
        self.sketch_creator = SobelOperator()

        # Transform for images
        self.image_transform = transforms.Compose(
            [transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
        # Transform for sketches
        self.sketch_transform = transforms.Compose(
            [transforms.Grayscale(),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
        self.trainset = torchvision.datasets.ImageFolder(DATA_DIR)

    def __getitem__(self, index):
        # Get image but ignore label from the dataset
        image, _ = self.trainset.__getitem__(index)
        
        # Create the sketch of the image
        sketch = self.sketch_creator.forward(image)
        sketch = self.sketch_transform(image.filter(ImageFilter.FIND_EDGES))
        
        sketch[sketch <= THRESHOLD_VALUE] = 0
        sketch[sketch > THRESHOLD_VALUE] = 1

        # Return the image and sketch
        return self.image_transform(image), sketch
        
    def __len__(self):
        return len(self.trainset)