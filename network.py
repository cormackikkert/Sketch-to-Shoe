import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from constants import *

from timeit import default_timer as timer

import matplotlib.pyplot as plt

""" 
This module Implements patchGan and Unet generator shown here: 
https://github.com/phillipi/pix2pix
"""

def weights_init(m):
    """ Initialize parameters. Code from:
     https://github.com/pytorch/examples/blob/master/dcgan/main.py#L90-L96
     """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class UnetBlock(nn.Module):
    """ A UNET Block. Performs a convolution on an input, 
        passes it though a module (another unet block)
        Then deconvles it and concatenates the output with the original block
        This is done to shuttle important information across the network without
        losing it in the bottleneck layer """
    def __init__(self, outer, inner, module = None, innermost = False, outermost = False):
        super(UnetBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost

        # Going down the unet
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downconv = nn.Conv2d(outer, inner, kernel_size = 4, stride= 2, padding = 1)
        self.downnorm = nn.InstanceNorm2d(inner)

        # Going up the unet. Channels is multiplied by two as it is concatenated (unless it is the innermost layer)
        self.uprelu = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(inner * (1 if innermost else 2), outer, kernel_size = 4, stride = 2, padding = 1)
        self.upnorm = nn.InstanceNorm2d(outer)

        self.tanh = nn.Tanh()

        self.module = module

    def forward(self, orig):
        if self.outermost == True:
            x = self.downconv(self.downrelu(orig))
        else:
            x = self.downnorm(self.downconv(self.downrelu(orig)))
        x = self.module(x) if self.module else x
        if self.outermost == True:
            return self.tanh(self.upconv(x))

        x = self.upnorm(self.upconv(self.uprelu(x)))
        # Dont concatenate if this is the last layer
        return torch.cat([x, orig], 1) 


class Generator(nn.Module):
    """ Define the generator, which is composed of UNETS"""
    def __init__(self):
        super(Generator, self).__init__()
        self.unet6 = UnetBlock(8 * NET_FACTOR, 8 * NET_FACTOR, innermost=True)
        self.unet5 = UnetBlock(8 * NET_FACTOR, 8 * NET_FACTOR, module=self.unet6)
        self.unet4 = UnetBlock(4 * NET_FACTOR, 8 * NET_FACTOR, module=self.unet5)
        self.unet3 = UnetBlock(2 * NET_FACTOR, 4 * NET_FACTOR, module=self.unet4)
        self.unet2 = UnetBlock(1 * NET_FACTOR, 2 * NET_FACTOR, module=self.unet3)
        self.unet1 = UnetBlock(3, 1 * NET_FACTOR, module=self.unet2, outermost = True)

    def forward(self, x):
        return self.unet1(x)


class Discriminator(nn.Module):
    """ A Patch discriminator, Its like a convolutional neural network except
        it kinda stops half way, performs a sigmoid then averages all the results.
        The idea is that focusing on individual patches will increase the 
        texture quality produced by the generator """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, NET_FACTOR, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, True)

        self.conv2 = nn.Conv2d(NET_FACTOR, 2 * NET_FACTOR, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(2 * NET_FACTOR)
        self.relu2 = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(2 * NET_FACTOR, 4 * NET_FACTOR, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(4 * NET_FACTOR)
        self.relu3 = nn.LeakyReLU(0.2, True)

        self.conv4 = nn.Conv2d(4 * NET_FACTOR, 8 * NET_FACTOR, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(8 * NET_FACTOR)
        self.relu4 = nn.LeakyReLU(0.2, True)

        self.conv5 = nn.Conv2d(8 * NET_FACTOR, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid5 = nn.Sigmoid()

        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.sigmoid5(self.conv5(x))
        return x


class CGAN(nn.Module):
    """ The conditional GAN. Uses both a generator and discriminator
        And uses a seperate Adam optimizer for each."""

    def __init__(self):
        super(CGAN, self).__init__()
        self.generator = Generator().cuda()
        self.discriminator = Discriminator().cuda()

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.G_optimizer = optim.Adam(self.generator.parameters(), lr = 2e-4, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr = 5e-5, betas=(0.5, 0.999))

        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)
        
        self.BCE = nn.BCELoss()
        self.L1Loss = nn.L1Loss()

    def loss(self, input, is_real):
        target_tensor = self.real_label.expand_as(input) if is_real else self.fake_label.expand_as(input)
        return self.BCE(input, target_tensor.cuda())

    
    def generate(self, condition):
        noise = torch.rand(MINI_BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).cuda()
        return self.generator(torch.cat([condition, torch.cat([noise, noise], 1)], 1))

    def one_off_generate(self, condition):
        noise = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE).cuda()
        return self.generator(torch.cat([condition, torch.cat([noise, noise], 1)], 1))

    def backward_D(self, sketches, targets, fakes):
        # Get loss from fake image
        fake_input = torch.cat([sketches, fakes], 1)
        fake_pred = self.discriminator(fake_input.detach()).squeeze()

        loss_fake = self.loss(fake_pred, False)

        # Get loss from real image
        real_input = torch.cat([sketches, targets], 1)
        real_pred = self.discriminator(real_input.detach()).squeeze()
        loss_real = self.loss(real_pred, True)

        # return average of both losses
        return (loss_fake + loss_real) / 2

    def backward_G(self, sketches, targets, fakes):
        # Find loss from discriminator 
        fake_input = torch.cat([sketches, fakes], 1)
        fake_pred = self.discriminator(fake_input).squeeze()

        loss = self.loss(fake_pred, True)

        # Get L1 loss. (Difference betweeeen output and the target image)
        # Having an objective function helps stabilize the training and
        # Trains the GAN to avoid blurring
        lossl1 = self.L1Loss(fakes, targets) * L1_WEIGHT

        return loss + lossl1
    
    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def optimize_parameters(self, sketches, targets):
        fakes = self.generate(sketches)

        # Optimize D
        self.set_requires_grad(self.discriminator, True)
        self.D_optimizer.zero_grad()

        D_loss = self.backward_D(sketches, targets, fakes)
        D_loss.backward()

        self.D_optimizer.step()

        # Optimize G
        self.set_requires_grad(self.discriminator, False) # Dont update discriminator with generators gradient
        self.G_optimizer.zero_grad()

        G_loss = self.backward_G(sketches, targets, fakes)
        G_loss.backward()

        self.G_optimizer.step()

        return G_loss, D_loss

# Code to test time complexity of the GAN with respect to the input image sizes
""" 
plt.title("Image generation time")

gen = Generator()

all_times = []
cases = []
for n in range(7, 12, 4):
    image_size = 2 ** n
    cases.append(image_size)
    avg_times = []
    for trial in range(100):
        image = torch.rand(1, 3, image_size, image_size)
        start_time = timer()
        gen(image)
        total_time = timer() - start_time
        avg_times.append(total_time)
    all_times.append(sum(avg_times) / len(avg_times))

plt.plot(cases, all_times, label="image generation size")
plt.show()
"""