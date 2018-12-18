from constants import *
from data import *

import network
import numpy as np
import matplotlib.pyplot as plt

import torch

# Initialize dataset
dataset = SketchImageDataset()
trainloader = torch.utils.data.DataLoader(dataset, batch_size = MINI_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)


def save_checkpoint(state, filename):
    """ Implement saving loading of GAN with the method shown here: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2"""
    torch.save(state, filename + '.tar')

def img_show(sketches, fakes, images, filename = None):
    """ Function for showing sketchs, the images created by the GAN and the actual images """

    fig = plt.figure(figsize = (IMAGE_SIZE, IMAGE_SIZE))
    cols = MINI_BATCH_SIZE
    rows = 3

    for i in range(cols):
        # Show sketches on top row
        sketch = sketches[i] / 2 + 0.5 # unnormalize
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(sketch.numpy()[0], interpolation='nearest')

        # Show faked images on the middle row
        fake = fakes[i] / 2 + 0.5 # unnormalize
        fig.add_subplot(rows, cols, cols + i + 1)
        plt.imshow(np.transpose(fake.numpy(), (1, 2, 0)), interpolation='nearest')

        # Show actual images on the bottom row
        image = images[i] / 2 + 0.5 # unnormalize
        fig.add_subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)), interpolation='nearest')


    # Save the picture
    if filename:
        fig.savefig(filename + '.png', bbox_inches='tight')

if __name__ == "__main__":
    # Initialize a GAN on the gpu
    cGAN = network.CGAN().cuda()

    # Load a previous GAN to keep training it
    checkpoint = torch.load(NETWORK_FILE_NAME)
    cGAN.generator.load_state_dict(checkpoint['g_state_dict'])
    cGAN.discriminator.load_state_dict(checkpoint['d_state_dict'])
    cGAN.G_optimizer.load_state_dict(checkpoint['g_optimizer'])
    cGAN.D_optimizer.load_state_dict(checkpoint['d_optimizer'])

    # Variables to store the average losses for the generator and discriminator
    G_running_loss = 0
    D_running_loss = 0

    for epoch in range(4, EPOCH_NUM):
        for i, data in enumerate(trainloader):

            images, sketches = data
            images, sketches = images.cuda(), sketches.cuda()

            # Update GAN and keep track of losses
            D_loss, G_loss = cGAN.optimize_parameters(sketches, images)

            D_running_loss += D_loss.item()
            G_running_loss += G_loss.item()
            
            if i % UPDATE_SIZE == UPDATE_SIZE - 1:
                details = f'samples_{epoch + 1}_{i + 1}'

                # Save image of generation and print information about losses
                img_show(sketches.cpu(), cGAN.generate(sketches).detach().cpu(), images.cpu(), filename = details)
                print(f"{epoch + 1}, {i + 1}, D:{D_running_loss / UPDATE_SIZE}, G:{G_running_loss / UPDATE_SIZE}")
            
                save_checkpoint({
                    'epoch': epoch + 1,
                    'g_state_dict': cGAN.generator.state_dict(),
                    'd_state_dict': cGAN.discriminator.state_dict(),
                    'g_optimizer':  cGAN.G_optimizer.state_dict(),
                    'd_optimizer': cGAN.D_optimizer.state_dict()
                }, details)

                D_running_loss = 0
                G_running_loss = 0

