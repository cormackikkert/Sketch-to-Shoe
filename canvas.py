from canvas import *
from constants import *
from gui import *
from events import *
import math
import numpy as np
import pygame

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import scipy.misc

from data import *

def dist(x1, y1, x2, y2):
    # Faster function for calculating distances
    return abs(x1 - x2) + abs(y1 - y2)

class Canvas(Rect):
    """ A object that allows the user to draw on it by clicking """
    @EventAdder
    def __init__(self, canvas_size, x, y, display, **kwargs):
        super().__init__(x, y, canvas_size, canvas_size)
        self.canvas_size = canvas_size
        self.clicked = False
        self.state = True # Draw if true, erase if False
        self.strokeWidth = 2
        self.box_size = self.canvas_size // IMAGE_SIZE
        self.display  = display
        self.matrix = [[0 for i in range(IMAGE_SIZE)] for i in range(IMAGE_SIZE)]   

    def set_stroke(self, x):
        self.strokeWidth = x

    def set_state(self, x):
        self.state = x

    def clear(self):
        self.matrix = [[0 for i in range(IMAGE_SIZE)] for i in range(IMAGE_SIZE)]   
        self.evManager.push(SketchEvent())
        
    def notify(self, event):
        if isinstance(event, TickEvent):
            if self.clicked:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Check if each pixel is close enough to the mosue to be effected by it
                # Not very efficient but works well for this relatively small grid
                for x in range(IMAGE_SIZE):
                    for y in range(IMAGE_SIZE):
                        if dist(x * self.box_size + self.x + self.box_size / 2, y * self.box_size + self.y + self.box_size / 2, mouse_x, mouse_y) < self.strokeWidth:
                            self.matrix[y][x] = int(self.state)


        if isinstance(event, MouseClick):
            if event.action == 'P' and self.collidepoint(Vector2d(*pygame.mouse.get_pos())):
                self.clicked = True

            # If the user has stopped holding the mouse update the other canvas showing the realistic shoe
            elif self.clicked == True:
                self.clicked = False
                self.evManager.push(SketchEvent())
        
                

        if isinstance(event, RenderEvent):
            """ Render the canvas, drawing black boxes for black pixels and white boxes for white pixels """
            for x in range(IMAGE_SIZE):
                for y in range(IMAGE_SIZE):
                    colour = (0, 0, 0) if self.matrix[y][x] else (255, 255, 255)
                    self.display.fill(colour, rect = (self.x + x * self.box_size,
                                                      self.y + y * self.box_size,
                                                      self.box_size, self.box_size))            
            

class Preview(Rect):
    """ Takes the sketch of the shoe on the canvas and turns it into an image before displaying it """
    @EventAdder
    def __init__(self, canvas_size, x, y, display, canvas, network, **kwargs):
        super().__init__(x, y, canvas_size, canvas_size)
        self.display = display
        self.canvas = canvas
        self.network = network

        self.sketch_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.surf = pygame.surface.Surface((self.w, self.h))
        self.surf.fill((255, 255, 255))

    def createSketch(self):
        d = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(np.array(self.canvas.matrix)), 0).cuda(), 0)
        img = self.network.one_off_generate(d)
        
        img = img[0] / 2 + 0.5 # unnormalize
        array = np.transpose(img.detach().cpu().numpy(), (2, 1, 0)) # Re arrange axis
        
        array = array * 255
        
        # Constrain values that are too small or too big
        array[array > 255] = 255
        array[array < 0] = 0

        self.surf = pygame.surfarray.make_surface(array)

        self.surf = pygame.transform.scale(self.surf, (512, 512))

    def save(self, filename):
        d = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(np.array(self.canvas.matrix)), 0).cuda(), 0)
        img = self.network.one_off_generate(d)
        
        img = img[0] / 2 + 0.5 # unnormalize
        array = np.transpose(img.detach().cpu().numpy(), (2, 1, 0)) # Re arrange axis
        
        array = array * 255

        # Constrain values
        array[array > 255] = 255
        array[array < 0] = 0

        # Save image
        scipy.misc.imsave(filename + ".jpg", array)

    def notify(self, event):
        if isinstance(event, SketchEvent):
            self.createSketch()

        if isinstance(event, RenderEvent):
            self.display.blit(self.surf, (self.x, self.y))
            
            

    