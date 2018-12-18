from vectors import *
import pygame

NET_FACTOR = 64 # Filters in convolutional layers is a multiple of this
IMAGE_SIZE = 128
MINI_BATCH_SIZE = 4
EPOCH_NUM = 15
UPDATE_SIZE = 2000
L1_WEIGHT = 100
THRESHOLD_VALUE = 0.4
LEARNING_RATE = 0.0001

# Gan which is used in the network (Change this)
NETWORK_FILE_NAME = r"C:\Users\corma\Google Drive\Year 12\Advanced Programming\cGAN - Assignment\samples_7_4000.tar"

GLOBAL_COLOUR = Vector3d(0, 255, 20)
GLOBAL_COLOUR_DARK = Vector3d(0, 255, 20)

# Data directory
DATA_DIR = r"C:\Users\corma\Documents\Programming\Datasets\ut-zap50k-images-square"
