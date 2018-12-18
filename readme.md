# Sketch to Image 
This small app demonstrates the applications of GANs in assistive design. It converts a sketch of a shoe into an image of a shoe, using the architecture detailed in pix2pix.
Requirements:
For training download ut-zap50k-images-square zip file, and reference it in constants.py.
PIL, Pytorch, TorchVision, Numpy, Pygame, SciPy, Matplotlib.
Pytorch required CUDA to use the GPU. If you don’t have CUDA just get rid of the cuda methods in the code (.cuda()).
During training matplotlib is also used.
If you are using the network without the GUI you might find the built-in batch dimension unitintuibe. To get around this before putting any image through the generator call the unsqueeze(0) method on it to add a batch dimension, which will allow the image to be processed. 
Due to how convolutional neural networks work the image size is represented as a (mini_batch_size, channel, image_size, image_size). For pygame and matplotlib there is no minibatch_dimension, and images dimensions are represented in the more traditional format: (image_size, image_size, channel).
To fix this index the shape to get a single image and call numpy.transpose to switch the dimensions around. This is done in training.py and in the canvas class.

## Usage
To run the app, execute application.py

### Drawing
To draw click and hold on the square on the right. Once the mouse button is released the image on the right updates, with an image that represents the sketch being turned into a shoe

![image](https://user-images.githubusercontent.com/35324619/50137572-a6c6da80-02ef-11e9-83c7-7e8fa59bcaaf.png)

This slider (below) adjusts the stroke weight. Click and drag on the rectangle to change the stroke weight, with a higher position corresponding to a thicker stroke.  The stroke width for the lowest, highest and middle values are shown below.

![image](https://user-images.githubusercontent.com/35324619/50137583-afb7ac00-02ef-11e9-979f-5b2ba69153aa.png)
![image](https://user-images.githubusercontent.com/35324619/50137590-b34b3300-02ef-11e9-9b87-18090a61699c.png)

This toggle (below) allows you to both draw and erase from the canvas. When the toggle is on the left clicking and dragging on the canvas draws. When it is on the right clicking and dragging erases.
Click anywhere on the rectangle to change the mode.

![image](https://user-images.githubusercontent.com/35324619/50137592-b514f680-02ef-11e9-8ae3-1fa0716da3e0.png)

Clicking on the save button opens up a prompt where the image of the shoe (right box) will be saved as a jpg. Type what the filename should be then hit enter to save the image.

![image](https://user-images.githubusercontent.com/35324619/50137598-b80fe700-02ef-11e9-82c1-2df31d5b5f77.png)
![image](https://user-images.githubusercontent.com/35324619/50137603-bba36e00-02ef-11e9-92d2-5be4007a2593.png)

The last button (below) clears the canvas (box on the right):

![image](https://user-images.githubusercontent.com/35324619/50137606-be9e5e80-02ef-11e9-9107-e54240d7421b.png)

# Update
I added an activation function on the last layer of the neural network, which made the training process much faster, and made the outputs much more realistic:

![image](https://user-images.githubusercontent.com/35324619/50137608-c0682200-02ef-11e9-85a2-a320f40e1323.png)
