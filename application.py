import canvas, events, gui, vectors, constants
import pygame 
import torch

import network

pygame.display.set_caption('Shoe generator')
clock = pygame.time.Clock()
pygame.init()

gameDisplay = pygame.display.set_mode((1250, 600), pygame.DOUBLEBUF) # Create display
controller = events.EventManager()

cGAN = network.CGAN().cuda()

# Load cGAN parameters
checkpoint = torch.load(constants.NETWORK_FILE_NAME)
cGAN.generator.load_state_dict(checkpoint['g_state_dict'])
cGAN.discriminator.load_state_dict(checkpoint['d_state_dict'])

draw_canvas = canvas.Canvas(512, 20, 44, gameDisplay, evManager=controller)
render_canvas = canvas.Preview(512, 1250 - 20 - 512, 44, gameDisplay, draw_canvas, cGAN, evManager=controller)

# Define input box here so python doesnt immediatly put it in garbage after creating it
input_box = False 

def input_func():
    global input_box
    input_box = gui.InputBox(625 - 240, 200, 480, 200, (255, 200, 0), lambda n: render_canvas.save(n), 'Save file as:', gameDisplay, evManager=controller)
    controller.registerListener(input_box)

# Define objects used in scene
sketch_objects = [
    gui.TextBoxEvent(537, 44, 176, 30, "Stroke Weight", gameDisplay, evManager=controller),
    gui.Scroller(625, 90, 200,  [2, 15], draw_canvas.set_stroke, gameDisplay, evManager=controller),
    gui.TextBoxEvent(537, 300, 176, 30, "Draw / Erase", gameDisplay, evManager=controller),
    gui.Toggle(537, 340, 176, 60, draw_canvas.set_state, gameDisplay, evManager=controller),
    events.KeyboardController(evManager=controller),
    draw_canvas,
    render_canvas,
    gui.Button('SAVE', input_func, gameDisplay, evManager=controller, rect=gui.Rect(537, 410, 176, 40)),
    gui.Button('CLEAR', draw_canvas.clear, gameDisplay, evManager=controller, rect=gui.Rect(537, 460, 176, 40))


]

for obj in sketch_objects:
    controller.registerListener(obj)

while True:
    gameDisplay.fill((0, 0, 0))
    
    controller.push(events.TickEvent())
    controller.push(events.RenderEvent())

    pygame.display.update()

    clock.tick(30)

