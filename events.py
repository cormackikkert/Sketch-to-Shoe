import pygame

# Base event class
class Event:
    def __init__(self):
        self.name = type(self).__name__

    def __str__(self):
        return self.name

class TickEvent(Event):
    pass

class RenderEvent(Event):
    pass

class SketchEvent(Event):
    pass

class MouseClick(Event):
    def __init__(self, type, action):
        super().__init__()
        self.type = type
        self.action = action

class KeyEvent(Event):
    def __init__(self, type, action):
        super().__init__()
        self.type = type
        self.action = action

class EventManager:
    """ Stores objects and notifies them about events that occur"""
    def __init__(self):
        from weakref import WeakKeyDictionary
        self.listeners = WeakKeyDictionary()

    def registerListener(self, listener):
        self.listeners[listener] = 1
        try:
            listener.register()
        except:
            pass

    def unregisterListener(self, listener):
        if listener in self.listeners.keys():
            del self.listeners[listener]

    def push(self, event):
        for listener in sorted(self.listeners.keys(), key=lambda x: x.layer, reverse = isinstance(event, RenderEvent)):
            if listener.notify(event):
                break

def EventAdder(func):
    """ registers an objects with the event manager.
        Since every object does this I found it easier to make it a 
        decorator """
    def wrapper(self, *args, **kwargs):
        self.evManager = kwargs['evManager']
        self.evManager.registerListener(self)
        self.layer = 0
        
        func(self, *args, **kwargs)

        # If a layer is specified, defines the objects layer
        # Overwrites what was in the objects constructor
        if 'layer' in kwargs:
            self.layer=kwargs['layer']
    return wrapper


class KeyboardController:
    """ Object that handles all keyboard inputs, and notifies
        the event manager about them """
    @EventAdder
    def __init__(self, **kwargs):
        # Has a low layer so that it is the first to react to a events
        # Useful for tick events, since you want to react to key presses early in a frame
        self.layer = -2

    # Checks to see if any buttons were clicked or released, and if so, fires events for them
    def notify(self, event):
        if isinstance(event, TickEvent):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Quiting the window is discouraged, since data will only be saved if you exit the game using the button
                    # You can only do it like that since globalachievements (the object that saves the data) is created
                    # in the constants file. The constants file imports from here so importing the Constants file will
                    # cause circular imports.
                    pygame.quit()

                # Mouse events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.evManager.push(MouseClick('LEFT', 'P'))
                    if event.button == 3:
                        self.evManager.push(MouseClick('RIGHT', 'P'))

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.evManager.push(MouseClick('LEFT', 'R'))
                    if event.button == 3:
                        self.evManager.push(MouseClick('RIGHT', 'R'))


                # Key press events
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_UP:
                        self.evManager.push(KeyEvent('UP', 'P'))

                    if event.key == pygame.K_DOWN:
                        self.evManager.push(KeyEvent('DOWN', 'P'))

                    if event.key == pygame.K_LEFT:
                        self.evManager.push(KeyEvent('LEFT', 'P'))

                    if event.key == pygame.K_RIGHT:
                        self.evManager.push(KeyEvent('RIGHT', 'P'))

                    if event.key == pygame.K_q:
                        self.evManager.push(KeyEvent('Q', 'P'))

                    if event.key == pygame.K_w:
                        self.evManager.push(KeyEvent('W', 'P'))

                    if event.key == pygame.K_e:
                        self.evManager.push(KeyEvent('E', 'P'))

                    if event.key == pygame.K_r:
                        self.evManager.push(KeyEvent('R', 'P'))

                    if event.key == pygame.K_t:
                        self.evManager.push(KeyEvent('T', 'P'))

                    if event.key == pygame.K_y:
                        self.evManager.push(KeyEvent('Y', 'P'))

                    if event.key == pygame.K_u:
                        self.evManager.push(KeyEvent('U', 'P'))

                    if event.key == pygame.K_i:
                        self.evManager.push(KeyEvent('I', 'P'))

                    if event.key == pygame.K_o:
                        self.evManager.push(KeyEvent('O', 'P'))

                    if event.key == pygame.K_p:
                        self.evManager.push(KeyEvent('P', 'P'))

                    if event.key == pygame.K_a:
                        self.evManager.push(KeyEvent('A', 'P'))

                    if event.key == pygame.K_s:
                        self.evManager.push(KeyEvent('S', 'P'))

                    if event.key == pygame.K_d:
                        self.evManager.push(KeyEvent('D', 'P'))

                    if event.key == pygame.K_f:
                        self.evManager.push(KeyEvent('F', 'P'))

                    if event.key == pygame.K_g:
                        self.evManager.push(KeyEvent('G', 'P'))

                    if event.key == pygame.K_h:
                        self.evManager.push(KeyEvent('H', 'P'))

                    if event.key == pygame.K_j:
                        self.evManager.push(KeyEvent('J', 'P'))

                    if event.key == pygame.K_k:
                        self.evManager.push(KeyEvent('K', 'P'))

                    if event.key == pygame.K_l:
                        self.evManager.push(KeyEvent('L', 'P'))

                    if event.key == pygame.K_z:
                        self.evManager.push(KeyEvent('Z', 'P'))

                    if event.key == pygame.K_x:
                        self.evManager.push(KeyEvent('X', 'P'))

                    if event.key == pygame.K_c:
                        self.evManager.push(KeyEvent('C', 'P'))

                    if event.key == pygame.K_v:
                        self.evManager.push(KeyEvent('V', 'P'))

                    if event.key == pygame.K_b:
                        self.evManager.push(KeyEvent('B', 'P'))

                    if event.key == pygame.K_n:
                        self.evManager.push(KeyEvent('N', 'P'))

                    if event.key == pygame.K_m:
                        self.evManager.push(KeyEvent('M', 'P'))

                    if event.key == pygame.K_MINUS:
                        self.evManager.push(KeyEvent('MINUS', 'P'))

                    if event.key == pygame.K_EQUALS:
                        self.evManager.push(KeyEvent('EQUALS', 'P'))

                    if event.key == pygame.K_BACKSPACE:
                        self.evManager.push(KeyEvent('BACKSPACE', 'P'))

                    if event.key == pygame.K_RETURN:
                        self.evManager.push(KeyEvent('ENTER', 'P'))

                # Key release events
                if event.type == pygame.KEYUP:

                    if event.key == pygame.K_UP:
                        self.evManager.push(KeyEvent('UP', 'R'))

                    if event.key == pygame.K_DOWN:
                        self.evManager.push(KeyEvent('DOWN', 'R'))

                    if event.key == pygame.K_LEFT:
                        self.evManager.push(KeyEvent('LEFT', 'R'))

                    if event.key == pygame.K_RIGHT:
                        self.evManager.push(KeyEvent('RIGHT', 'R'))

                    if event.key == pygame.K_w:
                        self.evManager.push(KeyEvent('W', 'R'))

                    if event.key == pygame.K_a:
                        self.evManager.push(KeyEvent('A', 'R'))

                    if event.key == pygame.K_s:
                        self.evManager.push(KeyEvent('S', 'R'))

                    if event.key == pygame.K_d:
                        self.evManager.push(KeyEvent('D', 'R'))

                    if event.key == pygame.K_q:
                        self.evManager.push(KeyEvent('Q', 'R'))

                    if event.key == pygame.K_e:
                        self.evManager.push(KeyEvent('E', 'R'))

                    if event.key == pygame.K_BACKSPACE:
                        self.evManager.push(KeyEvent('BACKSPACE', 'R'))

