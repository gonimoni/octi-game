import pygame
import numpy as np

ROWS, COLS = 6, 7
HEIGHT, WIDTH  = 1000, 700
FRAME = 10
SQUARE_SIZE = 100
LINE_WIDTH = 2
PADDING = 2


#RGB
RED = (255,48,48)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (151,255,255)
GREENOCT = (0,104,139)
REDOCT = (139,26,26)
BOARD_COLOR = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0]])
FPS = 60

# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.01
epsiln_decay = 100