import numpy as np
import pygame
import time
from State import State
from Constant import *



white = ((255,255,255))


class Graphics:
    def __init__(self):
        pygame.init()
        win = pygame.display.set_mode((865, 650))
        pygame.display.set_caption('octi-gonen konki')
        self.win = win
        
    def draw_octagon(self):
        print(self.calc_base_pos())

    def draw_all_pieces(self, state: State):
        board = state.board
        row, col = board.shape
        for row in range(ROWS):
            for col in range(COLS):
                self.draw_board((row, col))
                self.draw_troop(state, (row, col))

    def draw_troop(self, state: State, row_col):
        # row, col = row_col # tuple for pos
        # board = state.board # ?
        # number = board[row][col] #
        TextFont = pygame.font.SysFont("David", 40)
        player = 1
        color = BLUE
        if state.player ==1:
            player=1
            color = GREENOCT
        else:
            player =2
            color = REDOCT
        Textplayer = TextFont.render("Current Player:",1 ,(8,8,8))
        
        pygame.draw.rect(self.win, color, (260,615,30,30) )
        self.win.blit(Textplayer, (15,610))

        pos = self.calc_base_pos(row_col) # 
        if (state.getcolor(row_col) == 1):
            pygame.draw.lines(self.win, GREENOCT, False, [[pos[0] + 30, pos[1] + 10], [pos[0] + 60, pos[1] + 10], [pos[0] + 80, pos[1] + 30], [pos[0] + 80, pos[1] + 60], 
            [pos[0] + 60, pos[1] + 80], [pos[0] + 30, pos[1] + 80], [pos[0] + 10, pos[1] + 60], [pos[0] + 10, pos[1] + 30], [pos[0] + 30, pos[1] + 10]], 3)
        if (state.getcolor(row_col) == -1):
            pygame.draw.lines(self.win, REDOCT, False, [[pos[0] + 30, pos[1] + 10], [pos[0] + 60, pos[1] + 10], [pos[0] + 80, pos[1] + 30], [pos[0] + 80, pos[1] + 60], 
            [pos[0] + 60, pos[1] + 80], [pos[0] + 30, pos[1] + 80], [pos[0] + 10, pos[1] + 60], [pos[0] + 10, pos[1] + 30], [pos[0] + 30, pos[1] + 10]], 3)
        piece = state.board[row_col]
        for i in range(8):
            if (piece>>i)&1 == 1:
                if i %2== 0:
                    self.draw_arrow_even(state,row_col,i)
                elif i %2 == 1:
                    self.draw_arrow_odd(state,row_col,i)
                
    def draw_arrow_even (self, state:State, row_col,i):
        arrow = pygame.image.load('Img/Arrow3.png')
        arrow = pygame.transform.scale(arrow, (SQUARE_SIZE, SQUARE_SIZE))
        
        pos = self.calc_base_pos(row_col)
        pos = ((pos[0] - 5), pos[1]-3)
        arrow = pygame.transform.rotate(arrow,i*-45)
        self.win.blit(arrow, pos)

    def draw_arrow_odd (self, state:State, row_col,i):
        arrow = pygame.image.load('Img/Arrow3.png')
        arrow = pygame.transform.scale(arrow, (SQUARE_SIZE, SQUARE_SIZE))
        
        pos = self.calc_base_pos(row_col)
        pos = ((pos[0] - 25), pos[1]-23)
        arrow = pygame.transform.rotate(arrow,i*-45)
        self.win.blit(arrow, pos)

    def draw_board(self, row_col):
        board_color = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0]])
        row, col = row_col # tuple for pos
        number = board_color[row][col] # 0 empty 1 us 2 enemy
        pos = self.calc_base_pos(row_col) # 
        color = self.calc_color(number)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE-PADDING, SQUARE_SIZE-PADDING))
        
        
    def calc_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + SQUARE_SIZE//2 + FRAME
        x = col * SQUARE_SIZE + SQUARE_SIZE//2 + FRAME
        return x, y

    def calc_base_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + FRAME
        x = col * SQUARE_SIZE + FRAME
        return x + 75, y

    def calc_num_pos(self, row_col, font, number):
        row, col = row_col
        font_width, font_height = font.size(str(number))
        y = row * SQUARE_SIZE + FRAME + (SQUARE_SIZE - font_height)//2
        x = col * SQUARE_SIZE + FRAME + (SQUARE_SIZE - font_width)//2
        return x, y

    def calc_row_col(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE - 1
        row = y // SQUARE_SIZE
        return row, col

    def calc_color(self, number):
            if number == 0:
                return WHITE
            elif number == 1:
                return GREEN
            else:
                return RED

    def draw(self, state):
        self.win.fill(LIGHTGRAY)
        self.draw_all_pieces(state)

    def draw_square(self, row_col, color):
        pos = self.calc_base_pos(row_col)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE, SQUARE_SIZE))

    def blink(self, row_col, color):
        row, col = row_col
        player = self.board[row][col]
        for i in range (3):
            self.draw_square((row, col), color)
            pygame.display.update()
            time.sleep(0.2)
            self.draw_piece((row, col))
            pygame.display.update()
            time.sleep(0.2)
    


   

        
    