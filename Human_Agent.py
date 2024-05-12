import pygame
from Graphics import *
from Environment import Environment

class Human_Agent:

    def __init__(self, player: int, env:Environment, graphics: Graphics) -> None:
        self.player = player
        self.env = env
        self.graphics = graphics
        self.mode = 0
        self.current = None

    def get_Action (self, events= None, state : State= None , graphics : Graphics = None, train = None):
        for event in events:
            if self.mode == 0:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row_col = self.graphics.calc_row_col(pos) 
                    if state.getcolor(row_col) == state.player:
                        self.mode = 1
                        self.current = row_col
                        print('chose:', row_col)
                        return None
                    else:
                        print('ilegal - choose player again')
            elif self.mode == 1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row_col = self.graphics.calc_row_col(pos) 
                    if row_col == self.current:
                        self.mode = 2
                        print('choose arrow-(number between 0-7)')
                        return None
                    else:
                        action = self.current, row_col, -1
                        if self.env.Is_legel_move(state, action):
                            self.mode = 0
                            return action
                        else:
                            print('illegal move')
                            return None
            else:
                if event.type == pygame.KEYUP:
                    if 0<=int(pygame.key.name(event.key))<=7:
                        arrow = int(pygame.key.name(event.key))
                        action = self.current,self.current,arrow                  
                        if self.env.is_legel_arrow(state, action):
                            self.mode= 0
                            return action
                    else:
                        print('illegal arrow - choose again')
        else:
            return None