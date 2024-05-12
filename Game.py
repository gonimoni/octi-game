import numpy as np
import pygame
from Graphics import Graphics
from Constant import *
from State import State
from Environment import *
from Human_Agent import Human_Agent
from Random_Agent import *
from MinMax_Agent import *
from Alpha_BetaAgent import *
from DQN_Agent import *

def main ():
    
    graphics = Graphics()
    state = State()
    env = Environment(state)

    run = True
    clock = pygame.time.Clock()
    pygame.display.update()

    
    
    player1 = Human_Agent(player=1, env=env, graphics=graphics)
    # player1 = Random_Agent(player=1, env= env)
    # player1 = DQN_Agent(player=1,env=env,train=False, parametes_path="Data/params_15.pth")
    # player1 = MinMaxAgent(player=1,depth=2,environment=env)
    # player1 = AlphaBetaAgent(player=1,depth=3,environment=env)
    # player2 = Human_Agent(player=2, env=env, graphics=graphics)
    player2 = Random_Agent(player=2, env= env)
    # player2 = MinMaxAgent(player=2,depth=2,environment=env)
    # player2 = AlphaBetaAgent(player=2,depth=2,environment=env)
    # player2 = DQN_Agent(player=-1,env=env,train=False, parametes_path="Data/params_12.pth")
    

    player = player1

    while(run):

        clock.tick(FPS)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
               exit(0)
        
        action = player.get_Action(events=events,state= env.state,graphics=graphics)
        if action:
            env.Move(state=env.state, action=action)
            if player == player1:
                player = player2
            else:
                player = player1

        if env.end_of_game(env.state)!=0:
            if env.end_of_game(env.state) == 1:
                print('player1 win!')
            else:
                print('player 2 win!')
            pygame.time.delay(200)
            break
        
        

        graphics.draw(state)
        pygame.display.update()
        
        
    pygame.quit()           

if __name__ == '__main__':
    main() 