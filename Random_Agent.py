import Game
import numpy as np
from Environment import Environment
from State import State
from Graphics import *
import random

class Random_Agent:
    def __init__(self,player:int= None , env :  Environment= None) -> None:
          self.player = player
          self.env =  env

    def get_Action (self, events= None, state : State= None , graphics : Graphics = None, train = None):
         actions = self.env.get_legal_actions(state)
         action = random.choice(actions)
         return action