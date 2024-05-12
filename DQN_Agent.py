import torch
import random
import math
from DQN import DQN
from Constant import *
from State import State
from Environment import*

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.setTrainMode()
        self.env = env

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state:State, epoch = 0, events= None, train = True, graphics = None, black_state = None ) -> tuple:
        actions = self.env.get_legal_actions(state=state)
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        # if self.player == 1:
        state_tensor = state.toTensor()
        actions_tensor = self.actions_to_tensors(actions=actions)
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(actions_tensor),1))
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, actions_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        
        for i, state_tensor in enumerate(states_tensor):
            if dones[i].item():
                actions.append(((0,0), -1, -1))
            else:
                state = State.tensorToState(state_tensor= state_tensor, player=self.player)
                action = self.get_Action(state, train=False)
                actions.append(action)
        return self.actions_to_tensors(actions)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def actions_to_tensors (self, actions):
        flattend_actions = []
        for action in actions:
            flattend_actions.append((action[0][0], action[0][1], action[1], action[2]))
        
        action_np = np.array(flattend_actions, dtype=np.float32)
        action_tensor = torch.from_numpy(action_np)
        return action_tensor

    def __call__(self, events= None, state=None):
        return self.get_Action(state)