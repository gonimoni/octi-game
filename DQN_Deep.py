import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 66 # state: board = 8 * 8 = 64 + action (2) 
layer1 = 64
layer2 = 128
layer3 = 256
layer4 = 128
layer5 = 64
output_size = 1 # Q(state, action)
gamma = 0.99 


class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, layer3)
        self.linear4 = nn.Linear(layer3, layer4)
        self.linear5 = nn.Linear(layer4, layer5)
        self.output = nn.Linear(layer5, output_size)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        x = F.leaky_relu(self.linear5(x))
        x = self.output(x)
        return x
    
    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return self.MSELoss(Q_value, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)