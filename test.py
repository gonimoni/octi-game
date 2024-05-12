# pipenv install torch --index https://download.pytorch.org/whl/cu121  
import numpy as np
import torch
from State import State

# num = int ('1010001001',2)
# print(num)
# print(bin(num))

# blue = int ('1000000000',2)
# res = num & blue
# print(bin(res))

# print(bin(num>>4))
# print ((num>>8)&1)

# from State import *
# from Environment import *

# state= State()
# env = Environment(state)
# my_list=[]

# my_list=env.get_Actions((1,1),None,None,state)
# print(len(my_list))

board = np.array([[0,0, 0, 0, 0, 0, 0], 
                [0, 767, 0, 0, 0, 511, 0], 
                [0, 767, 0, 0, 0, 511, 0], 
                [0, 767, 0, 0, 0, 511, 0], 
                [0, 767, 0, 0, 0, 511, 0], 
                [0, 0, 0, 0, 0, 0, 0]])
# boardflip = np.fliplr(board)
# pieces = np.where(board>>8&1==1)
# print(pieces)
# print(list(zip(pieces[0], pieces[1])))

# from State import *
# current = 1,1
# state=State()
# print(state.board[current])

# from Environment import *

# env = Environment(state)
# print(len(env.get_legal_actions(state)))

# for i in range(0,10):
#     print(i)




# tensor = torch.tensor(board)

# print(tensor)

# print(tensor.fliplr())
# tenso1 = tensor & 1
# tensor = tensor>>1
# tensor2 = tensor & 1

# tensor3 = torch.stack([tenso1, tensor2])
# print (tensor3.shape)
# print (tensor3)

state = State(board=board)
print(state.board)
bitTensor = state.toBitTensor()
print(bitTensor.shape)
print(bitTensor)
print(bitTensor.reshape(-1,6,7))

newState = State.bitTensorToState(bitTensor=bitTensor)
print(newState.board.shape)
print(newState.board)