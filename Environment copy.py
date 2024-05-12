import numpy as np
from State import State
from Graphics import *
from Constant import *
import torch

print()

class Environment_copy:
    def __init__(self, state=None) -> None:
        if state:
            self.state = state
        else:
            self.state = self.get_init_state()


    def get_legal_actions(self, state:State):
        if state.player==1:
            list_B= []
            pieces = np.where(state.board>>9&1==1)
            list_1=list(zip(pieces[0], pieces[1]))
            for i in range(len(list_1)):
                # print(list_1[i])
                list_B = list_B + self.get_Actions( list_1[i],state)
            return list_B  
        elif state.player==-1:
            list_R= []
            pieces = np.where(state.board>>8&1==1)
            list_1=list(zip(pieces[0], pieces[1]))
            for i in range(len(list_1)):
                list_R = list_R + self.get_Actions( list_1[i],state)
            return list_R     

        
    
    def get_Actions(self, row_col, state: State= None):
        
        #הוזזת שחקן מקום מסוים בתנאי שקיים החץ המסוים
        my_list = []
        my_row, my_col =row_col
        piece=state.board[row_col]
        
        directions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),]
         #למעלה
        action = (row_col, (my_row - 1, my_col),-1)
        if self.Is_legel_move(state, action ):
            my_list.append(action)

        # אלכסון למעלה ימין
        action=( row_col, (my_row - 1, my_col + 1),-1)
        if self.Is_legel_move(state,action):
            my_list.append(action) 

        # ימינה
        action =  (row_col, (my_row, my_col + 1),-1)
        if self.Is_legel_move(state,action ):
            my_list.append(action)       

        #  אלכסון למטה ימין
        action =  (row_col, (my_row + 1, my_col + 1),-1)
        if self.Is_legel_move(state, action):
            my_list.append(action)

        # למטה
        action= (row_col, (my_row + 1, my_col),4)
        if self.Is_legel_move(state,action ):
            my_list.append(action)    

         #  אלכסון למטה שמאל
        action = (row_col, (my_row + 1, my_col - 1),-1) 
        if  self.Is_legel_move(state, action):
            my_list.append(action)

        # שמאלה
        action = (row_col, (my_row, my_col - 1),-1)
        if self.Is_legel_move(state, action):
            my_list.append(action)

        # אלכסון למעלה שמאל
        action = (row_col, (my_row - 1, my_col - 1),-1)
        if self.Is_legel_move(state, action):
            my_list.append(action)

        #הוספת חץ במידה ואפשר 
        

        if piece>>0&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),0))
        
        if piece>>1&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),1))
        
        if piece>>2&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),2))
        
        if piece>>3&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),3))

        if piece>>4&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),4))

        if piece>>5&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),5))

        if piece>>6&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),6))

        if piece>>7&1 == 0:
            my_list.append(((my_row,my_col),(my_row, my_col),7))



        return my_list

    def is_legel_arrow(self, state, action):
        current, destination, arrow = action
        piece = state.board[current]
        bit = (piece>>arrow) & 1
        return bit == 0

    def Is_legel_move(self,state:State,action):
        current, destination, arrow= action
        piece = state.board[current]
        row, col = current
        des_row, des_col = destination
        if state.board[current] == 0 or state.player != state.getcolor(current) : # current is not a player
           return False
       
        if current == destination: # add arrow
            return self.is_legel_arrow(state, action)

        if des_row > 5 or des_row< 0 or des_col > 6 or des_col < 0: # destination in board
            return False
       
        if state.board[destination] != 0: # destination is empty
            return False      

        if  row-des_row ==1 and col-des_col==0 and piece>>0 & 1: #למעלה
            return True
        if  row-des_row ==1 and col-des_col==-1 and piece>>1 & 1: #למעלה ימין
            return True
        if  row-des_row ==0 and col-des_col==-1 and piece>>2 & 1:#ימינה
            return True
        if  row-des_row ==-1 and col-des_col==-1 and piece>>3 & 1: #למטה ימין
            return True
        if  row-des_row ==-1 and col-des_col==0 and piece>>4 & 1: #למטה
            return True
        if  row-des_row ==-1 and col-des_col==1 and piece>>5 & 1: #למטה שמאל
            return True
        if  row-des_row ==0 and col-des_col==1 and piece>>6 & 1: #שמאל
            return True
        if  row-des_row ==1 and col-des_col==1 and piece>>7 & 1: #שמאל למעלה
            return True
        
        return False
    
    def end_of_game(self, state: State):
       col_1 = 1
       for i in range(1, 5):
           if state.getcolor((i,col_1))==-1 :
               return -1
           
       col_2 = 5
       for i in range(1, 5):
           if state.getcolor((i,col_2))==1:
               return 1
           
       return 0
    
    def Move(self, state: State, action):

        current, destination, arrow = action
        if current == destination:
           num= 2**arrow
           new_num = state.board[current]+ num
           state.board[current] = new_num
        else:
            state.board[destination] = state.board[current]
            state.board[current] = 0

        state.SwitchPlayer()

    def get_next_state(self, state: State, action):
        next_state = state.copy()
        self.Move(next_state ,action)
        return next_state
    
    def toTensor (self, list_states, device = torch.device('cpu')) -> tuple:
        list_board_tensors = []
        list_legal_actions = []
        for state in list_states:
            board_tensor, legal_actions = state.toTensor(device)
            list_board_tensors.append(board_tensor)
            list_legal_actions.append(torch.tensor(legal_actions))
        return torch.vstack(list_board_tensors), torch.vstack(list_legal_actions)
    
    def reward (self, state : State, action = None) -> tuple:
        if action:
            next_state = self.get_next_state(action, state)
        else:
            next_state = state
        if (self.end_of_game(next_state)):
            if self.end_of_game(next_state)==1:
                return 1,True
            elif self.end_of_game(next_state)==-1:
                return -1,True
            else:
                return 0,False
        else: 
            return 0,False
    
    def get_init_state(self, Rows_Cols = (ROWS, COLS)):
        rows, cols = Rows_Cols
        board = np.zeros([rows, cols],int)
        board[1][1] = 512
        board[2][1] = 512
        board[3][1] = 512
        board[4][1] = 512
        board[1][5] = 256
        board[2][5] = 256
        board[3][5] = 256
        board[4][5] = 256
        return State (board, player=1)

    @staticmethod
    def action_toTensor (action):
        return torch.tensor([action[0][0], action[0][1],action[1][0],action[1][1],action[2]])

    @staticmethod
    def action_fromTensor (tensor):
        return ((tensor[0],tensor[1]),(tensor[2],tensor[3]),tensor[4])