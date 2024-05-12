import numpy as np
import torch

class State:
    def __init__(self, board=None, player=1):
        if board is None:
            self.init_board()
        else:
            self.board = board # np.array
        self.rows, self.cols = self.board.shape
        self.player = player

    def init_board (self):
         self.board = np.array([[0, 0, 0, 0, 0, 0, 0], 
                            [0, 512, 0, 0, 0, 256, 0], 
                            [0, 512, 0, 0, 0, 256, 0], 
                            [0, 512, 0, 0, 0, 256, 0], 
                            [0, 512, 0, 0, 0, 256, 0], 
                            [0, 0, 0, 0, 0, 0, 0]])
        
        # self.board = np.array([[0,0, 0, 0, 0, 0, 0], 
        #                     [0, 767, 0, 0, 0, 511, 0], 
        #                     [0, 767, 0, 0, 0, 511, 0], 
        #                     [0, 767, 0, 0, 0, 511, 0], 
        #                     [0, 767, 0, 0, 0, 511, 0], 
        #                     [0, 0, 0, 0, 0, 0, 0]])
        
    def __hash__(self) -> int:
        return hash(repr(self.board))

    def get_blank_pos (self):
        pos = np.where(self.board == 0)
        row = pos[0].item()
        col = pos[1].item()
        return row, col

    def SwitchPlayer (self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def __eq__(self, other):
        return np.equal(self.board, other.board).all()

    def copy (self):
        newBoard = np.copy(self.board)
        return State (newBoard, player=self.player)

    def getcols(self):
        return self.cols

    def getrows(self):
        return self.rows        
    
    def getcolor(self,row_col):

        piece = self.board[row_col[0],row_col[1]]
        if piece>=512:
            return 1
        if 256<= piece < 512 :
            return -1
        return 0
    
    def getplay(self,row_col):

        piece = self.board[row_col[0],row_col[1]]
        if piece>=512:
            return 1
        if 256<= piece < 512 :
            return 2
        return 0
    
    def reverse (self):
        reversed = self.copy()
        reversed.player = reversed.player * -1
        for row in range(6):
            for col in range(7):
                if reversed.board[row,col] >>9&1==1:
                    reversed.board[row,col]= reversed.board[row,col] -256
                elif reversed.board[row,col] >>8&1==1:
                    reversed.board[row,col]= reversed.board[row,col] +256
        reversed.board = np.fliplr(reversed.board)
        return reversed

    def toTensor (self, device = torch.device('cpu'),legal_actions=[]) -> tuple:
        return self.toBitTensor(device=device)
        # board_np = self.board.reshape(-1)
        # board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)/1000
        # return board_tensor
    
    def toBitTensor (self, device = torch.device('cpu'),legal_actions=[]):
        # board_tensor = torch.tensor(self.board)
        blue_board = torch.tensor(self.board)
        red_board = torch.tensor(self.board)
        blue_board[blue_board < 512] = 0
        red_board[red_board > 511] = 0
        bitTensors= []
        for i in range(8):
            temp_board = blue_board & 1
            temp_board = temp_board + (red_board & 1) * -1
            bitTensors.append(temp_board)
            blue_board = blue_board >> 1
            red_board = red_board >> 1
        
        blue_board = blue_board >> 1
        red_board = red_board * -1
        bitTensors.append(blue_board + red_board)
        bitTensor = torch.stack(bitTensors).float()
        return bitTensor.reshape(-1)

    [staticmethod]
    def tensorToState (state_tensor, player):
        return State.bitTensorToState(bitTensor=state_tensor, player=player)
        # board = state_tensor.reshape([6,7]).cpu().numpy()*1000
        # board = board.astype(int)
        # return State(board, player=player)
    
    [staticmethod]
    def bitTensorToState (bitTensor, player=1):
        blue_board = bitTensor.reshape([9, 6, 7]).cpu().numpy().astype(int)
        red_board = bitTensor.reshape([9, 6, 7]).cpu().numpy().astype(int)
        blue_board[blue_board<0] = 0
        blue_board[8] = blue_board[8] << 1
        red_board[red_board>0]=0
        red_board = red_board * -1

        board = blue_board[8] + red_board[8]

        for i in range(7, -1, -1):
            board = board << 1
            board = board | red_board[i] | blue_board[i]
            
        
        board = board.astype(int)
        return State(board, player=player)

    def score (self):
        return self.board.sum()