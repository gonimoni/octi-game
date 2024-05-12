from State import State
from Environment import Environment
from Constant import *
from Graphics import *
MAXSCORE = 1000

class AlphaBetaAgent:

    def __init__(self, player, depth = 2, environment: Environment = None):
        self.player = player
        if self.player == 1:
            self.opponent = 2
        else:
            self.opponent = 1
        self.depth = depth
        self.environment : Environment = environment


    def evaluate (self, gameState : State):
        # player_score, opponent_score = gameState.score(player = self.player)
        # score =  player_score - opponent_score 
        int_p = 0
        if self.player==1:
            int_p=2
        elif self.player==2:
            int_p = 1     
        score = 0
        col_op = 0
        if self.player == 1:
            col_op = 5
        else:
            col_op = 1

        for row in range(0,6):
            for col in range(0,7):
                    for i in range(1,5):
                        if row == i and col -col_op <0 :
                            if gameState.board[row][col]>>2&1==1:
                                if gameState.getplay((row,col))==self.player:
                                    score+=5
                                elif gameState.getplay((row,col))==int_p:
                                    score-=5
                            if gameState.board[row][col]>>1&1==1 or gameState.board[row][col]>>3&1==1:
                                if gameState.getplay((row,col))==self.player:
                                    score+=3
                                elif gameState.getplay((row,col))==int_p:
                                    score-=3
                        if row == i and col -col_op >0 :
                            if gameState.board[row][col]>>6&1==1:
                                if gameState.getplay((row,col))==self.player:
                                    score+=5
                                elif gameState.getplay((row,col))==int_p:
                                    score-=5
                            if gameState.board[row][col]>>5&1==1 or gameState.board[row][col]>>7&1==1:
                                if gameState.getplay((row,col))==self.player:
                                    score+=3
                                elif gameState.getplay((row,col))==int_p:
                                    score-=3
                        

        # col_B = 1
        # col_R = 5
        # for i in range(1,5):
                # if gameState.board[i][col_B]>512:
                    # if gameState.getplay((i,col_B))==self.player:
                    #    score+=6
                    # elif gameState.getplay((i,col_B))==int_p:
                    #    score-=6   
                # if 256<gameState.board[i][col_R]<512:
                    # if gameState.getplay((i,col_R))==self.player:
                    #    score+=6
                    # elif gameState.getplay((i,col_R))==int_p:
                    #    score-=6    
        list_B = [(1,1),(2,1),(3,1),(4,1)]
        list_R = [(1,5),(2,5),(3,5),(4,5)]
        Abs=0
        for row in range(0,6):
            for col in range(0,7):
                    if gameState.getcolor((row,col))==-1:
                        if gameState.getplay((row,col))==self.player:
                            for pos in list_B:
                                Abs = abs(row-pos[0])+abs(col-pos[1])
                                score +=(40-(Abs*10))
                                if Abs == 0:
                                    score +=  100
                        if gameState.getplay((row,col))==int_p:
                            for pos in list_B:
                                Abs = abs(row-pos[0])+abs(col-pos[1])
                                score -=(40-(Abs*10))
                                if Abs == 0:
                                    score -=  100 
                    if gameState.getcolor((row,col))==1:
                        if gameState.getplay((row,col))==self.player:
                            for pos in list_R:
                                Abs = abs(row-pos[0])+abs(col-pos[1])
                                score +=(40-(Abs*10))
                                if Abs == 0:
                                    score +=  100
                        if gameState.getplay((row,col))==self.player:
                            for pos in list_R:
                                Abs = abs(row-pos[0])+abs(col-pos[1])
                                score -=(40-(Abs*10))
                                if Abs == 0:
                                    score -=  100          

        return score

    def get_Action(self, events= None, state : State= None , graphics : Graphics = None, train = None):
        visited = set()
        value, bestAction = self.minMax(state, visited)
        return bestAction


    def minMax(self, state:State, visited:set):
        depth = 0
        alpha = -MAXSCORE
        beta = MAXSCORE
        return self.max_value(state, visited, depth, alpha, beta)


    def max_value (self, state:State, visited:set, depth, alpha, beta):
        
        value = -MAXSCORE

        # stop state
        if depth == self.depth or self.environment.end_of_game(state):
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(state, action)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.min_value(newState, visited,  depth + 1, alpha, beta)
                if newValue > value:
                    value = newValue
                    bestAction = action
                    alpha = max(alpha, value)
                if value >= beta:
                    return value, bestAction
                    

        return value, bestAction 

    def min_value (self, state:State, visited:set, depth, alpha, beta):
        
        value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.end_of_game(state):
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(state,action)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.max_value(newState, visited,  depth + 1, alpha, beta)
                if newValue < value:
                    value = newValue
                    bestAction = action
                    beta = min(beta, value)
                if value <= alpha:
                    return value, bestAction

        return value, bestAction    