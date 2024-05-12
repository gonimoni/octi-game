from Random_Agent import Random_Agent
from Environment import Environment
from DQN_Agent import *
from MinMax_Agent import *

class Tester:
    def __init__(self, player1 = DQN_Agent , player2 = Random_Agent, env= Environment) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player_win = 0
        games = 0
        while games < games_num:
            print(games,end="\r")
            action = player.get_Action(state=env.state, train=False)
            env.Move(action=action, state=env.state)
            player = self.switchPlayers(player)
            winner = env.end_of_game(env.state)
            if winner !=0:
                player_win += winner
                env.state = env.get_init_state()
                games += 1
                player = self.player1
        return player_win

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    env = Environment()
    File_Num = 15
    path = f"Data/params_{File_Num}.pth"
    # player2 = DQN_Agent(env=env, player=-1,parametes_path=path)
    # player1 = Random_Agent(env=env, player=1)
    player2 = Random_Agent(env=env, player=-1)
    player1 = MinMaxAgent(player=1,depth=2,environment=env)
    test = Tester(env=env,player1=player1, player2=player2)
    print(test.test(100))