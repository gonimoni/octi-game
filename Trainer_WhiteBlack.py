from Environment import Environment

from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
import torch
from Tester import Tester
from State import State
from MinMax_Agent import MinMaxAgent

epochs = 2000000
start_epoch = 0
C = 100
learning_rate = 0.0001
batch_size = 64
state = State()
env = Environment(state)
MIN_Buffer = 4000

File_Num = 15
path_load= None
path_Save=f'Data/params_{File_Num}.pth'
buffer1_path = f'Data/buffer1_{File_Num}.pth'
buffer2_path = f'Data/buffer2_{File_Num}.pth'
results_path=f'Data/results_{File_Num}.pth'
random_results_path1 = f'Data/random_results1_{File_Num}.pth'
random_results_path2 = f'Data/random_results2_{File_Num}.pth'


# add pathes for both players


def main ():
    
    player1 = DQN_Agent(player=1, env=env,parametes_path=None)
    player1_hat = DQN_Agent(player=1, env=env, train=False)
    Q1 = player1.DQN
    Q1_hat = Q1.copy()
    player1_hat.DQN = Q1_hat

    player2 = DQN_Agent(player=-1, env=env,parametes_path=None)
    player2_hat = DQN_Agent(player=-1, env=env, train=False)
    Q2 = player2.DQN
    Q2_hat = Q2.copy()
    player2_hat.DQN = Q2_hat

    
    buffer1 = ReplayBuffer(path=None) 
    buffer2 = ReplayBuffer(path=None) 
    
    results_file = [] #torch.load(results_path)
    results = [] #results_file['results'] # []
    avgLosses1 = []
    avgLosses2 = [] #results_file['avglosses']     #[]
    avgLoss1 = 0 
    avgLoss2 = 0
    loss = 0
    res = 0
    best_res = -200
    loss_count = 0
    tester1 = Tester(player1=player1, player2=Random_Agent(player=-1, env=env), env=env)
    tester2 = Tester(player1=Random_Agent(player=1, env=env), player2=player2, env=env)
    
    random_results_1, random_results_2 = [], [] #torch.load(random_results_path)   # []
    best_random = 0 #max(random_results)
    
    
    # init optimizer
    optim1 = torch.optim.Adam(Q1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optim1,100000, gamma=0.90)
  
    optim2 = torch.optim.Adam(Q2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optim2,100000, gamma=0.90)    
    

    for epoch in range(start_epoch, epochs):
        # print(f'epoch = {epoch}', end='\r')
        state_1 = env.get_init_state()
        step = 0
        action_2 = None
        while not env.end_of_game(state_1):
            # Sample Environement
            step += 1
            action_1 = player1.get_Action(state=state_1,epoch=epoch) 
            after_state_1 = env.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = env.reward(after_state_1)
            if action_2 is not None:
                buffer2.push(state_2, action_2,- reward_1, after_state_1, end_of_game_1)
            if end_of_game_1:
                res += reward_1
                buffer1.push(state_1, action_1, reward_1, after_state_1, True)
                break
            state_2 = after_state_1
            action_2 = player2.get_Action(state=state_2, epoch=epoch)
            after_state_2 = env.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = env.reward(state=after_state_2)
            if end_of_game_2:
                res += reward_2
                buffer2.push(state_2, action_2, - reward_2, after_state_2, True)
            buffer1.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer1) < MIN_Buffer:
                continue
            
            # Train White NN
            states, actions, rewards, next_states, dones = buffer1.sample(batch_size)
            Q_values = Q1(states, actions)
            next_actions = player1_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q1_hat(next_states, next_actions) 

            loss = Q1.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            scheduler1.step()
            if loss_count <= 1000:
                avgLoss1 = (avgLoss1 * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss1 += (loss.item()-avgLoss1)* 0.0001 
            
            # Train Black NN
            states, actions, rewards, next_states, dones = buffer2.sample(batch_size)
            Q_values = Q2(states, actions)
            next_actions = player2_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q2_hat(next_states, next_actions) 

            loss = Q2.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim2.step()
            optim2.zero_grad()
            scheduler2.step()
            if loss_count <= 1000:
                avgLoss2 = (avgLoss2 * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss2 += (loss.item()-avgLoss2)* 0.00001 
            
        if epoch % C == 0:
            Q1_hat.load_state_dict(Q1.state_dict())
            Q2_hat.load_state_dict(Q2.state_dict())


        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses1.append(avgLoss1)
            avgLosses2.append(avgLoss2)
            results.append(res)
            if best_res < res:      
                best_res = res
            res = 0
        
        if (epoch+1) % 500 == 0:
            random_results_1.append(tester1(100))
            random_results_2.append(-tester2(100))
            print(f'tester1: {random_results_1[-1]} tester2: {random_results_2[-1]}')

        if (epoch+1) % 1000 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses1':avgLosses1 ,'avglosses2':avgLosses2 }, results_path)
            torch.save(buffer1, buffer1_path)
            player1.save_param(path_Save)
            torch.save(random_results_1, random_results_path1)
            torch.save(random_results_2, random_results_path2)
        
        print (f'epoch={epoch} step={step} loss={loss:.5f} avgloss1={avgLoss1:.5f} avgloss2={avgLoss2:.5f}', end=" ")
        print (f'learning rate={scheduler1.get_last_lr()[0]} path={path_Save} res= {res}')

    torch.save({'epoch': epoch, 'results': results, 'avglosses1':avgLosses1 ,'avglosses2':avgLosses2}, results_path)
    torch.save(buffer1, buffer1_path)
    torch.save(buffer2, buffer2_path)
    torch.save(random_results_1, random_results_path1)
    torch.save(random_results_2, random_results_path2)
if __name__ == '__main__':
    main()