import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

episodes = 200
runs = 20
actions = 2
alpha = 0.5
gamma = 0.9
beta = 1
delta = -2
zeta = -1



class Agent():
    def __init__(self, name, e_greedy):
        self.name = name
        self.e_greedy = e_greedy
        self.reward = 0
        self.stats = []
        self.q_table = np.zeros([runs, actions])

    def choose_action(self, state):
        action = None
        if np.random.random() < self.e_greedy:
            action = round(np.random.random())
        else:
            action = np.argmax(self.q_table[state])
            
        return action

    def updateQ(self, state, action, reward):
        next_state = state + 1
        if next_state == runs:
            print("Round ended")
            return None
        
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_table[state, action] = new_value

    def total_reward(self, r):
        self.reward += r

    def reset(self):
        self.reward = 0
        self.e_greedy -= 0.01

if __name__ == "__main__":
    #reward_grid
    reward_grid=pd.DataFrame({
                        0:[(alpha, beta),(delta, zeta)], 
                        1:[(zeta, delta),(beta, alpha)]
                    }) 
    reward_grid.index = [0,1]
    print(reward_grid)

    player1 = Agent('p1', 1)
    player2 = Agent('p2', 1)
    cumulative_average_0 = []
    cumulative_average_1 = []
    for episode in range(0, episodes):
        player1.reset()
        player2.reset()
        for run in range(0, runs):
            action1 = player1.choose_action(run)
            action2 = player2.choose_action(run)
            [r1, r2] = reward_grid[action1][action2]
            player1.updateQ(run, action1, r1)
            player2.updateQ(run, action2, r2)
            player1.total_reward(r1)
            player2.total_reward(r2)
        # print("col player chose {}, row player chose {} -> reward {}".format(action1, action2, reward_grid[action1][action2]))
        cumulative_average_0.append(np.cumsum(player1.q_table[:,0]) / (np.arange(runs) + 1))
        cumulative_average_1.append(np.cumsum(player1.q_table[:,1]) / (np.arange(runs) + 1))

        player1.stats.append(player1.reward)
        player2.stats.append(player2.reward)
        print("Episode:{} -> P1 reward {}, P2 reward {}".format(episode, player1.reward, player2.reward))

    # plt.plot(range(0,episodes), player1.stats, label='player1')
    # plt.plot(range(0,episodes), player2.stats, label='player2')
    # plt.plot(player1.q_table[:,0])
    # plt.plot(player2.q_table[:,0])

    # avg = np.zeros(20)
    # for a in cumulative_average_0:
    #     avg += a

    plt.plot(range(0,episodes), cumulative_average_1)
    plt.xscale('log') 
    # plt.plot(range(0,episodes), cumulative_average_1)
    print(player1.q_table)
    print(player1.q_table[0][0])
    plt.show()



