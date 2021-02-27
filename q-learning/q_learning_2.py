import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

episodes = 150
runs = 20
actions = [0, 1] 
learning_rate = 0.5
gamma = 0.9
epsilon = 1


#reward_grid
reward_grid=pd.DataFrame({
                    0:[(1, 1),(-1, -1)], 
                    1:[(-1, -1),(1, 1)]
                }) 
reward_grid.index = [0,1]
print(reward_grid)


class Agent():
    def __init__(self, name, e_greedy):
        self.name = name
        self.e_greedy = e_greedy
        self.episode_reward = 0
        self.cumulative_reward = 0
        self.avg_reward_episode = []
        self.total_reward_evolution = []
        self.q_action_per_episode = {0:[],1:[]}
        self.state = 0
        self.q_table = np.zeros([runs, len(actions)])

    def next_action(self, state):
        #exploration
        action = None
        if np.random.random() < self.e_greedy:
            action = np.random.choice(actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def updateQ(self, state, action, reward):
        next_state = state + 1
        if next_state >= runs - 1:
            next_state = runs - 1
        
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)
        self.q_table[state, action] = new_value


        
        # old_value = self.q_table[self.state, action]
        # next_max = np.max(self.q_table[self.state])
        
        # new_value = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)
        # self.q_table[self.state, action] = new_value

    def total_reward(self, r):
        self.episode_reward += r
        self.cumulative_reward += r
        self.total_reward_evolution.append(self.cumulative_reward)

    def update_stats(self):
        self.q_action_per_episode[0].append(self.q_table[0][0])
        self.q_action_per_episode[1].append(self.q_table[0][1])

    def reset(self):
        ''' resets rewards for next episode and epsilon decrement '''
        self.episode_reward = 0
        self.e_greedy -= 0.01 
        # make sure we do not use negative values
        if self.e_greedy < 0.0: 
            self.e_greedy = 0.0

if __name__ == "__main__":

    player1 = Agent('Agent1', epsilon)
    player2 = Agent('Agent2', epsilon)
    cumulative_average_0 = []
    cumulative_average_1 = []
    for episode in range(0, episodes):
        player1.reset()
        player2.reset()
        for run in range(0, runs):
            action1 = player1.next_action(run)
            action2 = player2.next_action(run)
            [r1, r2] = reward_grid[action1][action2]
            #stop update q-table after 100th episode, testing phase
            if episode < 100:
                player1.updateQ(run, action1, r1)
                player2.updateQ(run, action2, r2)

            player1.total_reward(r1)
            player2.total_reward(r2)

        # print("col player chose {}, row player chose {} -> reward {}".format(action1, action2, reward_grid[action1][action2]))
        cumulative_average_0.append(np.cumsum(player1.q_table[:,0]) / (np.arange(runs) + 1))
        cumulative_average_1.append(np.cumsum(player1.q_table[:,1]) / (np.arange(runs) + 1))

        player1.avg_reward_episode.append(player1.episode_reward / runs)
        player2.avg_reward_episode.append(player2.episode_reward / runs)

        player1.update_stats()
        player2.update_stats()

        print("Episode:{} -> R1 {}, R2 {}".format(episode, player1.episode_reward, player2.episode_reward))

    #avg reward per episode
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(range(1,episodes+1), player1.avg_reward_episode, label=player1.name, color='blue')
    # plt.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    # plt.title("Mean reward/episode") 
    # plt.ylabel('mean reward per 20 runs')
    # plt.legend(loc="upper left")
    # ax2 = fig.add_subplot(212)
    # ax2.plot(range(1,episodes+1), player2.avg_reward_episode, label=player2.name, color='orange')
    # plt.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    # plt.xlabel('episodes')  
    # plt.ylabel('mean reward per 20 runs') 
    # plt.legend(loc="upper left")
    # plt.savefig("q-learning/avg_reward_episode.png", format="png")


    # total reward evolution
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(range(1,(episodes * runs) + 1), player1.total_reward_evolution, label=player1.name, color='blue')
    # plt.axvline(x=2000, label='egreedy = 0', color='black', linestyle="--")
    # plt.title("Total reward as function of time") 
    # plt.ylabel('total reward')
    # plt.legend(loc="upper left")
    # ax2 = fig.add_subplot(212)
    # ax2.plot(range(1,(episodes * runs) + 1), player2.total_reward_evolution, label=player2.name, color='orange')
    # plt.axvline(x=2000, label='egreedy = 0', color='black', linestyle="--")
    # plt.xlabel('time(episodes x runs)')  
    # plt.ylabel('total reward') 
    # plt.legend(loc="upper left")
    # plt.savefig("q-learning/total_reward_time.png", format="png")

    # action per episode
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(range(1,episodes + 1), player1.q_action_per_episode[0], label='p1, a0', color='blue')
    ax1.plot(range(1,episodes + 1), player1.q_action_per_episode[1], label='p1, a1', color='orange')
    plt.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    plt.title("Q-action values per episode") 
    plt.ylabel('Q-value')
    plt.legend(loc="upper left")
    ax2 = fig.add_subplot(212)
    ax2.plot(range(1,episodes + 1), player2.q_action_per_episode[0], label='p2, a0', color='blue')
    ax2.plot(range(1,episodes + 1), player2.q_action_per_episode[1], label='p2, a1', color='orange')
    plt.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    plt.xlabel('episodes')  
    plt.ylabel('Q-value') 
    plt.legend(loc="upper left")
    plt.savefig("q-learning/q_action_values_per_episode.png", format="png")


    #plt.plot(cumulative_average_0)
    # plt.xscale('log') 
    # plt.plot(cumulative_average_1)
    # print(player1.q_table)
    # print(player1.q_table[0][0])
    # plt.plot(action_0_per_epi)
    # plt.plot(action_1_per_epi)
    # plt.show()



