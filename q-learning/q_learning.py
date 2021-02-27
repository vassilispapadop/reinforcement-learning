import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

episodes = 150
runs = 20
actions = [0, 1] 
learning_rate = 0.1
discount_factor = 0.9
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
        self.q_action_per_episode = {}
        self.initial_state = (0,0)
        self.current_state = self.initial_state
        self.q_table = {self.initial_state:[0, 0]}


    def next_action(self, state):
        #exploration
        action = None
        if np.random.random() < self.e_greedy:
            action = np.random.choice(actions)
        else:
            action = np.argmax(self.q_table[self.current_state])
        return action

    def updateQ(self, time_step, action, reward):
        
        # New state discovered
        if (action, reward) not in self.q_table:
            self.q_table[(action, reward)] = [0, 0]

        old_value = self.q_table[self.current_state][action]
        next_max = np.max(self.q_table[self.current_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        self.q_table[self.current_state][action] = new_value
        self.current_state = (action,reward)


    def total_reward(self, r):
        self.episode_reward += r
        self.cumulative_reward += r
        self.total_reward_evolution.append(self.cumulative_reward)

    def update_stats(self):
        for key, value in self.q_table.items():
            if key == self.initial_state:
                continue
            if key not in self.q_action_per_episode:
                self.q_action_per_episode[key] = {0:[], 1:[]}

            self.q_action_per_episode[key][0].append(value[0])
            self.q_action_per_episode[key][1].append(value[1])


    def reset(self):
        ''' resets rewards for next episode and epsilon decrement '''
        self.current_state = self.initial_state
        self.episode_reward = 0
        self.e_greedy -= 0.01 
        # make sure we do not use negative values
        if self.e_greedy < 0.0: 
            self.e_greedy = 0.0

        print("epsilon agent {} is:{}".format(self.name, self.e_greedy))

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
            # if episode < 100:
            player1.updateQ(run, action1, r1)
            player2.updateQ(run, action2, r2)

            player1.total_reward(r1)
            player2.total_reward(r2)

        player1.avg_reward_episode.append(player1.episode_reward / runs)
        player2.avg_reward_episode.append(player2.episode_reward / runs)

        player1.update_stats()
        player2.update_stats()


        print("Episode:{} -> R1 {}, R2 {}".format(episode, player1.episode_reward, player2.episode_reward))

    print(player1.q_table)

    #avg reward per episode
    f = plt.figure(figsize=(20,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.plot(range(1,episodes+1), player1.avg_reward_episode, label=player1.name, color='blue')
    ax1.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    ax1.title.set_text("Mean reward/episode") 
    ax1.set_ylabel('mean reward per 20 runs')
    ax1.set_xlabel('episodes') 
    ax1.legend(loc="upper left")

    ax2.plot(range(1,episodes+1), player2.avg_reward_episode, label=player2.name, color='orange')
    ax2.axvline(x=100, label='egreedy = 0', color='black', linestyle="--")
    ax2.title.set_text("Mean reward/episode") 
    ax2.set_ylabel('mean reward per 20 runs')
    ax2.set_xlabel('episodes') 
    ax2.legend(loc="upper left")
    plt.savefig("q-learning/avg_reward_episode.png", format="png")



    # total reward evolution
    f = plt.figure(figsize=(20,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.plot(range(1,(episodes * runs) + 1), player1.total_reward_evolution, label=player1.name, color='blue')
    ax1.axvline(x=2000, label='egreedy = 0', color='black', linestyle="--")
    ax1.title.set_text("Total reward as function of time, Agent 0") 
    ax1.set_ylabel('total reward')
    ax1.set_xlabel('time(episodes x runs)')
    ax1.legend(loc="upper left")

    ax2.plot(range(1,(episodes * runs) + 1), player2.total_reward_evolution, label=player2.name, color='orange')
    ax2.axvline(x=2000, label='egreedy = 0', color='black', linestyle="--")
    ax2.title.set_text("Total reward as function of time, Agent 1") 
    ax2.set_ylabel('total reward')
    ax2.set_xlabel('time(episodes x runs)')
    ax2.legend(loc="upper left")
    plt.savefig("q-learning/total_reward_time.png", format="png")
    

    # action per episode
    f = plt.figure(figsize=(20,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    for key in player1.q_table.keys():
        if key == (0,0): 
            continue
        
        ax1.plot(player1.q_action_per_episode[key][0], label="st:{}, ac:A".format(key))
        ax1.plot(player1.q_action_per_episode[key][1], label="st:{}, ac:D".format(key))

    ax1.axvline(x=100, color='black', linestyle="--")
    ax1.title.set_text("Agent 0 Q(s,a) per episode") 
    ax1.set_ylabel('Q-values')
    ax1.set_xlabel('episodes')
    ax1.legend(loc="upper left")

    for key in player2.q_table.keys():
        if key == (0,0): 
            continue
    
        ax2.plot(player2.q_action_per_episode[key][0], label="st:{}, ac:A".format(key))
        ax2.plot(player2.q_action_per_episode[key][1], label="st:{}, ac:D".format(key))

    ax2.axvline(x=100, color='black', linestyle="--")
    ax2.title.set_text("Agent 1 Q(s,a) per episode") 
    ax2.set_ylabel('Q-values')
    ax2.set_xlabel('episodes')
    ax2.legend(loc="upper left")
    plt.savefig("q-learning/q_action_values_per_episode.png", format="png")


