import numpy as np
import pandas as pd
# rewards
alpha = 1
beta = 1
delta = -1
zeta = -1


class Agent():
    def __init__(self, name, e_greedy):
        self.name = name
        self.e_greedy = e_greedy

    def choose_action(self):
        action = None
        if np.random.rand(1) > 0.49:
            action = 'A'
        else:
            action = 'D'
            
        return action


if __name__ == "__main__":
    episodes = 10
    #grid
    grid=pd.DataFrame({
                        "A":[(alpha, beta),(delta, zeta)], 
                        "D":[(zeta, delta),(beta, alpha)]
                    }) 
    grid.index = ['A','D']

    player1 = Agent('col_player', 1)
    player2 = Agent('row_player', 1)
    i = episodes
    while i > 0:
        choice1 = player1.choose_action()
        choice2 = player2.choose_action()
        print("col player chose {}, row player chose {} -> reward {}".format(choice1, choice2, grid[choice1][choice2]))
        i -= 1






