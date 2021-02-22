import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



class Grid: # Environment
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions, obey_prob):
    self.rewards = rewards
    self.actions = actions
    self.obey_prob = obey_prob

  def non_terminal_states(self):
    return self.actions.keys()

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def non_deterministic_move(self, action):
    p = np.random.random()
    if p <= self.obey_prob:
      return action
    if action == 'U' or action == 'D':
      return np.random.choice(['L', 'R'])
    elif action == 'L' or action == 'R':
      return np.random.choice(['U', 'D'])

  def move(self, action):
    actual_action = self.deterministic(action)
    if actual_action in self.actions[(self.i, self.j)]:
      if actual_action == 'U':
        self.i -= 1
      elif actual_action == 'D':
        self.i += 1
      elif actual_action == 'R':
        self.j += 1
      elif actual_action == 'L':
        self.j -= 1
    return self.rewards.get((self.i, self.j), 0)

  def check_move(self, action):
    i = self.i
    j = self.j
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        i -= 1
      elif action == 'D':
        i += 1
      elif action == 'R':
        j += 1
      elif action == 'L':
        j -= 1
    # return a reward (if any)
    reward = self.rewards.get((i, j), 0)
    return ((i, j), reward)

  def get_transition_probs(self, action):
    # returns a list of (probability, reward, s') transition tuples
    probs = []
    state, reward = self.check_move(action)
    probs.append((self.obey_prob, reward, state))
    disobey_prob = 1 - self.obey_prob
    if not (disobey_prob > 0.0):
      return probs
    if action == 'U' or action == 'D':
      state, reward = self.check_move('L')
      probs.append((disobey_prob / 2, reward, state))
      state, reward = self.check_move('R')
      probs.append((disobey_prob / 2, reward, state))
    elif action == 'L' or action == 'R':
      state, reward = self.check_move('U')
      probs.append((disobey_prob / 2, reward, state))
      state, reward = self.check_move('D')
      probs.append((disobey_prob / 2, reward, state))
    return probs

  def all_states(self):
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid(obey_prob=1.0, step_cost=None):
  g = Grid(3, 4, (2, 0))
  rewards = {(0, 3): 1, (1, 3): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  g.set(rewards, actions, obey_prob)
  if step_cost is not None:
    g.rewards.update({
      (0, 0): step_cost,
      (0, 1): step_cost,
      (0, 2): step_cost,
      (1, 0): step_cost,
      (1, 2): step_cost,
      (2, 0): step_cost,
      (2, 1): step_cost,
      (2, 2): step_cost,
      (2, 3): step_cost,
    })
  return g

def print_values(V, g):
    for i in range(g.width):
      print("---------------------------")
      for j in range(g.height):
        v = V.get((i,j), 0)
        if v >= 0:
          print(" %.3f|" % v, end="")
        else:
          print("%.3f|" % v, end="")
      print("")

def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")


THETA = 1e-15
GAMMAS = [0.2, 0.6, 0.9]
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def best_action_value(grid, V, s, gamma):
  # finds the highest value action (max_a) from state s, returns the action and value
  best_a = None
  best_value = float('-inf')
  grid.set_state(s)
  R = grid.rewards[s]
  # loop through all possible actions to find the best current action
  for a in ALL_POSSIBLE_ACTIONS:
    transititions = grid.get_transition_probs(a)
    expected_v = 0
    expected_r = 0
    for (prob, r, state_prime) in transititions:
      # expected_r += prob * r
      expected_v += prob * V[state_prime]
    # v = -0.04 + gamma * (expected_r +  expected_v)
    v =  R + gamma * expected_v
    if v > best_value:
      best_value = v
      best_a = a
  return best_a, best_value

def calculate_state_values(grid, gamma):
  change_history = []
  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0
  
  V[(0,3)] = 1
  V[(1,3)] = -1
  # repeat until convergence
  # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
  iter = 0
  while iter < 200:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    change_acc = 0
    for s in grid.non_terminal_states():
      # print(s)
      old_v = V[s]
      _, new_v = best_action_value(grid, V, s, gamma)
      V[s] = new_v
      biggest_change = max(biggest_change, np.abs(old_v - new_v))
      change_acc += biggest_change
    
    change_history.append(change_acc)
      
    iter += 1
    # if biggest_change < THETA  :
    #   break
  return V, change_history

def initialize_random_policy():
  # policy is a lookup table for state -> action
  # we'll randomly choose an action and update as we learn
  policy = {}
  for s in grid.non_terminal_states():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
  return policy

def calculate_greedy_policy(grid, V, gamma):
  policy = initialize_random_policy()
  # find a policy that leads to optimal value function
  for s in policy.keys():
    grid.set_state(s)
    # loop through all possible actions to find the best current action
    best_a, _ = best_action_value(grid, V, s, gamma)
    policy[s] = best_a
  return policy


if __name__ == '__main__':

  grid = standard_grid(obey_prob=0.8, step_cost=-0.04)
  print("rewards:")
  print_values(grid.rewards, grid)
  convergence_plots = {}
  for g in GAMMAS:
    # calculate accurate values for each square
    V, history = calculate_state_values(grid, gamma=g)
    convergence_plots[g] = history


    # calculate the optimum policy based on our values
    # policy = calculate_greedy_policy(grid, V, gamma=g)

    # our goal here is to verify that we get the same answer as with policy iteration
    print("values for gamma %f:"%g)
    print_values(V, grid)
    # print("policy:")
    # print_policy(policy, grid)
  
  df = pd.DataFrame(convergence_plots)
  plt.figure()
  
  df.plot(lw=2, colormap='jet', marker='.', markersize=10,
          title='Convergence plot for different gammas')
  plt.xlabel('Iterations')
  plt.ylabel('Change of state values')
  plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
  plt.savefig("gammas.png", format="png")

