#This is the code for SARSA algorithm, and most of this 
# part is similar to the Q-Learning algorithm

from os import stat
from re import T
#from maze_generator import Maze
from maze_generator_middle import Maze
import numpy as np
import pandas as pd

class SARSA():

    def __init__(self, actions, learning_rate=0.05, reward_discount = 0.9, episode = 200,greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_discount = reward_discount
        self.episode = episode
        self.maze = Maze()
        self.state_x = self.maze.state_x
        self.state_y = self.maze.state_y
        self.state = [self.state_x,self.state_y]
        self.initial_state = [0,0]
        self.actions = self.maze.action_space
        #the part means the x,y,actions
        self.q_table = np.zeros((10,10,4))
        self.old_qtable= np.zeros((10,10,4))
        self.losses = []
        self.greedy = greedy
        #self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

            #将maze值和当前的值进行同步
    def update_state(self):
        self.state = [self.maze.state_x,self.maze.state_y]
        self.state_y = self.maze.state_y
        self.state_x = self.maze.state_x

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name="a"
                )
            )
    def calculate_losss(self):
        q_table = self.q_table
        dist = np.sqrt(np.sum(np.square(q_table-self.old_qtable)))
        print('distance',dist)
        self.losses.append(dist)
        for i in range(10):
            for j in range(10):
                for k in range(4):
                    self.old_qtable[i][j][k] = self.q_table[i][j][k]


    def choose_action(self,state):
        if np.random.uniform() <self.greedy:

            #self.check_state_exist(state)
            state_actions = self.q_table[state[0]][state[1]]
            #如果有很多个，需要随机选择一个

            action = np.random.choice(np.where(state_actions==np.max(state_actions))[0])
        else:
            action = np.random.randint(0,4)
        return action
    
    def update_q_table(self,state,action,reward,state_,action_,done):
        old_value = self.q_table[state[0]][state[1]][action]
        if done:
            new_value = reward
        else:
            new_value = reward + self.reward_discount*self.q_table[state_[0]][state_[1]][action_]
        
        self.q_table[state[0]][state[1]][action] = old_value + self.learning_rate*(new_value - old_value)

    def training(self):
        for i in range(self.episode):
            self.state=[0,0]
            step = 0
            while True:
                action = self.choose_action(self.state)
                state_, reward,done = self.maze.step(self.state, action)
                step += 1
                if step >50:
                    break
                if done:
                    
                    #直接在这里进行更新，就不去update函数里面更新了
                    new_value = reward
                    old_value = self.q_table[self.state[0]][self.state[1]][action]
                    self.q_table[self.state[0]][self.state[1]][action] = old_value + self.learning_rate*(new_value - old_value)

                    self.state=[0,0]
                    self.calculate_losss()
                    break
                else:
                    action_ = self.choose_action(state_)

                    self.update_q_table(self.state, action,reward,state_,action_,done)

                    self.state = state_
