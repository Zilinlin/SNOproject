# this is the code for REINFORCE算法

from cmath import pi
from email.contentmanager import raw_data_manager
from mmap import ACCESS_COPY
from re import S
#from maze_generator import Maze
from maze_generator_middle import Maze
import numpy as np
import pandas as pd
import math

class Reinforce():
    def __init__(self,actions,episode =200,epsilon=0.9,learning_rate=0.01,greedy=0.8):
        self.actions = actions
        self.maze = Maze()
        self.episode = episode
        self.initial_state = [0,0]
        #初始化theta和pi
        self.theta = np.zeros((10,10,4))
        self.pi = self.convert_from_theta_to_pi()
        self.old_pi  = self.convert_from_theta_to_pi()
        #初始化是上下左右四个方向都是0.25的概率
        #self.theta[:,:,:] = 0.25
        self.epsilon = epsilon
        self.lr = learning_rate
        self.greedy = greedy
        self.losses = []
        

    #https://blog.csdn.net/level_code/article/details/100401202
    def convert_from_theta_to_pi(self):
        beta = 1.0
        [x,y,z] = self.theta.shape
        pi = np.zeros((x,y,z))
        exp_theta = np.exp(beta * self.theta)
        for i in range(0,x):
            for j in range(0,y):
                pi[i,j,:] = exp_theta[i,j,:]/np.nansum(exp_theta[i,j,:])
        pi = np.nan_to_num(pi)
        return pi
    
    def calculate_loss(self):
        pi_table = self.pi
        dist = np.sqrt(np.sum(np.square(pi_table-self.old_pi)))
        self.losses.append(dist)
        for i in range(10):
            for j in range(10):
                for k in range(4):
                    self.old_pi[i][j][k] = self.pi[i][j][k]


    #根据theta表的各个动作的概率函数进行选择
    def choose_action(self, state):
        actions = [0,1,2,3]
        action = np.random.choice(actions,p=self.pi[state[0]][state[1]])
        return action

    #将他们进行正则化，即和为1
    def normalize(self,state):
        thetas = self.theta[state[0]][state[1]]
        sum = thetas[0] + thetas[1] +thetas[2] + thetas[3]
        thetas = thetas/sum
        self.theta[state[0]][state[1]] = thetas

    def update_theta(self,states,actions,terminal_reward):
        length = len(actions)
        state = states[length-1]
        action = actions[length-1]
        #print(action)
        old_value = self.theta[state[0]][state[1]][action]
        delta = math.pow(self.epsilon,1)*terminal_reward
        new_value = old_value + self.lr*delta
        self.theta[state[0]][state[1]][action] = new_value
        self.pi = self.convert_from_theta_to_pi()
        # if new_value <0:
        #     self.theta[state[0]][state[1]][action] =0
        # self.normalize(state)

    def training(self):
        for i in range(self.episode):
            #这两个变量记录这个episode过程的state和action，并且是一一对应的
            states = []
            actions = []
            terminal_reward = 0
            #self。state 表示当前智能体所在的位置
            self.state = [0,0]
            states.append(self.state)
            step =0
            while True:
                step += 1
                action = self.choose_action(self.state)
                actions.append(action)
                reward,state_ = self.maze.move(self.state,action)

                #防止进入循环，所以如果已经大于两百步，就直接放弃
                if step > 200:
                    break

                if reward!=0:
                    #当前episode结束
                    terminal_reward = reward
                    self.update_theta(states,actions,terminal_reward)
                    self.calculate_loss()
                    self.state = [0,0]

                    break
                else:
                    self.state = state_
                    states.append(state_)


