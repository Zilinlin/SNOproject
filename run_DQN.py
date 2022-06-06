# this is code for running and testing the DQN code
from argparse import Action
#from os import pread
import random
#from readline import write_history_file
from tkinter import font
from maze_env import Maze
from DQN import DQN
import numpy as np

maze = Maze()
DeepQN = DQN(maze.n_actions, maze.n_features,
    learning_rate=0.01,
    reward_discount=0.9,
    e_greedy=0.9,
    replace_target_iterations=200,
    memory_replay_size=2000,
    batch_size=32
    )

#现在的问题是怎么把环境传递到
def run_maze():
    step = 0
    for episode in range(300):
        current_state = maze.reset()
        count_step = 0
        while True:
            maze.render()
            count_step = count_step + 1
            print(current_state)
            action = DeepQN.choose_action(current_state)

            # 传入当前state得到之后的
            state_, reward, done = maze.step(action)
            print(reward)
            DeepQN.store_memory(current_state,action,reward,state_)

            #因为memory的储存是200，所以再进行200步之后进行学习
            if(step > 200) and (step % 5 ==0):
                print('start Deep Qlearning network')
                DeepQN.learn()

            current_state = state_

            if done:
                print("本次尝试的步骤：",count_step,"结果：",reward)
                break
            step += 1

    print('game over')
    #print(DeepQN.memory)
run_maze()

#显示迷宫的样子
#maze.show_maze()
DeepQN.plot_cost()