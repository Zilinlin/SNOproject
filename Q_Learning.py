from maze_generator import Maze
import numpy as np
import pandas as pd

class QLearning():
    def __init__(self, actions, learning_rate=0.05, reward_discount = 0.9, episode = 200):
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
        self.q_table = np.zeros((6,6,4))
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

    def choose_action(self,state):

        #self.check_state_exist(state)
        state_actions = self.q_table[state[0]][state[1]]
        #如果有很多个，需要随机选择一个

        action = np.random.choice(np.where(state_actions==np.max(state_actions))[0])

        return action

    # 这是一个学习过程中更新q-table的方法,state_是state采取action之后的
    def update_q_table(self,state,action,reward,state_):
        #self.check_state_exist(state)
        #self.check_state_exist(state_)
        #首先是旧的值
        old_value = self.q_table[state[0]][state[1]][action]
        # 这意味着还没有结束
        if(reward!=0):
            new_value = reward
        else:
            #new_value = reward + self.reward_discount*self.q_table.loc[state_,:].max()
            new_value = reward + self.reward_discount *np.max(self.q_table[state_[0]][state_[1]])

        self.q_table[state[0]][state[1]][action] = old_value + self.learning_rate*(new_value - old_value)


    def training(self):
        for i in range(self.episode):

            self.state = [0,0]
            print("-----i------------------initial state", self.state, self.initial_state)
            while True:

                action = self.choose_action(self.state)
                reward, state_ = self.maze.move(self.state,action)
                print(self.state,reward,state_)
                self.update_q_table(self.state,action,reward,state_)

                if reward!=0:
                    self.state = [0, 0]
                    print("------------------")
                    break
                else:
                    self.state = state_






