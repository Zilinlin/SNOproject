from random import random,sample
import sys
import time
import numpy as np

#we choose to use Tkinter  as the library to draw the maze picture
if sys.version_info.major ==2:
    import Tkinter as tk
else:
    import tkinter as tk

class Maze(tk.Tk, object):
    UNIT = 40 # pixel
    MAZE_H = 10 # height
    MAZE_W = 10 # width

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['U','D','L','R']
        self.action_count = 4
        self.title('Middle Maze')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT,
                                       self.MAZE_W * self.UNIT))
        
        # state means the current state/position of the robot
        self.state_x = 0
        self.state_y = 0
        self.hells = [[0, 1], [0, 4], [1, 8], [2, 8], [2, 3], [3, 3], [4, 8], [4, 7],  \
            [6, 14], [6, 4], [7, 7], [7, 9], [8, 7], [8, 4], [9, 5], [9, 3],  \
            [1,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1], \
            [1,2],[1,3],[1,4],[1,5],[1,6],[1,7]  ,\
                [4,3],[4,4],[4,5],[4,6] ,[7,3],[7,4],[6,2],[6,3],[6,5],[6,6],[6,7],[3,0] ]

        self.reward = [9,9]
        self.build_maze()

    def build_maze_hells(self):
        list = range(15)
        
        for i in range(13):
            rs = sample(list,2)
            self.hells.append([i+1,rs[0]])
            self.hells.append([i+1,rs[1]])
        print("maze sample:", self.hells)

    def show_maze(self):
        root = tk.Tk()
        root.mainloop()

    #xy表示格坐标，是格子的个数
    def _draw_rect(self,x,y,color):
        center = self.UNIT /2
        w = center - 5
        draw_x = self.UNIT * x + center
        draw_y = self.UNIT * y + center
        return self.canvas.create_rectangle(draw_x -w, draw_y-w, draw_x+w, draw_y+w,fill=color)

    def build_maze(self):
        h = self.MAZE_H * self.UNIT
        w = self.MAZE_W * self.UNIT
        #画布
        self.canvas = tk.Canvas(self,bg='white',height=h,width=w)
        #画线
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)

        for hell in self.hells:
            self._draw_rect(hell[0],hell[1],'black')
        self.hell_coords = []
        for hell in self.hells:
            self.hell_coords.append(self.canvas.coords(hell))

        # 奖励
        self.oval = self._draw_rect(self.reward[0], self.reward[1], 'yellow')
        # 玩家对象
        self.rect = self._draw_rect(0, 0, 'red')

        self.canvas.pack()

    # 这部分是进行活动并且返回当前得到的奖励
    # 如果返回0就是可以继续，如果不是0，那么这一轮就直接结束了
    # U:0 D:1 L:2 R:3
    def move(self,state,action):
        new_state=[0,0]
        if action==0:
            new_state[1] = state[1]-1
            new_state[0] = state[0]
            #self.state_y = self.state_y -1
        elif action==1:
            new_state[1] = state[1]+1
            new_state[0] = state[0]
            #self.state_y = self.state_y +1
        elif action==2:
            new_state[0] = state[0]-1
            new_state[1] = state[1]
            #self.state_x = self.state_x-1
        else:
            new_state[0] = state[0]+1
            new_state[1] = state[1]
            #self.state_x = self.state_x +1
        #有没有撞墙
        #current_cordinate = [self.state_x, self.state_y]
        if new_state[0] <0 or new_state[0]>=self.MAZE_H:
            return -1, new_state
        if new_state[1] < 0 or new_state[1] >=self.MAZE_H:
            return -1, new_state
        #有没有进入陷阱

        if new_state in self.hells:
            return -1, new_state
        if new_state == self.reward:
            return 1, new_state
        return 0,new_state
    
    #这个函数和上面的move函数意思一样，但是返回值稍微进行了改变
    #返回 new_dtate, reward, done(是否结束)
    def step(self,state,action):
        new_state=[0,0]
        if action==0:
            new_state[1] = state[1]-1
            new_state[0] = state[0]
            #self.state_y = self.state_y -1
        elif action==1:
            new_state[1] = state[1]+1
            new_state[0] = state[0]
            #self.state_y = self.state_y +1
        elif action==2:
            new_state[0] = state[0]-1
            new_state[1] = state[1]
            #self.state_x = self.state_x-1
        else:
            new_state[0] = state[0]+1
            new_state[1] = state[1]
            #self.state_x = self.state_x +1
        #有没有撞墙
        #current_cordinate = [self.state_x, self.state_y]
        if new_state[0] <0 or new_state[0]>=self.MAZE_H:
            return new_state,-1,True
        if new_state[1] < 0 or new_state[1] >=self.MAZE_H:
            return new_state,-1,True
        #有没有进入陷阱

        if new_state in self.hells:
            return new_state,-1,True
        if new_state == self.reward:
            print("-----------------------------reward------------------------")
            return new_state,1,True
        return new_state,0,False




