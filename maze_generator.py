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
    MAZE_H = 6 # height
    MAZE_W = 6 # width

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['U','D','L','R']
        self.action_count = 4
        self.title('Easy Maze')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT,
                                       self.MAZE_W * self.UNIT))
        self.build_maze()
        # state means the current state/position of the robot
        self.state_x = 0
        self.state_y = 0
        self.hells = [[3,2],[3,3],[3,4],[3,5],[4,5],[1,0],[1,1],[1,2],[1,4],[1,5]]
        self.reward = [5,5]
        # root = tk.Tk()
        # root.mainloop()


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

        self.hells = [
            self._draw_rect(3, 2, 'black'),
            self._draw_rect(3, 3, 'black'),
            self._draw_rect(3, 4, 'black'),
            self._draw_rect(3, 5, 'black'),
            self._draw_rect(4, 5, 'black'),
            self._draw_rect(1, 0, 'black'),
            self._draw_rect(1, 1, 'black'),
            self._draw_rect(1, 2, 'black'),
            self._draw_rect(1, 4, 'black'),
            self._draw_rect(1, 5, 'black')
        ]
        self.hell_coords = []
        for hell in self.hells:
            self.hell_coords.append(self.canvas.coords(hell))

        # 奖励
        self.oval = self._draw_rect(5, 5, 'yellow')
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
        if new_state[0] <0 or new_state[0]>5:
            return -1, new_state
        if new_state[1] < 0 or new_state[1] >5:
            return -1, new_state
        #有没有进入陷阱

        if new_state in self.hells:
            return -1, new_state
        if new_state == self.reward:
            return 1, new_state
        return 0,new_state



