from maze_generator import Maze
from Q_Learning import QLearning



maze = Maze()
qlearning = QLearning(maze.action_space)
qlearning.training()
print(qlearning.q_table)

current_state = [0,0]
print(current_state)
while True:
    action = qlearning.choose_action(current_state)
    reward, state_ = maze.move(current_state,action)
    if reward!=0:
        break
    else:
        current_state = state_
        print(current_state)
