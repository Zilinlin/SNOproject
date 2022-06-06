#from maze_generator import Maze
from turtle import done
from Q_Learning import QLearning
from SARSA import SARSA
from REINFORCE import Reinforce
import matplotlib.pyplot as plt

from maze_generator_middle import Maze


# ## this is the running code for Q-learning
# maze = Maze()
# maze.show_maze()
# qlearning = QLearning(maze.action_space,episode=10000,learning_rate=0.05)
# qlearning.training()
# print(qlearning.q_table)
# y = qlearning.losses
# x = range(len(y))
# plt.plot(x,y)
# plt.title('Q-Learning')
# plt.xlabel('episode')
# plt.ylabel('change')
# plt.show()

# current_state = [0,0]
# print(current_state)
# while True:
#     action = qlearning.choose_action(current_state,0.99)
#     state_,reward,done = maze.step(current_state,action)
#     if done:
#         break
#     else:
#         current_state = state_
#         print(current_state)
# print(qlearning.q_table[5][:])

# print("--------------start sarsa algorithm---------------")
# maze_sarsa = Maze()
# maze_sarsa.show_maze()
# sarsa = SARSA(maze_sarsa.action_space,episode=10000)
# sarsa.training()

# current_state = [0,0]
# while True:
#     action = sarsa.choose_action(current_state)
#     state_, reward, done = maze_sarsa.step(current_state,action)
#     if done:
#         break
#     else:
#          current_state = state_
#          print(current_state)
# print(sarsa.q_table[0][0])
# print("the end of sarsa algorithm")
# y = sarsa.losses
# x = range(len(y))
# plt.plot(x,y)
# plt.title('SARSA')
# plt.xlabel('episode')
# plt.ylabel('change')
# plt.show()


print("------------------start reinforce algorithm---------------")
maze_re = Maze()
reinforce = Reinforce(maze_re.action_space,episode=10000,learning_rate=0.5)
reinforce.training()
print(reinforce.pi[0,:])
#maze_re.show_maze()
current_state = [0,0]

y = reinforce.losses
x = range(len(y))
plt.plot(x,y)
plt.title('Policy Gradient')
plt.xlabel('episode')
plt.ylabel('change')
plt.show()


while True:
    action = reinforce.choose_action(current_state)
    reward,state_ = maze_re.move(current_state,action)
    if reward!=0:
        break
    else:
        current_state = state_
        print(current_state)
print("the end of reinforce algorithm")



