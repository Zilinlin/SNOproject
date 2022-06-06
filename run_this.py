'''
this is run_this.py
'''
from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        count_step = 0
        while True:
            # fresh env 刷新环境
            env.render()
            count_step = count_step+1


            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # 目前done只能显示游戏结束，如果对done值形式改变一下可以反应游戏胜利/失败。

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                # 前面步骤Q网络没更新好，此时学习不具意义。在运行一段时间后，可以进行神经网络的更新。
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print("本次所尝试的步数：",count_step)
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # tensorboard --logdir=logs 网页查看结构
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.99,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True,
                      e_greedy_increment=0.01
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
