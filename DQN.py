# the version of Deep Q-Network is the version in NIPs2013
import collections
from operator import matmul
from re import T
from selectors import SelectorKey
import this
from unicodedata import name
import tensorflow as tf
# about the problem of tensorflow version
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from collections import deque
import random
import datetime

np.random.seed(1)
tf.set_random_seed(1)

class  DQN():
    def __init__(
        self,
        n_actions,
        n_features, # this means the number of features of one state
        learning_rate=0.01,
        reward_discount = 0.9,
        e_greedy = 0.9,
        replace_target_iterations=300, #经过300次迭代之后，需要对网络参数进行替换
        memory_replay_size = 500,
        batch_size = 32,
        e_greedy_increment = None,
        output_graph = False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.discount = reward_discount
        self.greed_prob = e_greedy
        self.replace_iterations = replace_target_iterations
        self.memory_size = memory_replay_size
        self.mini_batch_size = batch_size

        # the count of leaning steps
        self.learning_step = 0
        # this is the memory (s,a,r,s')
        self.memory = np.zeros((self.memory_size, n_features*2 +2))

        #there are two netwoks, "evaluation network" "target network"
        #they will be stored in "target_network_params and "evaluation_network_params"
        self.build_net()
        #下面两个量分别表示两个网络的参数
        target_params = tf.get_collection('target_network_params')
        evaluation_params = tf.get_collection('evaluation_network_params')
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(target_params,evaluation_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        #this value used to store the count number of memory size
        self.memory_count = 0

        self.cost_all = [] #所有的cost

    def build_net(self):
        # build evaluation network
        # this is the part of states
        self.states = tf.placeholder(tf.float32, [None,self.n_features], name='states')
        #每个状态都会有对应的Q-target
        self.q_target = tf.placeholder(tf.float32, [None,self.n_actions],name='q_target')
        # create the variable 'eval_net'
        #?? c_names(collection_names) are the collections to store variables
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['evaluation_network_params', tf.GraphKeys.GLOBAL_VARIABLES],10, \
                tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)

            # 关于这个层的设计是这样的
            # 第一层，
            # relu函数，x=0 when x<0 else x=x
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features, n_l1], initializer = w_initializer, collections = c_names)
                b1 = tf.get_variable('b1',[1,n_l1], initializer = b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.states,w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[n_l1, self.n_actions], initializer = w_initializer, collections= c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer = b_initializer, collections = c_names)
                self.q_eval = tf.matmul(l1,w2) + b2
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval,self.q_target))
            
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        #-build target net, the important thing is thinking about the 
        self.states_ = tf.placeholder(tf.float32, [None,self.n_features],name = 'states_')
        with tf.variable_scope('target_net'):
            c_names = ['target_network_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer = w_initializer, collections = c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer = b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.states_,w1)+b1)
            
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[n_l1, self.n_actions],initializer = w_initializer, collections = c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions], initializer = b_initializer, collections = c_names)
                self.q_next = tf.matmul(l1,w2) + b2

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        #如果小于0.9，选择最优的选项，如果大于0.9，进行随机选择
        #当前值的选择通过对环境的观察来实现
        if np.random.uniform() < self.greed_prob:
            actions_value = self.sess.run(self.q_eval,feed_dict= {self.states:observation})
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def store_memory(self, state, action, reward, state_):
        #将当前步骤的动作储存到记忆中，
        this_step = np.hstack((state,[action,reward],state_))

        index = self.memory_count % self.memory_size
        self.memory[index,:] = this_step
        self.memory_count += 1
    
    #  the function of learn
    # 这一部分是网络如何学习的关键
    #其中有很多需要注意的网络如何学习的细节
    def learn(self):
        #如果正好运行300次之后，需要将network_target的参数传递给network_evaluation
        if self.learning_step%self.replace_iterations ==0:
            #进行替代
            #replace_evaluation_network = [tf.assign(t,e) for t,e in zip(self.target_params,self.evaluation_params)]
            self.sess.run(self.replace_target_op)
        
        #选择出来一个minibatch size的数据进行训练
        if self.memory_count > self.memory_size:
            indexes = np.random.choice(self.memory_size, size=self.mini_batch_size)
        else:
            indexes = np.random.choice(self.memory_count,size=self.mini_batch_size)
        
        current_batch_memory = self.memory[indexes,:]
        #？？这里不是太明白，可以到时候再继续看一看想一想
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict = {
                #拿到每个记录转移之后的状态，以及转移之前的状态
                self.states_: current_batch_memory[:,-self.n_features:],
                self.states:current_batch_memory[:,:self.n_features],
            })
        
        #the q_eval is calculated with real actions and rewards
        q_target = q_eval.copy()

        batch_index = np.arange(self.mini_batch_size,dtype=np.int32)
        eval_act_index = current_batch_memory[:,self.n_features ].astype(int)
        reward = current_batch_memory[:,self.n_features+1]
        print("reward",reward)

        #利用Q-learning的方法，更新q-target的参数值
        q_target[batch_index,eval_act_index] = reward + self.discount*np.max(q_next,axis=1)

        #训练evaluation network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                    feed_dict = {
                                        self.states: current_batch_memory[:,:self.n_features],
                                        self.q_target: q_target
                                    })

        self.cost_all.append(self.cost)
        #这里本来是有需要增加epsilon的，但是我没有选择加上去
        self.learning_step += 1
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_all)), self.cost_all)
        plt.show







