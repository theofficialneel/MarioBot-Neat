import gym
import os
import time
import random
import tensorflow as tf
import numpy as np
from gym.envs.registration import register

try:
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery':False},
        max_episode_steps=200,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    print("Error")

env = gym.make('FrozenLakeNoSlip-v0')

def clear():
    os.system('clear')

class Agent(object):
    def __init__(self, env):
        self.is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)
        
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)
        
    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                                       self.action_high,
                                       self.action_shape)
        return action

class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        Agent.__init__(self, env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)
        
        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def build_model(self):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.int32, shape=[1])
        self.action_in = tf.placeholder(tf.int32, shape=[1])
        self.target_in = tf.placeholder(tf.float32, shape=[1])
        
        self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.one_hot(self.action_in, depth=self.action_size)
        
        self.q_state = tf.layers.dense(self.state, units=self.action_size, name="q_table")
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
        
        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
        action_greedy = np.argmax(q_state)
        action_random = Agent.get_action(self, state)
        return action_random if random.random() < self.eps else action_greedy
    
    def train(self, experience):
        state, action, next_state, reward, done = ([exp] for exp in experience)
        
        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)
        
        feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
        self.sess.run(self.optimizer, feed_dict=feed)
        
        if experience[4]:
            self.eps = self.eps * 0.99
            
    def __del__(self):
        self.sess.close()

agent = QNAgent(env)

total_reward = 0
for ep in range(200):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state,action,next_state,reward,done))
        state = next_state
        total_reward += reward
        
        print("s:", state, "a:", action)
        print("Episode: {}, Total reward: {}, eps: {}".format(ep,total_reward,agent.eps))
        env.render()
        with tf.variable_scope("q_table", reuse=True):
            weights = agent.sess.run(tf.get_variable("kernel"))
            print(weights)
        time.sleep(0.05)
        clear()

env.close()
