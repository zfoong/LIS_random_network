# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:40:49 2022

@author: zfoong
"""

import numpy as np
import gym 
# from Model.Network_Model import *
from Model.random_network_model import *
import matplotlib.pyplot as plt
import time

# Parameters
population = 100
iteration = 100
render = False

# Record
reward_progression = []

# Initalize Gym Environment
env = gym.make("CartPole-v0")
state_size = 5 #env.observation_space.shape[0]
action_size = 1 #env.action_space.n
env.reset()

# Initlize Model
model = random_network_model(state_size, action_size, population)

for i in range(iteration):
    total_reward = 0
    model.reset_state()
    init_input = np.append(env.state, 0)
    model.input(init_input)
    
    while True:
        output = model.output()
        
        # Clip output
        output = int(np.clip(output, 0, 1)[0])
        
        # observation, reward, done, info = env.step(env.action_space.sample())
        observation, reward, done, info = env.step(output)
        
        # print("step", i, observation, reward, done, info)
        total_reward += reward
        
        if render:
            env.render()
            time.sleep(0.001)
        
        if done:
            env.reset()
            break
        
        input_ = np.append(observation, reward)
        model.input(input_)
        
    reward_progression.append(total_reward)


env.close()
plt.plot(reward_progression)