# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:12:42 2021

@author: zfoong
"""

    
import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg 
from scipy.special import softmax
from datetime import datetime
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.fft import fft, fftfreq
import matplotlib.gridspec as gridspec
from scipy import signal
import random
from Model.INetwork_Model import *

class random_network_model(INetwork_Model):
    
    def __init__(self, in_size=1, out_size=2, population=600):        
        # np.random.seed(100)
        self.inSize = in_size
        self.outSize = out_size
        self.resSize = population
        
        self.Win = None
        self.W = None
        self.Wout = None
        self.state = None
        
        self.max_delay = 6
        
        self.time_delay_matrix = np.random.randint(0, self.max_delay, size=(self.resSize, self.resSize))
        self.time_delay_modifier = np.empty((self.resSize, self.resSize, self.max_delay))
        
        for i in range(self.max_delay):
            self.time_delay_modifier[:,:,i] = (self.time_delay_matrix == i).astype(int)      
            
        self.buffer = np.zeros((self.resSize, 1, self.max_delay))
        
        self.network_init()
        self.state_history = []
        
        self.reward_mean = 0
        
    def print_connection(self):
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(self.Win)
        
        axs[1].imshow(self.W, aspect='auto', interpolation='nearest')
        # axs[1].matshow(self.W)

        # for (i, j), z in np.ndenumerate(self.W):
        #     axs[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        im = axs[2].imshow(self.Wout)
        
        axs[0].set_title('W in')
        axs[1].set_title('W')
        axs[2].set_title('W out')
        
        fig.subplots_adjust(right=0.8)
        
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout()
        plt.show()
        
    def network_init(self):
        
        self.lr = 0.01
        self.thrs = 1
        self.a = np.random.normal(0.5, 0.1,(self.resSize,1)) # leaking rate
        np.clip(self.a, 0.2, 0.9)
        
        # self.a = 0.7
        
        # self.a[:100] = 0.2
        # self.a[100:] = 0.7
        
        # plt.figure(figsize=(2,8))
        # plt.imshow(self.a)
        # plt.colorbar()
        # plt.show()
                
        self.Win = (np.random.rand(self.resSize,self.inSize) - 0.5) * 1
        Wout_index = np.random.randint(0,self.resSize,self.outSize)
        self.Wout = np.zeros((self.resSize,self.outSize))
        for idx, i in enumerate(Wout_index):
            self.Wout[i, idx] = 1
        self.W = np.random.rand(self.resSize,self.resSize) - 0.5 
        
        sub_count = 3
        divider = int(self.resSize/sub_count)
        coup_cnt = 4 # coupling neurons count
        
        weak_couple_modifier = np.zeros((self.resSize,self.resSize))
        for i in range(sub_count):
            weak_couple_modifier[i*divider:(i+1)*divider,i*divider:(i+1)*divider] = 1

        
        for i in range(sub_count):
            d_coup_lower = i*divider-int(coup_cnt/2)
            d_coup_upper = i*divider+int(coup_cnt/2)
            weak_couple_modifier[d_coup_lower:d_coup_upper, d_coup_lower:d_coup_upper] = 1
        
        self.W = self.W * weak_couple_modifier

        # --- Modify connection - pseudo-chaotic ---
        rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= 1.25 / rhoW
        
        # --- Initialize state ---
        # self.state = np.zeros((self.resSize,1))
        self.init_state = np.random.rand(self.resSize,1) - 0.5 
        self.state = self.init_state
        pass
    
    def input(self, input_):
        input_ = np.matrix(input_)
        self.state = (1-self.a)*self.state + self.a*np.array(np.tanh(np.dot(self.Win, input_.T) + np.dot(self.W, self.state)))
        self.state_history.append(self.state.copy())
    
    def learn(self, input_, reward=0):
        
        
        
        self.W += modifier
        pass
    
    def output(self):
        output = np.dot( self.Wout.T, self.state )
        return np.array(output).flatten()
    
    def self_update(self, noise=False, ni=0.01):
        n = np.zeros((self.resSize, 1))
        if noise:
            n = np.random.normal(0,1,(self.resSize, 1)) * ni
            
        for i in range(self.max_delay):
            self.buffer[:,:,i] += self.a * np.tanh(np.dot((self.W * self.time_delay_modifier[:,:,i]), self.state))
        
        self.state = (1-self.a)*self.state + self.buffer[:,:,0] + n     
        
        self.buffer = np.delete(self.buffer, (0), axis=2)
        self.buffer = np.dstack((self.buffer, np.zeros((self.resSize, 1))))
        
        self.state_history.append(self.state.copy())
    
    def mutate(self, mutate_mutiplier = 0.01):
        
        sparse_intensity = random.randint(4,9)
        sparse_modifier = np.ones((self.W.shape[0], self.W.shape[1]))
        for _ in range(sparse_intensity):
            sparse_modifier *= np.random.randint(2, size=(self.W.shape[0], self.W.shape[1]))
        self.W = self.W \
                 + np.random.rand(self.W.shape[0], self.W.shape[1]) \
                 * mutate_mutiplier \
                 * sparse_modifier
                
        # plt.imshow(sparse_modifier)
        # plt.show()
                
        sparse_modifier = np.ones((self.Wout.shape[0], self.Wout.shape[1]))
        for _ in range(sparse_intensity):
            sparse_modifier *= np.random.randint(2, size=(self.Wout.shape[0], self.Wout.shape[1]))
        self.Wout = self.Wout \
                    + np.random.rand(self.Wout.shape[0], self.Wout.shape[1]) \
                    * mutate_mutiplier \
                    * sparse_modifier
        
        return
        
    def save_model(self):
        path_name = "saved_model/"+ datetime.now().strftime('%d%m%Y_%H%M%S')
        os.mkdir(path_name)
        np.save(path_name+"/Win", self.Win)
        np.save(path_name+"/Wout", self.Wout)
        np.save(path_name+"/W", self.W)
        np.save(path_name+"/state", self.state)
    
    def load_model(self, path_name):
        self.Win = np.load(path_name+"/Win.npy")
        self.Wout = np.load(path_name+"/Wout.npy")
        self.W = np.load(path_name+"/W.npy")
        self.state = np.load(path_name+"/state.npy")
        
    def animate_func(self, i):    
        self.im.set_array(self.state_history[i])
        return [self.im]
    
    def plot_animation(self):
        fps = 25
        nSeconds = 20
        
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure( figsize=(2,8) )
        
        self.im = plt.imshow(self.state_history[0], interpolation='none', aspect='auto', vmin=0, vmax=1)
        
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = nSeconds * fps,
                                       interval = 1000 / fps, # in ms
                                       )
        
        anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
        
        print('Done!')
        
    def clear_state_history(self):
        self.state_history = []
        
    def reset_state(self):
        self.state = self.init_state
    
if __name__ == "__main__":
    
    def is_oscillate(r):
        if r > 1:
            return True
        return False

    def osc_metric(outputs, t):
        outputs = outputs[-t:]
        r = []
        for output in outputs:
            output = np.array(output.flatten())[0]
            N = len(output)
            S = 0
            norm = 1 / (N - 1)
            for i in range(len(output)-1):
                S += np.square(output[i+1] - output[i])
            r.append(np.sqrt(norm * S) / (norm * np.square(np.sum(output))))
        
        return np.mean(r)
    
    def run_model(model, ite, noise=False, clear_prev=False):
        
        if clear_prev is True:
            model.clear_state_history()
        
        outputs = []
        for i in range(iterations):
            model.self_update(noise)
            outputs.append(model.output())
            
        return outputs
    
    movie = False
    
    iterations = 2000
    # dt = 0.1
    # time = np.arange(0,100,dt)
    
    load_model = False
    load_path = "saved_model/2_by_2/"
    
    model = None
    if load_model == True:
        model = random_network_model()
        model.load_model(load_path)        
    else:
        model = random_network_model()
        
    model.print_connection()
    
    outputs = run_model(model, iterations, False, True)

    # --------------- validating output ------------------
    oscillation_model = True
    
    outputs = np.matrix(outputs)
    # for output in outputs.T:
    #     output = np.array(output.flatten())[0]
    #     if np.std(output[-10:]) < 0.01:
    #         print("No oscillation")
    #         oscillation_model = False
    
    # if oscillation_model:
    #     model.save_model()
    
    plt.plot(outputs)
    plt.show()
    
    state_history = np.squeeze(model.state_history)
    mean_state_history = np.mean(state_history, axis=1)
    plt.plot(mean_state_history)
    plt.show()

    # sample spacing
    T = 1.0 / 800.0
    yf = fft(mean_state_history)
    xf = fftfreq(iterations, T)[:iterations//2]
    import matplotlib.pyplot as plt
    plt.plot(xf, 2.0/iterations * np.abs(yf[0:iterations//2]))
    plt.grid()
    plt.show()
    
    for output in outputs.T:
        output = np.array(output.flatten())[0]
        f, Pxx_den = signal.periodogram(output[-50:], 1)
        plt.semilogy(f, Pxx_den)
        plt.show()
        
        if len(Pxx_den[Pxx_den > 0.0001]) == 0:
            print("No oscillation")
            oscillation_model = False
    
    display_count = 10
    sub_count = 3
    for i in range(sub_count):
        plt.figure(figsize=(5,20), dpi=300)
        sub_pop = int(model.resSize/sub_count)
        fig, axs = plt.subplots(display_count ,figsize=(5,20))
        for j in range(display_count):
            axs[j].plot(state_history[:,i*sub_pop+j])
        plt.show()
    
    # r = osc_metric(outputs.T, 50)
    # print(r)
    
    if movie is True:
        model.plot_animation()