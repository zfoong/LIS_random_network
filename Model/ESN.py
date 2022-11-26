# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:19:52 2022

@author: zfoong
"""

import numpy as np
from scipy import linalg 
from datetime import datetime
import os
from Model.INetwork_Model import *

class ESN(INetwork_Model):

    def __init__(self, in_size=1, out_size=1, population=1200, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.network_init(in_size, out_size, population)
        self.state_history = []
    
    def network_init(self, in_size=1, out_size=1, population=1200):
        self.inSize = in_size
        self.outSize = out_size
        self.resSize = population
        
        self.a = 0.7
        
        self.Win = (np.random.rand(self.resSize,self.inSize) - 0.5) * 1
        self.Wout = (np.random.rand(self.resSize,self.outSize) - 0.5) * 1
        self.W = np.random.rand(self.resSize,self.resSize) - 0.5 
        self.state = np.random.rand(self.resSize,1) - 0.5 
        
        rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= 1.25 / rhoW
    
    def input(self, input_):
        input_ = np.matrix(input_)
        self.state = (1-self.a)*self.state + self.a*np.array(np.tanh(np.dot(self.Win, input_.T) + np.dot(self.W, self.state)))
    
    def learn(self):
        pass
    
    def output(self):
        output = np.dot( self.Wout.T, self.state )
        return np.array(output).flatten()

    def save_model(self, path_name):
        path_name = path_name + "/" + datetime.now().strftime('%d%m%Y_%H%M%S')
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
        
    def reset_state(self):
        self.state = np.random.rand(self.resSize,1) - 0.5 