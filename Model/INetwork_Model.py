# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:15:58 2022

@author: zfoong
"""

from abc import ABC, abstractmethod
 
class INetwork_Model(ABC):
    
    @abstractmethod
    def network_init(self):
        pass
    
    @abstractmethod
    def input(self, input_):
        pass
    
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def output(self):
        pass
    
    @abstractmethod
    def save_model(self, path_name):
        pass
    
    @abstractmethod
    def load_model(self, path_name):
        pass
 
