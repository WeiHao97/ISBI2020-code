import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from numpy import genfromtxt
from torchdiffeq import odeint

import scipy
import csv
import pandas as pd
import networkx as nx
import os
import random

if __name__ == '__main__':
    pib_v1_ = genfromtxt('data_v1.csv', delimiter=',',skip_header=0).astype(float)
    pib_v2_ = genfromtxt('data_v2.csv', delimiter=',',skip_header=0).astype(float)
    pib_v3_ = genfromtxt('data_v2.csv', delimiter=',',skip_header=0).astype(float)
    num_obj = pib_v1_.shape[0]-1

    pib_v1 = np.zeros((num_obj, 16))
    pib_v2 = np.zeros((num_obj, 16))
    pib_v3 = np.zeros((num_obj, 16))
    for i in range(0,num_obj):
        pib_v1[i] = np.delete(pib_v1_[i + 1], list(range(0,43))+ list(range(45,49)) + list(range(51,53)) + list(range(55,81)) +list(range(87,99))+list(range(101,103)) + list(range(105,259)))
        pib_v2[i] = np.delete(pib_v2_[i + 1], list(range(0,43))+ list(range(45,49)) + list(range(51,53)) + list(range(55,81)) +list(range(87,99))+list(range(101,103)) + list(range(105,259)))
        pib_v3[i] = np.delete(pib_v3_[i + 1], list(range(0,43))+ list(range(45,49)) + list(range(51,53)) + list(range(55,81)) +list(range(87,99))+list(range(101,103)) + list(range(105,259)))
        
    filePath = './v1/'
    filelist = os.listdir(filePath)
    Connect_v1 =  np.zeros((num_obj, 16, 16))
    for i in range(0,num_obj):
        temp = genfromtxt(filePath + filelist[i], delimiter=',').astype(float)
        temp = np.delete(temp, list(range(0,25)) + list(range(27,31)) + list(range(33,35)) + list(range(37,63)) +list(range(69,81))+list(range(83,85)) + list(range(87,123)), 0)
        temp = np.delete(temp, list(range(0,25)) + list(range(27,31)) + list(range(33,35)) + list(range(37,63)) +list(range(69,81))+list(range(83,85)) + list(range(87,123)), 1)
        temp = nx.from_numpy_matrix(temp)
        temp = nx.laplacian_matrix(temp)
        Connect_v1[i] = scipy.sparse.csr_matrix.todense(temp)
    
        
    X_train = torch.tensor(pib_v1).float()
    X_test = torch.tensor(pib_v1).float()
    y_train = torch.tensor(pib_v2).float()
    y_test = torch.tensor(pib_v3).float()
    L_train = torch.tensor(Connect_v1).float()
    L_test = torch.tensor(Connect_v1).float()
    
    
    class ODEFunc(nn.Module):

        def __init__(self,L):
            super(ODEFunc, self).__init__()
            self.B = nn.Parameter(torch.zeros(16,16))
            torch.nn.init.ones_(self.B) #or any other init metho
            self.B.data = -0.008*self.B.data
            self.L = L

        def forward(self, t, y):
            B_trans = torch.tensor(np.transpose(self.B.detach().numpy())).float()
            M = self.L*(self.B + nn.Parameter(B_trans))/2
            return y @ M.t().float()
    
        def setL(self,L_):
            self.L = L_
        
    class ODEBlock(nn.Module):

        #initialized as an ODE Function
        #count the time
        def __init__(self, odefunc):
            super(ODEBlock, self).__init__()
            self.odefunc = odefunc

        #foorward pass 
        #input the ODE function and input data into the ODE Solver (adjoint method)
        # to compute a forward pass
        def forward(self,integration_time, x, L):
            self.odefunc.setL(L)
            out = odeint(self.odefunc, x , integration_time.type_as(x))
            return out
            
    model = ODEBlock(ODEFunc(L_train[0]))
    loss = nn.MSELoss()##nn.L1Loss()
    
    def myloss(y,pred_y,reg):
        return loss(y,pred_y) + np.linalg.det(reg)
        
    optimizer = optim.SGD(model.parameters(), lr=1e-3)#optim.RMSprop(model.parameters(), lr=1e-3)
    loss_values = []
    loss_values_test = []
    itr = 0
    loss_train = 1
    while loss_train > 0.0001 and itr<2000:
        integration_t= torch.tensor([0,0.3]).float()###############time
        optimizer.zero_grad()
        loss_train = 0
        loss_test = 0
        ll = 0
        for i in range(0,X_train.shape[0]):
            pred_y = model(integration_t,X_train[i], L_train[i])
            my_B = model.odefunc.B.data.numpy()
            reg =(np.transpose(my_B) + my_B)/2
            loss_train = loss_train + myloss(pred_y[1], y_train[i],reg)
            ll = ll + loss(pred_y[1], y_train[i])
        ll = ll/X_train.shape[0]
        loss_train = loss_train/X_train.shape[0]
        loss_values.append(ll.item())
        loss_train.backward()
        optimizer.step()
        itr = itr + 1
        
        with torch.no_grad():
            integration_t= torch.tensor([0,0.4]).float()###############time
            for i in range(0,X_test.shape[0]):
                pred_y = model(integration_t,X_test[i], L_test[i])
                loss_test = loss_test + loss(pred_y[1], y_test[i])
            loss_test = loss_test/X_test.shape[0]
            loss_values_test.append(loss_test.item())
        #print('Iter {:04d} | train Loss {:.6f}| test Loss {:.6f}'.format(itr, loss_train.item(),loss_test.item()))
        
        x1 = range(2, itr)
        y1 = loss_values[2:itr]
        plt.plot(x1, y1, '-')
        plt.title('Loss_train vs. iterations')
        plt.ylabel('Loss_train')
        plt.savefig("pre_train.png")
        plt.close()
        
        x1 = range(2, itr)
        y1 = loss_values_test[2:itr]
        plt.plot(x1, y1, '-')
        plt.title('Loss_test vs. iterations')
        plt.ylabel('Loss_test')
        plt.savefig("pre_test.png")
        
        torch.save(model.state_dict(), './pretain_model.pth')
        
