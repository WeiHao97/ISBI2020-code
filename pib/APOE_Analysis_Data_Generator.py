import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from numpy import genfromtxt
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import scipy
import csv
import pandas as pd
import networkx as nx
import os
import random

if __name__ == '__main__':
    pib_v1_ = genfromtxt('data_v1.csv', delimiter=',',skip_header=0).astype(float)
    pib_v2_ = genfromtxt('data_v2.csv', delimiter=',',skip_header=0).astype(float)
    num_obj = pib_v1_.shape[0]-1
    pib_v1 = np.zeros((num_obj, 16))
    pib_v2 = np.zeros((num_obj, 16))
    for i in range(0,num_obj):
        pib_v1[i] = np.delete(pib_v1_[i + 1], list(range(0,43))+ list(range(45,49)) + list(range(51,53)) + list(range(55,81)) +list(range(87,99))+list(range(101,103)) + list(range(105,259)))
        pib_v2[i] = np.delete(pib_v2_[i + 1], list(range(0,43))+ list(range(45,49)) + list(range(51,53)) + list(range(55,81)) +list(range(87,99))+list(range(101,103)) + list(range(105,259)))

    # define empty list
    positive = []

    # open file and read the content in a list
    with open('APOE_POS.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]

            # add item to the list
            positive.append(current_place)

    da = pd.read_csv('./data_v3.csv')
    list_pos = []
    list_neg = []
    list_all = []
    for i in range(0,da.shape[0]):
        if da['subjectid'][i] in positive:
            list_pos.append(i)
        else:
            list_neg.append(i)
        list_all.append(i)

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

    X_pos = torch.tensor(pib_v1[list_pos]).float()
    X_neg = torch.tensor(pib_v1[list_neg]).float()
    y_pos = torch.tensor(pib_v2[list_pos]).float()
    y_neg = torch.tensor(pib_v2[list_neg]).float()
    L_pos = torch.tensor(Connect_v1[list_pos]).float()
    L_neg = torch.tensor(Connect_v1[list_neg]).float()

    class ODEFunc(nn.Module):

        def __init__(self,L):
            super(ODEFunc, self).__init__()
            self.B = nn.Parameter(torch.zeros(16,16))
            torch.nn.init.ones_(self.B) #or any other init metho
            self.B.data = -0.01*self.B.data
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

    model_pos = ODEBlock(ODEFunc(L_pos[0]))#apoe+ model
    model_neg = ODEBlock(ODEFunc(L_neg[0]))#apoe- model
    loss = nn.MSELoss()##nn.L1Loss()

    def myloss(y,pred_y,reg):
        return loss(y,pred_y) + np.linalg.det(reg)

    optimizer = optim.SGD(model_pos.parameters(), lr=1e-3)#optim.RMSprop(model.parameters(), lr=1e-3)
    loss_values = []
    itr = 0
    loss_train = 1
    integration_t= torch.tensor([0,0.3]).float()###############time
    while loss_train > 0.0002 and itr<40:
        optimizer.zero_grad()
        loss_train = 0
        ll = 0
        for i in range(0,X_pos.shape[0]):
            pred_y = model_pos(integration_t,X_pos[i], L_pos[i])
            my_B = model_pos.odefunc.B.data.numpy()
            reg =(np.transpose(my_B) + my_B)/2
            loss_train = loss_train + myloss(pred_y[1], y_pos[i],reg)
            ll = ll + loss(pred_y[1], y_pos[i])
        ll = ll/X_pos.shape[0]
        loss_train = loss_train/X_pos.shape[0]
        loss_values.append(ll.item())
        loss_train.backward()
        optimizer.step()
        itr = itr + 1
        #print('Iter {:04d} | train Loss {:.6f}'.format(itr, loss_train.item()))

    x1 = range(2, itr)
    y1 = loss_values[2:itr]
    plt.plot(x1, y1, '-')
    plt.title('Loss_train vs. iterations')
    plt.ylabel('Loss_train')
    plt.savefig("APOE_pos_2.png")
    plt.close()

    optimizer = optim.SGD(model_neg.parameters(), lr=1e-3)#optim.RMSprop(model.parameters(), lr=1e-3)
    loss_values = []
    itr = 0
    loss_train = 1
    while loss_train > 0.0002 and itr<40:
        optimizer.zero_grad()
        loss_train = 0
        ll = 0
        for i in range(0,X_neg.shape[0]):
            pred_y = model_neg(integration_t,X_neg[i], L_neg[i])
            my_B = model_neg.odefunc.B.data.numpy()
            reg =(np.transpose(my_B) + my_B)/2
            loss_train = loss_train + myloss(pred_y[1], y_neg[i],reg)
            ll = ll + loss(pred_y[1], y_neg[i])
        ll = ll/X_neg.shape[0]
        loss_train = loss_train/X_neg.shape[0]
        loss_values.append(ll.item())
        loss_train.backward()
        optimizer.step()
        itr = itr + 1
        #print('Iter {:04d} | train Loss {:.6f}'.format(itr, loss_train.item()))

    x1 = range(2, itr)
    y1 = loss_values[2:itr]
    plt.plot(x1, y1, '-')
    plt.title('Loss_train vs. iterations')
    plt.ylabel('Loss_train')
    plt.savefig("APOE_neg_2.png")
    plt.close()

    loss_list = []
    loss_list.append(loss(torch.tensor(model_pos.odefunc.B.data.numpy()),torch.tensor(model_neg.odefunc.B.data.numpy())).numpy())

    for j in range(0,10000):#10000 random permutation
        class ODEFunc(nn.Module):

            def __init__(self,L):
                super(ODEFunc, self).__init__()
                self.B = nn.Parameter(torch.zeros(16,16))
                torch.nn.init.ones_(self.B) #or any other init metho
                self.B.data = -0.01*self.B.data
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

        random.shuffle(list_all)#creat random testing
        list_pos = list_all[0:45]
        list_neg = list_all[45:112]
        X_pos = torch.tensor(pib_v1[list_pos]).float()
        X_neg = torch.tensor(pib_v1[list_neg]).float()
        y_pos = torch.tensor(pib_v2[list_pos]).float()
        y_neg = torch.tensor(pib_v2[list_neg]).float()
        L_pos = torch.tensor(Connect_v1[list_pos]).float()
        L_neg = torch.tensor(Connect_v1[list_neg]).float()

        model_pos = ODEBlock(ODEFunc(L_pos[0]))
        model_neg = ODEBlock(ODEFunc(L_neg[0]))

        optimizer = optim.SGD(model_pos.parameters(), lr=1e-3)#optim.RMSprop(model.parameters(), lr=1e-3)
        itr = 0
        loss_train = 1
        while loss_train > 0.0002 and itr<40:
            optimizer.zero_grad()
            loss_train = 0
            for i in range(0,X_pos.shape[0]):
                pred_y = model_pos(integration_t,X_pos[i], L_pos[i])
                my_B = model_pos.odefunc.B.data.numpy()
                reg =(np.transpose(my_B) + my_B)/2
                loss_train = loss_train + myloss(pred_y[1], y_pos[i],reg)
            loss_train = loss_train/X_pos.shape[0]
            loss_train.backward()
            optimizer.step()
            itr = itr + 1

        optimizer = optim.SGD(model_neg.parameters(), lr=1e-3)#optim.RMSprop(model.parameters(), lr=1e-3)
        itr = 0
        loss_train = 1
        while loss_train > 0.0002 and itr<40:
            optimizer.zero_grad()
            loss_train = 0
            for i in range(0,X_neg.shape[0]):
                pred_y = model_neg(integration_t,X_neg[i], L_neg[i])
                my_B = model_neg.odefunc.B.data.numpy()
                reg =(np.transpose(my_B) + my_B)/2
                loss_train = loss_train + myloss(pred_y[1], y_neg[i],reg)
            loss_train = loss_train/X_neg.shape[0]
            loss_train.backward()
            optimizer.step()
            itr = itr + 1

        loss_list.append(loss(torch.tensor(model_pos.odefunc.B.data.numpy()),torch.tensor(model_neg.odefunc.B.data.numpy())).numpy())
        print(j)

    with open('LOSS_M_list_group_2.txt', 'w') as filehandle:
        for listitem in loss_list:
            filehandle.write('%s\n' % listitem)
