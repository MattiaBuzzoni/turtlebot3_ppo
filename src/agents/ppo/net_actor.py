"""This file contains a neural network module. It define our actor and critic networks in PPO."""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))


        Xout = self.fc1(x) 
        #Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        #Xout = self.bn2(Xout)
        Xout = self.fc2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class NetActor(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_neurons=512,
                 **kwargs):
        super(NetActor, self).__init__()

        self.rb1 = ResBlock(in_dim, in_dim)
        self.rb2 = ResBlock(in_dim + in_dim, in_dim + in_dim)
        self.rb3 = ResBlock(n_neurons + in_dim, n_neurons)

        self.out1 = nn.Linear(in_dim + in_dim, out_dim - 1)
        nn.init.orthogonal_(self.out1.weight, gain=0.01)
        self.out2 = nn.Linear(in_dim + in_dim, out_dim - 1)
        nn.init.orthogonal_(self.out2.weight, gain=0.01)

    def forward(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)

        X0 = obs

        # X0 = self.bn1(X)
        X = self.rb1(X0, True)
        X = self.rb2(torch.cat([X0, X], dim=-1), True)
        # X = self.rb3(torch.cat([X0, X], dim=-1), True)

        output1 = torch.tanh(self.out1(X))
        output2 = torch.tanh(self.out2(X))
        output = torch.cat((output1, output2), -1)
        return output