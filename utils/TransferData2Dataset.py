#Transfer data of mat into Dataset
# writen by summerwine at 2020-11-21

import os, sys, time
import matplotlib
import numpy as np
from torch.utils.data.dataset import Dataset
import scipy.io as sio
import torch


class DataProcessing(Dataset):

    def __init__(self, root_path, data_path, data_usage, max_value, min_value,freq_star=None):
        self.root_path = root_path
        self.data_path = data_path
        self.data_usage = data_usage
        data_mat = sio.loadmat(root_path+data_path)
        if data_usage=='train':
            self.u0_real_train = data_mat['U0_real_train']
            self.u0_imag_train = data_mat['U0_imag_train']
            self.max_value = max_value
            self.min_value = min_value
            self.x_train = data_mat['x_train']
            self.y_train = data_mat['z_train']
            self.sx_train =data_mat['sx_train']
            self.m_train = data_mat['m_train']
            self.m0_train = data_mat['m0_train']
            self.f_train = data_mat['f_train']
            self.data = np.concatenate([self.x_train,self.y_train,self.sx_train,self.u0_real_train,self.u0_imag_train,self.m_train,self.m0_train,self.f_train],1)
            #self.data = np.concatenate([self.x_train,self.y_train,self.sx_train,self.u0_real_train,self.u0_imag_train,self.m_train,self.m0_train],1)
            #self.data[:,0:3] = 2.0 * (self.data[:,0:3]-self.min_value)/(self.max_value-self.min_value) - 1.0
        elif data_usage=='test':
            self.du_real_star = data_mat['dU_real_star']
            self.du_imag_star = data_mat['dU_imag_star']
            try:
                self.U0_real_star = data_mat['U0_real_star']
                self.U0_imag_star = data_mat['U0_imag_star']
            except:
                print('no information about background wavefield!!!')
            self.max_value = max_value
            self.min_value = min_value
            self.x_star = data_mat['x_star']
            self.y_star = data_mat['z_star']
            self.sx_star =data_mat['sx_star']
            self.f_star = np.ones_like(self.x_star)*freq_star
            self.data = np.concatenate([self.x_star,self.y_star,self.sx_star,self.du_real_star,self.du_imag_star],1)
            self.data[:,0:3] = 2.0 * (self.data[:,0:3]-self.min_value)/(self.max_value-self.min_value) - 1.0
        else:
            print('Wrong data usage!!!')

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

    def __len__(self):
        return len(self.data)

class data_prefetcher():
    def __init__(self,loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
            #self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            #self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            #self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return 