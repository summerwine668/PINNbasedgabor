import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.encoding import get_embedder

class Basicblock(nn.Module):
    """
    The basic block for MLP
    """
    def __init__(self,in_planes,out_planes):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Linear(in_planes,out_planes)

    def forward(self, x):
        out = torch.sin(self.layer1(x))
        return out

def weight_init(m):
    """
    THe initilization for weights in the neural networks
    """
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias, 0)
    if classname.find('Conv')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias,0)
        
            
class GaborLayerd(nn.Module):
    """
    Gabor-like filter as used in GaborNet. but the center point d is learnable
    """

    def __init__(self, in_features, out_features, freq, last_layer_type, alpha=1.0, beta=1.0, **kwargs):
        super().__init__()
        self.learned_delta = kwargs['learned_delta']
        self.learned_gamma = kwargs['learned_gamma']
        self.learned_phi = kwargs['learned_phi']
        self.delta = nn.Parameter(
                torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,)) # default
            )
        self.phi = nn.Parameter(
                torch.distributions.Uniform(-np.pi, np.pi).sample((out_features,))
            )
        self.gamma = nn.Parameter(
            torch.distributions.Uniform(0.5, 1.0).sample((out_features,)) # default
        )
            
        self.omega = freq
        self.theta = nn.Parameter(torch.distributions.Uniform(-np.pi, np.pi).sample((out_features,)))
        self.learned_theta = kwargs['learned_theta']
        self.last_layer_type = last_layer_type
        self.activation_p = nn.Sigmoid()
        # self.activation_p = nn.Tanh()
        # return
    
    def soft_constraint(self, para, value):
        return (2.0 * self.activation_p(para) - 1.0) * value
        
    def transform_coordinates(self, x, theta):
        x_ = x[...,0:1] * torch.cos(theta) + x[...,1:2] * torch.sin(theta)
        # x_ = x[...,0:1] * torch.cos(self.soft_constraint(self.theta, torch.pi)) + x[...,1:2] * torch.sin(self.soft_constraint(self.theta, torch.pi))
        return x_
    
    def transform_coordinates_y(self, x, theta):
        y_ = -1.0 * x[...,0:1] * torch.sin(theta) + x[...,1:2] * torch.cos(theta)
        # y_ = -1.0 * x[...,0:1] * torch.sin(self.soft_constraint(self.theta, torch.pi)) + x[...,1:2] * torch.cos(self.soft_constraint(self.theta, torch.pi))
        return y_
        
    def forward(self, x, d, v, theta=None):
        theta = self.theta
        gamma = self.gamma
        phi = self.phi
        delta = self.delta[None,:]
        D = (self.transform_coordinates(x-d, theta) ** 2 + gamma ** 2 * self.transform_coordinates_y(x-d, theta)**2)
        if self.last_layer_type == 5:
            return torch.cos(self.omega * v * self.transform_coordinates(x - d, theta) + (2.0 * self.activation_p(phi)-1.0)* torch.pi) * torch.exp(-0.5 * D * delta**2), \
                   torch.sin(self.omega * v * self.transform_coordinates(x - d, theta) + (2.0 * self.activation_p(phi)-1.0)* torch.pi) * torch.exp(-0.5 * D * delta**2) # sin(freq / v * x + phi) * exp(-0.5 * D * gamma)
        else:
            print('unknown last layer type')
    

class GaborNetL(nn.Module):
    '''
    Gabor layer applied to the last layer of the PINN
      - direct sum the real and imaginary part of the gabor layer
      - add another branch for center point of the gabor layer
    '''
    
    def __init__(self, in_channels, out_channels, layers, **kwargs) -> None:
        super(GaborNetL, self).__init__()
        self.layers = layers
        self.last_layer_type = kwargs['last_layer_type']
        self.embedding_fn, self.input_cha = get_embedder(kwargs['encoding_config'])
        self.in_planes = self.input_cha
        self.activation_d = nn.Sigmoid()
        if self.last_layer_type == 5:
            self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)-1]).apply(weight_init)
            self.filter = GaborLayerd(in_channels, layers[-1], **kwargs)
            self.linear = nn.Linear(layers[-2], layers[-1])
            dim_d = kwargs['dim_d'] 
            self.lineard = nn.Sequential(
                nn.Linear(in_channels, dim_d),
                nn.GELU(),
                nn.Linear(dim_d, 2)
            )
        else:
            raise ValueError('unknown last layer type')
            
        
    def _make_layer(self, block, layers):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)
    
    def forward(self, x, v=1.0/1.5):
        # out = self.layer1(self.embedding_fn(x))
        out = torch.sin(self.layer1(self.embedding_fn(x)))
        if self.last_layer_type==5:
            out_real, out_imag = self.filter(x, (2.0 * self.activation_d(self.lineard(x)) - 1.0), v)
            out_real, out_imag = out_real * (self.linear(out)), out_imag * (self.linear(out))
            out = torch.cat([torch.sum(out_real, dim=1, keepdim=True), torch.sum(out_imag, dim=1, keepdim=True)], dim=1)
        return out
    
