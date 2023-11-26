import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encoding import get_embedder
from utils.utils import *

class Basicblock(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Linear(in_planes,out_planes)

    def forward(self, x):
        out = torch.sin(self.layer1(x))
        return out

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias, 0)
    if classname.find('Conv')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias,0)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(MLP, self).__init__()
        self.layers = layers
        self.in_planes = in_channels
        self.layer1 = self._make_layer(Basicblock,self.layers[:len(layers)])
        self.linear = nn.Linear(layers[-1],out_channels)

    def _make_layer(self, block, layers):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self,x):
        out = self.layer1(x)
        out = self.linear(out)
        return out

class PINNmodel(nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, hidden_layers, **kwargs) -> None:
        super(PINNmodel, self).__init__()
        if layer_type == 'mlp':
            self.model = MLP(in_channels, out_channels, hidden_layers).apply(weight_init) # default initiliazation
            # self.model = MLP(in_channels, out_channels, hidden_layers)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.model(x)
    