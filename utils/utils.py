import torch
from torch.autograd import grad
import random, os
import numpy as np
import torch.nn.functional as F
import math


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def gradient(y,x,grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad_ = grad(y,[x],grad_outputs=grad_outputs,create_graph=True)[0]
    return grad_

def divergence(y,x):
    div = 0
    for i in range(y.shape[-1]):
        div += grad(y[...,i],x,torch.ones_like(y[...,i]),create_graph=True)[0][...,i:i+1]
    return div

def laplace(y,x):
    grad_ = gradient(y,x)
    return divergence(grad_,x)

def normalizer(x, lb, ub):
    return 2.0 * (x - lb) / (ub - lb) - 1.0

class helmholtz_loss_pde(torch.nn.Module):
    def __init__(self,  dx, dy, lb, ub, regularized=1.0,causal=False,pde_loss_penelty=0.0, v_background=1.5, device='cuda:0'):
        super(helmholtz_loss_pde, self).__init__()
        self.dx = dx
        self.dy = dy
        self.ub = ub
        self.lb = lb
        self.device = device
        self.regularized = regularized
        self.pde_loss_penelty = pde_loss_penelty
        self.causal = causal
        self.v_background = v_background / 2.0
    
    def regloss(self, x, y, sx, f, du_pred_out):
        factor_d = F.relu((self.v_background * 3.14/f)**2-(sx-x)**2-(y-0.025)**2)*10e7*f
        loss_reg = torch.sum(factor_d*torch.pow(du_pred_out[:,0:1],2)) +  torch.sum(factor_d*torch.pow(du_pred_out[:,1:2],2))
        return loss_reg / (x.shape[0] * 2)
    
    def _pde_res_penelty(self, x, y, pde_loss):
        return torch.sum(torch.pow(gradient(pde_loss, x), 2)) + torch.sum(torch.pow(gradient(pde_loss, y) , 2)) 
    
    def _laplace(self, u_left, u_right, u , u_top, u_down):
        return (u_left + u_right - 2.0 * u) / (self.dx**2) + (u_top + u_down - 2.0 * u) / (self.dy**2)
    
    def _query(self, x, y, sx, net, embedding_fn):
        x_input = torch.cat((x,y,sx),1)
        x_input = 2.0 * (x_input - self.lb) / (self.ub - self.lb) - 1.0
        return net(embedding_fn(x_input))
    
    def causal_loss(self, f_real_pred, f_imag_pred, x, y, sx, interval=0.025, max_r_step=142, epsilon=1e-7):
        loss_temp = 0.0
        lr = ((torch.pow(f_real_pred, 2)) + (torch.pow(f_imag_pred, 2)))
        distance_s_r = (sx-x)**2 + (y- 0.025) **2
        loss = torch.sum(lr[ distance_s_r <= interval**2])
        for iz in range(1,max_r_step):
            loss += math.exp(-1.0*epsilon*loss_temp) * torch.sum(lr[((distance_s_r<=((iz+1)*interval)**2) & (distance_s_r>(iz*interval)**2))])
            loss_temp  +=  torch.sum(lr[((distance_s_r<=((iz+1)*interval)**2) & (distance_s_r>(iz*interval)**2))]).item()
        loss = 1.0/max_r_step *loss
        return loss, loss_temp
        
    def forward(self, x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train, du_pred_out, net, embedding_fn, derivate_type='ad', epsilon=1e-7):
        if derivate_type == 'fd':
            with torch.no_grad():
                du_pred_out_left = self._query(x - self.dx, y, sx, net, embedding_fn)
                du_pred_out_right = self._query(x + self.dx, y, sx, net, embedding_fn)
                du_pred_out_top = self._query(x, y-self.dy, sx, net, embedding_fn)
                du_pred_out_down = self._query(x, y+self.dy, sx, net, embedding_fn)
            du_laplace_real = self._laplace(du_pred_out_left[:,0:1], du_pred_out_right[:,0:1], du_pred_out[:,0:1], du_pred_out_top[:,0:1], du_pred_out_down[:,0:1])
            du_laplace_imag = self._laplace(du_pred_out_left[:,1:2], du_pred_out_right[:,1:2], du_pred_out[:,1:2], du_pred_out_top[:,1:2], du_pred_out_down[:,1:2])
        elif derivate_type == 'ad':
            du_real_xx = laplace(du_pred_out[:,0:1], x)
            du_imag_xx = laplace(du_pred_out[:,1:2], x)
            du_real_yy = laplace(du_pred_out[:,0:1], y)
            du_imag_yy = laplace(du_pred_out[:,1:2], y)
            du_laplace_real = du_real_xx + du_real_yy
            du_laplace_imag = du_imag_xx + du_imag_yy 
        f_real_pred = omega**2 * m_train*du_pred_out[:,0:1] + du_laplace_real + omega**2 * (m_train-m0_train) * u0_real_train
        f_imag_pred = omega**2 * m_train*du_pred_out[:,1:2] + du_laplace_imag + omega**2 * (m_train-m0_train) * u0_imag_train
        if self.pde_loss_penelty > 0:
            pde_loss_pen = self._pde_res_penelty(x, y, f_real_pred) + self._pde_res_penelty(x, y, f_imag_pred)
        else:
            pde_loss_pen = 0.0
        if self.regularized != 'None':
            loss_reg = self.regloss(x, y, sx, omega, du_pred_out)
            loss_pde =  (torch.sum(torch.pow(f_real_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))) / (x.shape[0] * 2) 
            return loss_pde + self.regularized * loss_reg + self.pde_loss_penelty * pde_loss_pen , loss_pde, loss_reg, f_real_pred, f_imag_pred
        elif self.causal:
            loss_pde, loss_pde_wo_weight = self.causal_loss(f_real_pred, f_imag_pred, x, y, sx, interval=0.025, max_r_step=142, epsilon=epsilon) 
            return loss_pde, loss_pde_wo_weight/(x.shape[0]*2), f_real_pred, f_imag_pred
        else:
            return (torch.sum(torch.pow(f_real_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))) / (x.shape[0] * 2) + self.pde_loss_penelty * pde_loss_pen, f_real_pred, f_imag_pred

def error_l2(du_real, du_imag, du_real_ref, du_imag_ref):
    error_du_real = np.linalg.norm(du_real-du_real_ref,2)/np.linalg.norm(du_real_ref,2)
    error_du_imag = np.linalg.norm(du_imag-du_imag_ref,2)/np.linalg.norm(du_imag_ref,2)
    return error_du_real, error_du_imag
