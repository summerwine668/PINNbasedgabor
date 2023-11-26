from utils.encoding import get_embedder
from utils.utils import *
from utils.vis import plot
from utils.TransferData2Dataset import DataProcessing
from model_zoo import build_model
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import scipy.io as sio
import matplotlib.pyplot as plt


def train(cfg, net, embedding_fn, train_data):
    device = torch.device(cfg.gabor2d.device)
    seed_torch(cfg.gabor2d.seed)
    
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.gabor2d.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.gabor2d.step_size, gamma=cfg.gabor2d.gamma)
    min_loss = 9e20
    sw = SummaryWriter(os.path.join(cfg.tensorboard_path, str(cfg.gabor2d.fre))+f'layer_type{cfg.gabor2d.last_layer_type}_gabor_width{cfg.gabor2d.hidden_layers[-1]}')
    
    ub, lb = torch.Tensor([cfg.gabor2d.ub]).to(device), torch.Tensor([cfg.gabor2d.lb]).to(device)
    fdm_loss_pde = helmholtz_loss_pde(0.001,0.001,ub, lb, cfg.gabor2d.regularization, v_background=cfg.gabor2d.regular_v, device=device)
    
    for epoch in range(cfg.gabor2d.epochs):
        epoch_loss, pde_loss, reg_loss = 0, 0, 0
        input_data = torch.Tensor(train_data.data[:,:])
        randperm = np.random.permutation(len(input_data))
        batch_size = int(len(input_data)/cfg.gabor2d.n_batches)
        if cfg.gabor2d.plot_pde_residual and (epoch % cfg.gabor2d.test_every==0):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for batch_idx in range(cfg.gabor2d.n_batches):
            start_,end_ = batch_idx*batch_size,(batch_idx+1)*batch_size
            randperm_idx = randperm[start_:end_]
            x_train, y_train, sx_train, u0_real_train, u0_imag_train, m_train, m0_train = input_data[randperm_idx,0:1].to(device),input_data[randperm_idx,1:2].to(device),input_data[randperm_idx,2:3].to(device),input_data[randperm_idx,3:4].to(device),input_data[randperm_idx,4:5].to(device),input_data[randperm_idx,5:6].to(device),input_data[randperm_idx,6:7].to(device)
            
            optimizer.zero_grad()
            x = x_train.clone().detach().requires_grad_(True)
            y = y_train.clone().detach().requires_grad_(True)
            sx = sx_train.clone().detach().requires_grad_(True)
            omega = 2 * torch.pi * cfg.gabor2d.fre
            x_input = torch.cat((x, y), 1)
            x_input = normalizer(x_input, lb, ub)
            if cfg.gabor2d.model_type == 'gaborl':
                du_pred_out = net(x_input, torch.sqrt(m_train))
            else:
                x_input = embedding_fn(x_input)
                du_pred_out = net(x_input)
            if cfg.gabor2d.regularization != 'None':
                loss, pde_loss, reg_loss, loss_pde_real, _ = fdm_loss_pde(x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train,du_pred_out, net, embedding_fn, derivate_type=cfg.gabor2d.derivate_type)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().item()
                pde_loss += pde_loss.detach().cpu().item()
                reg_loss += reg_loss.detach().cpu().item()
            else:
                loss,loss_pde_real, _ = fdm_loss_pde(x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train,du_pred_out, net, embedding_fn, derivate_type=cfg.gabor2d.derivate_type)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().item()
            if cfg.gabor2d.plot_pde_residual and epoch % cfg.gabor2d.test_every == 0:
                sc = ax.scatter(x.detach().cpu().numpy(),y.detach().cpu().numpy(),sx.detach().cpu().numpy(),c=loss_pde_real.detach().cpu().numpy(),cmap='coolwarm', alpha=0.8) 
        if cfg.gabor2d.plot_pde_residual and (epoch % cfg.gabor2d.test_every==0):
            ax.set_xlabel('X-axis')
            ax.set_ylim(0.025,cfg.gabor2d.axisz)
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Sx-axis')       
            plt.colorbar(sc)
            sw.add_figure(f'loss_map/real{cfg.gabor2d.fre}', fig, epoch) 
        scheduler.step()
        sw.add_scalar('train/loss', epoch_loss, epoch)
        sw.add_scalar('train/pdeloss', pde_loss, epoch)
        sw.add_scalar('train/regloss', reg_loss, epoch)
        if epoch % cfg.gabor2d.print_loss_every == 0:
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                if not os.path.exists(cfg.checkpoint_path):
                    os.makedirs(cfg.checkpoint_path)
                else:
                    torch.save({'net':net.state_dict()}, os.path.join(cfg.checkpoint_path, 'best_net.pth'))
            print(f'Epoch: {epoch}; Loss: {epoch_loss}; Equ Loss: {pde_loss}; Reg Loss: {reg_loss}')
        
        if epoch !=0 and epoch % cfg.gabor2d.save_model_every == 0:
            torch.save({'net':net.state_dict()}, os.path.join(cfg.checkpoint_path, f'net_{epoch}.pth'))
            
        if epoch !=0 and epoch % cfg.gabor2d.test_every == 0:
            if cfg.gabor2d.test_file is not None:
                du_real_eval, du_imag_eval, error_real, error_imag = test_ent(cfg, True, out_vis=True)
                sw.add_scalar('test/accuracy for real parts', error_real, epoch)
                sw.add_scalar('test/accuracy for imag parts', error_imag, epoch)
            else:
                du_real_eval, du_imag_eval = test_ent(cfg, True, out_vis=True)
            sw.add_figure(f'Wavefield/real{cfg.gabor2d.fre}',plot(du_real_eval, cfg.gabor2d.vmin, cfg.gabor2d.vmax, cfg.gabor2d.axisx, cfg.gabor2d.axisz),epoch)
            sw.add_figure(f'Wavefield/imag{cfg.gabor2d.fre}',plot(du_imag_eval, cfg.gabor2d.vmin, cfg.gabor2d.vmax,cfg.gabor2d.axisx, cfg.gabor2d.axisz),epoch)
  
def test(net, embedding_fn, out_vis, cfg):
    device = torch.device(cfg.gabor2d.device)
    nx, nz = cfg.gabor2d.nx, cfg.gabor2d.nz
    if cfg.gabor2d.test_file is None:
        x = 2.0 * (np.arange(nx).reshape(nx,1).repeat(nz,axis=0).reshape(nx*nz,1)*0.025 - 0.0) / ((nx-1)*0.025 - 0.0) - 1.0
        y = 2.0 * (np.arange(nz).reshape(nz,1).repeat(nx,axis=1).T.reshape(nx*nz,1)*0.025 - 0.0) / ((nz-1)*0.025 - 0.0) - 1.0
        x_input = torch.cat(((torch.Tensor(x)).to(device),(torch.Tensor(y)).to(device)),1)
    else:
        test_data=DataProcessing(cfg.gabor2d.data_root_path, cfg.gabor2d.test_file,
                                 'test',np.array([cfg.gabor2d.axisx,cfg.gabor2d.axisz,cfg.gabor2d.axisx]),np.array([0.0,0.0,0.0]),freq_star=cfg.gabor2d.fre)
        test_indexs = np.arange(nx * nz) + cfg.gabor2d.test_source * nx * nz
        x_input = torch.Tensor(test_data.data[test_indexs[:None],0:2]).to(device)
    if cfg.gabor2d.model_type == 'gaborl':
        m = sio.loadmat(cfg.gabor2d.data_root_path + cfg.gabor2d.velocity_file)
        m = m[list(m.keys())[-1]].T.reshape(-1,1)
        m = torch.Tensor(1000.0/m).to(torch.float32).to(device)
        du_pred_out = net(x_input.to(device), m)
    else:
        x_input = embedding_fn(x_input)
        du_pred_out = net(x_input.to(device))
    du_real_eval, du_imag_eval = du_pred_out[:,0:1], du_pred_out[:,1:2]
    if cfg.task_type == 'test':
        real_err_fig = plot((du_real_eval.detach().cpu().numpy().reshape(nx, nz)-test_data.data[test_indexs[:None],3:4].reshape(nx,nz)), cfg.gabor2d.vmin, cfg.gabor2d.vmax, cfg.gabor2d.axisx, cfg.gabor2d.axisz)
        imag_err_fig = plot((du_imag_eval.detach().cpu().numpy().reshape(nx, nz)-test_data.data[test_indexs[:None],4:5].reshape(nx,nz)), cfg.gabor2d.vmin, cfg.gabor2d.vmax, cfg.gabor2d.axisx, cfg.gabor2d.axisz)
        torch.save({'true':test_data.data[test_indexs[:None],3:4].reshape(nx,nz),'pred':du_real_eval.detach().cpu().numpy().reshape(nx,nz)},f'test_{cfg.net_name}_real_{cfg.gabor2d.fre}_state{cfg.gabor2d.test_file}.pt')
        torch.save({'true':test_data.data[test_indexs[:None],4:5].reshape(nx,nz),'pred':du_imag_eval.detach().cpu().numpy().reshape(nx,nz)},f'test_{cfg.net_name}_imag_{cfg.gabor2d.fre}_state{cfg.gabor2d.test_file}.pt')
        real_err_fig.savefig(f'test_{cfg.net_name}_real_err_{cfg.gabor2d.fre}_state{cfg.gabor2d.test_file}.png', bbox_inches='tight', dpi=300)
        imag_err_fig.savefig(f'test_{cfg.net_name}_imag_err_{cfg.gabor2d.fre}_state{cfg.gabor2d.test_file}.png', bbox_inches='tight', dpi=300)
    if not out_vis:
        print('test done, current no ground truth')
    else:
        if cfg.gabor2d.test_file is not None:
            error_real, error_imag = error_l2(du_real_eval.detach().cpu().numpy(),du_imag_eval.detach().cpu().numpy(), test_data.data[test_indexs[:None],3:4], test_data.data[test_indexs[:None],4:5])
            return du_real_eval.detach().cpu().numpy().reshape(nx,nz), du_imag_eval.detach().cpu().numpy().reshape(nx,nz), error_real, error_imag
        else:
            return du_real_eval.detach().cpu().numpy().reshape(nx,nz), du_imag_eval.detach().cpu().numpy().reshape(nx,nz)

def train_ent(cfg):
    embedding_fn, input_cha = get_embedder(cfg.gabor2d.encoding_config)
    if cfg.gabor2d.model_type == 'pinn':
        net = build_model(cfg.gabor2d.model_type, cfg.net_name, input_cha, cfg.gabor2d.out_channels, hidden_layers=cfg.gabor2d.hidden_layers, input_scale=cfg.gabor2d.scale, n_layers=len(cfg.gabor2d.hidden_layers))
        macs, params = get_model_complexity_info(net, (input_cha,), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    elif cfg.gabor2d.model_type == 'gaborl':
        net = build_model(cfg.gabor2d.model_type, cfg.net_name, 2, cfg.gabor2d.out_channels, hidden_layers=cfg.gabor2d.hidden_layers, freq=cfg.gabor2d.fre, alpha=cfg.gabor2d.alpha, beta=cfg.gabor2d.beta, last_layer_type=cfg.gabor2d.last_layer_type, encoding_config=cfg.gabor2d.encoding_config, dim_d=cfg.gabor2d.dim_d, learned_theta=cfg.gabor2d.learned_theta, dim_theta=cfg.gabor2d.dim_theta, learned_delta=cfg.gabor2d.learned_delta,dim_delta=cfg.gabor2d.dim_delta,learned_gamma=cfg.gabor2d.learned_gamma,dim_gamma=cfg.gabor2d.dim_gamma,learned_phi=cfg.gabor2d.learned_phi,dim_phi=cfg.gabor2d.dim_phi)
    
        macs, params = get_model_complexity_info(net, (2,), as_strings=True, print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
    ub, lb = np.array([cfg.gabor2d.ub]), np.array([cfg.gabor2d.lb])
    if cfg.gabor2d.train_file != None:
        train_data_path = cfg.gabor2d.train_file
    else:
        train_data_path = '{}_{}Hz_train_data.mat'.format(cfg.gabor2d.model_name,cfg.gabor2d.fre)
    print(f'{train_data_path}')
    train_data = DataProcessing(cfg.gabor2d.data_root_path, train_data_path,'train',ub,lb)
    train(cfg, net, embedding_fn, train_data)

def test_ent(cfg, is_train=False, out_vis=False):
    embedding_fn, input_cha = get_embedder(cfg.gabor2d.encoding_config)
    if cfg.gabor2d.model_type == 'pinn':
        net = build_model(cfg.gabor2d.model_type, cfg.net_name, input_cha, cfg.gabor2d.out_channels, hidden_layers=cfg.gabor2d.hidden_layers, input_scale=cfg.gabor2d.scale, n_layers=len(cfg.gabor2d.hidden_layers))
    elif cfg.gabor2d.model_type == 'gaborl':
        net = build_model(cfg.gabor2d.model_type, cfg.net_name, 2, cfg.gabor2d.out_channels, hidden_layers=cfg.gabor2d.hidden_layers, freq=cfg.gabor2d.fre, alpha=cfg.gabor2d.alpha, beta=cfg.gabor2d.beta, last_layer_type=cfg.gabor2d.last_layer_type, encoding_config=cfg.gabor2d.encoding_config, dim_d=cfg.gabor2d.dim_d, learned_theta=cfg.gabor2d.learned_theta, dim_theta=cfg.gabor2d.dim_theta, learned_delta=cfg.gabor2d.learned_delta,dim_delta=cfg.gabor2d.dim_delta,learned_gamma=cfg.gabor2d.learned_gamma,dim_gamma=cfg.gabor2d.dim_gamma,learned_phi=cfg.gabor2d.learned_phi,dim_phi=cfg.gabor2d.dim_phi)
    
    if is_train:
        state = torch.load(os.path.join(cfg.checkpoint_path, 'best_net.pth'))
    else:
        state = torch.load(cfg.gabor2d.state_dict_file)
    state_dict = state['net']
    net.load_state_dict(state_dict)
    net.to(cfg.gabor2d.device)
    if not out_vis:
        test(net, embedding_fn, out_vis, cfg)
    else:
        return test(net, embedding_fn, out_vis, cfg)
        
def gaborlpinn_ent(cfg):
    if cfg.task_type == 'train':
        train_ent(cfg)
    else:
        test_ent(cfg)
