import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys, shutil, random, bisect
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.use('Agg')


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
    
def plot_results(du_real_pred,du_imag_pred,du_real_star, du_imag_star, epoch,source_number,freq,nx,nz,fre,vmin,vmax,axisx,axisz,model_name):
    # Error
    error_du_real = np.linalg.norm(du_real_star-du_real_pred,2)/np.linalg.norm(du_real_star,2)
    error_du_imag = np.linalg.norm(du_imag_star-du_imag_pred,2)/np.linalg.norm(du_imag_star,2)

    print('Error u_real: %e, Error u_imag: %e' % (error_du_real,error_du_imag))
    scipy.io.savemat('du_real_pred_atan-{}.mat'.format(fre),{'du_real_pred':du_real_pred})
    scipy.io.savemat('du_imag_pred_atan-{}.mat'.format(fre),{'du_imag_pred':du_imag_pred})

    scipy.io.savemat('du_real_star-{}.mat'.format(fre),{'du_real_star':du_real_star})
    scipy.io.savemat('du_imag_star-{}.mat'.format(fre),{'du_imag_star':du_imag_star})

    ## plot the real parts of the scattered wavefield for i th source
    #source_number = 8 ## 1-9
    a = (source_number-1)*nx*nz
    b = (source_number)*nx*nz
    du_real_star_is = du_imag_star[a:b:1]
    du_real_pred_is = du_imag_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_imag: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-imag.png')

    du_real_star_is = du_real_star[a:b:1]
    du_real_pred_is = du_real_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_real: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=vmin,vmax=vmax,extent=[0, axisx, axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-real.png')
    
def plot(d,vmin,vmax,axisx, axisz):
    fig = plt.figure(figsize=(axisx,axisz))
    ax = plt.gca()
    im = ax.imshow(d.T,vmin=vmin, vmax=vmax, extent=[0,axisx,axisz,0.0],aspect=1, cmap='jet')
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    return fig
