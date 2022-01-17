import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py as h5
import sys
import volume_renderer
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

image_pixel_dimension = 50
maximum = 0.0
chosen_colormap = cm.get_cmap('inferno', 256)

def theta(scale, location, x):
    return np.piecewise(x, [x < 1e-3, x > 1.0-1e-3], \
                           [0.0, 1.0, lambda x: 0.5*(1.0+np.tanh(scale*(x-location)/(x*(1.0-x))))])

def gaussianTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*np.exp( -6.0*(x - 1.0)**2 )
    return r,g,b,a

def quadraticTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*x**2
    return r,g,b,a

def linearTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    #print("linearTransferFunction:",frac,max_opacity,cutoff,flush=True)

    frac = cutoff
    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    cutoff = np.clip(cutoff, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)  # maps cutoff --> 0
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*x
    return r,g,b,a


def constantTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    frac = cutoff
    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    cutoff = np.clip(cutoff, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*theta(100.0, cutoff, x)
    return r,g,b,a


def init():
    image = np.zeros((image_pixel_dimension, image_pixel_dimension))
    im = plt.imshow(image)
    plt.axis('off')
    return im,


def animate(i):
    global maximum
    print('Rendering Scene ' + str(i+1) + ' of ' + str(len(sys.argv[1:])), flush=True)

    # Load Datacube
    f = h5.File(sys.argv[i+1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['energy_density'])
    points = (x, y, z)
    
    if i==0:
        maximum = np.amax(datacube)
        
    print("Max:",i,maximum,flush=True)

    # this is where the image array is produced
    eFO = 0.266 # freeze-out temperature in GeV
    TFO = 0.154 # freeze-out temperature in GeV
    image = volume_renderer.render_volume(points, datacube, (0.0, np.pi/4.0), N=image_pixel_dimension, \
                                          transferFunction=linearTransferFunction, \
                                          scale_max=maximum, cutoff=eFO, use_log_densities=True)
    
    # z-axis in image points up by default
    # swap axes to get conventional heavy-ion orientation
    image = np.swapaxes(image, 0, 1)

    im = plt.imshow(image)
    plt.axis('off')
    plt.savefig('animation_frames/frame' + str(i) + '.png', dpi=500, bbox_inches='tight', pad_inches = 0)
    return im,



def main():

    # Plot Volume Rendering
    fig = plt.figure(figsize=(8,8), dpi=500)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
        
    # Do Volume Rendering at Different Viewing Angles
    ani = animation.FuncAnimation(fig, animate, np.arange(len(sys.argv[1:])), \
                                  init_func=init, blit=True)

    f = "animation_log_ed_complete.mp4"
    FFwriter = animation.FFMpegWriter(fps=40, extra_args=['-vcodec', 'libx264'])
    ani.save(f, writer=FFwriter)
    #f = "animation.gif" 
    #ani.save(f, writer='imagemagick', fps=10)

    return 0

  
if __name__== "__main__":
  main()


