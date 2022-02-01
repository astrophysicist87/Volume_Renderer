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
from mpl_toolkits.axes_grid1 import make_axes_locatable

image_pixel_dimension = 500
chosen_colormap = cm.get_cmap('inferno', 256)

def theta(scale, location, x):
    return np.piecewise(x, [x < 1e-3, x > 1.0-1e-3], \
                           [0.0, 1.0, lambda x: 0.5*(1.0+np.tanh(scale*(x-location)/(x*(1.0-x))))])

def gaussianTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    frac = cutoff
    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    cutoff = np.clip(cutoff, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)  # maps cutoff --> 0
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




def main():

    # Plot Volume Rendering
    fig = plt.figure(figsize=(1,1), dpi=500)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
        
    # Load Datacube
    f = h5.File(sys.argv[1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['energy_density'])
    points = (x, y, z)
        
    maximum = sys.argv[2] if len(sys.argv) > 2 else np.amax(datacube)
        
    # this is where the image array is produced
    eFO = 0.266 # freeze-out temperature in GeV
    TFO = 0.154 # freeze-out temperature in GeV
    image = volume_renderer.render_volume(points, datacube, (0.0, np.pi/4.0), N=image_pixel_dimension, \
                                          transferFunction=linearTransferFunction, \
                                          max_opacity=0.5, scale_max=maximum, \
                                          cutoff=eFO, use_log_densities=True)

    image = np.swapaxes(image, 0, 1)

    im = plt.imshow(image)
    
    plt.axis('off')
    plt.imsave(fname='old_animation_frames/frame' + str(i) + '.png', \
               arr=image, cmap=chosen_colormap, format='png')

    return 0

  
if __name__== "__main__":
  main()




