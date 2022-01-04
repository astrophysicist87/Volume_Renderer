import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm                                                                          
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import volume_renderer

chosen_colormap = cm.get_cmap('inferno', 256)

def theta(scale, location, x):
    if x < 1e-3:
        return 0.0
    elif x > 1.0-1e-3:
        return 1.0
    else:
        return 0.5*(1.0+np.tanh(a*(x-b)/(x*(1.0-x))))

def gaussianTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*np.exp( -(x - 1.0)**2/0.1 )
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

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*x
    return r,g,b,a

def main():
    # Load Datacube
    f = h5.File(sys.argv[1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['temperature'])
    
    
    # this is where the image array is produced
    TFO = 0.154 # freeze-out temperature in GeV
    image = volume_renderer.render_volume((x,y,z), datacube, (0.0, np.pi/4.0), N=250, \
                                          transferFunction=linearTransferFunction, cutoff=TFO)

    # Plot Volume Rendering
    plt.figure(figsize=(4,4), dpi=500)
    
    # z-axis in image points up by default
    # swap axes to get conventional heavy-ion orientation
    image = np.swapaxes(image, 0, 1)

    plt.imshow(image)
    plt.axis('off')
    
    # Save figure
    plt.savefig('volumerender0.png', dpi=500, bbox_inches='tight', pad_inches = 0)
        
    return 0
    


  
if __name__== "__main__":
  main()

