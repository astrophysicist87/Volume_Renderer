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

maximum = 0.0
chosen_colormap = cm.get_cmap('inferno', 256)

def theta(scale, location, x):
    #if x < 1e-3:
    #    return 0.0
    #elif x > 1.0-1e-3:
    #    return 1.0
    #else:
    #    return 0.5*(1.0+np.tanh(a*(x-b)/(x*(1.0-x))))
    return np.piecewise(x, [x < 1e-3, x > 1.0-1e-3], \
                           [0.0, 1.0, lambda x: 0.5*(1.0+np.tanh(scale*(x-location)/(x*(1.0-x))))])

def gaussianTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*np.exp( -6.0*(x - 1.0)**2 )*theta(25.0, cutoff, x)
    return r,g,b,a

def quadraticTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*x**2*theta(25.0, cutoff, x)
    return r,g,b,a

def linearTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    a = max_opacity*x*theta(25.0, cutoff, x)
    return r,g,b,a



def animate(i):
    global maximum
    print('Rendering Scene ' + str(i+1) + ' of ' + str(len(sys.argv[1:])), flush=True)

    # Load Datacube
    f = h5.File(sys.argv[i+1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['temperature'])
    points = (x, y, z)
    
    if i==0:
        maximum = np.amax(datacube)

    # this is where the image array is produced
    TFO = 0.154 # freeze-out temperature in GeV
    image = volume_renderer.render_volume(points, datacube, (0.0, np.pi/4.0), N=250, \
                                          transferFunction=quadraticTransferFunction,
                                          scale_max=maximum, cutoff=TFO)
        
    # z-axis in image points up by default
    # swap axes to get conventional heavy-ion orientation
    image = np.swapaxes(image, 0, 1)

    plt.imshow(image)
    plt.axis('off')



def main():

    # Plot Volume Rendering
    fig = plt.figure(figsize=(8,8), dpi=500)
        
    # Do Volume Rendering at Different Viewing Angles
    ani = animation.FuncAnimation(fig, animate, np.arange(len(sys.argv[1:])))

    f = "animation.mp4"
    FFwriter = animation.FFMpegWriter(fps=6, extra_args=['-vcodec', 'libx264'])
    ani.save(f, writer=FFwriter)
    #f = "animation.gif" 
    #ani.save(f, writer='imagemagick', fps=20)

    return 0

  
if __name__== "__main__":
  main()


