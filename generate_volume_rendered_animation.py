import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

Nangles  = 10
datacube = None
points   = None

def animate(i):
    print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles), flush=True)

    # this is where the image array is produced
    image = volume_renderer.render_volume(points, datacube, (0.0, 2.0*i*np.pi/Nangles))
        
    # z-axis in image points up by default
    # swap axes to get conventional heavy-ion orientation
    image = np.swapaxes(image, 0, 1)

    plt.imshow(image)
    plt.axis('off')



def main():
    global datacube
    global points

    # Load Datacube
    f = h5.File(sys.argv[1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['temperature'])
    points = (x, y, z)

    # Plot Volume Rendering
    fig = plt.figure(figsize=(4,4), dpi=80)
        
    # Do Volume Rendering at Different Viewing Angles
    ani = animation.FuncAnimation(fig, animate, np.arange(Nangles))

    f = "animation.gif" 
    ani.save(f, writer='imagemagick', fps=10)

    return 0

  
if __name__== "__main__":
  main()

