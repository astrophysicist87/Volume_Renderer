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

def main():
    # Load Datacube
    f = h5.File(sys.argv[1], 'r')
    x = np.array(f['x'])
    y = np.array(f['y'])
    z = np.array(f['z'])
    datacube = np.array(f['temperature'])
    
    
    # this is where the image array is produced
    image = volume_renderer.render_volume((x,y,z), datacube, (0.0, np.pi/6.0), N=500)

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

