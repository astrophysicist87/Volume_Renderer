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
import matplotlib.image as mpimg

image_pixel_dimension = 501
chosen_colormap = cm.get_cmap('inferno', 256)

def init():
    image = np.zeros((image_pixel_dimension, image_pixel_dimension))
    im = plt.imshow(image)
    plt.axis('off')
    return im,


def animate(i):
    print('Rendering Scene ' + str(i+1) + ' of ' + str(len(sys.argv[1:])), flush=True)
    im = plt.imshow(mpimg.imread(sys.argv[i+1]))
    plt.axis('off')
    return im,



def main():

    # Plot Volume Rendering
    fig = plt.figure(figsize=(1,1), dpi=500)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
        
    # Do Volume Rendering at Different Viewing Angles
    ani = animation.FuncAnimation(fig, animate, np.arange(len(sys.argv[1:])), \
                                  init_func=init, blit=True)

    f = "complete_movie.mp4"
    FFwriter = animation.FFMpegWriter(fps=40, extra_args=['-vcodec', 'libx264'])
    ani.save(f, writer=FFwriter)
    #f = "animation.gif" 
    #ani.save(f, writer='imagemagick', fps=10)

    return 0

  
if __name__== "__main__":
  main()


