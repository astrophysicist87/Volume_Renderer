import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import volume_renderer_module.volume_renderer as volume_renderer

chosen_colormap = cm.get_cmap('RdBu', 256)

#colors1 = plt.cm.PuRd(np.linspace(0, 1, 128))
#colors2 = plt.cm.BuPu(np.linspace(0, 1, 128))
#
#colors = np.vstack((colors1, colors2))
#chosen_colormap = LinearSegmentedColormap.from_list('my_colormap', colors)



def constantTransferFunction(x0, **kwargs):
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5
    cutoff       = kwargs.get("cutoff")       if "cutoff"       in kwargs else 0.0

    #print(np.amin(x0),np.amax(x0))
    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(1.0-x)), axes=[2,0,1])
    a = max_opacity*np.abs(x-0.5)  #*np.heaviside(x-0.2,0.5)
    return r,g,b,a

def nucleon(x, y, z, r0):
    arg = 0.5*((x-r0[0])**2+(y-r0[1])**2+(z-r0[2])**2)/0.5**2
    cookie_cutter = np.heaviside(2.0-arg**2,0.0)
    return -np.sign(r0[2])*cookie_cutter*np.exp(-arg**2), cookie_cutter

def distance_to_plane(normal, displacement):
    return np.abs(normal @ displacement)/np.sqrt(normal.dot(normal))

def main():
    # Set grid.
    x = np.linspace(-10,10,101)
    y = np.linspace(-10,10,101)
    z = np.linspace(-10,10,101)
    
    X,Y,Z = np.meshgrid(x,y,z)

    #nucA = np.loadtxt("C:\\Users\\chris\\Desktop\\Research\\UIUC\\Volume_Renderer"\
    #                  "\\pre_collision_frames\\nucleusA.dat")
    #nucB = np.loadtxt("C:\\Users\\chris\\Desktop\\Research\\UIUC\\Volume_Renderer"\
    #                  "\\pre_collision_frames\\nucleusB.dat")
    nucA = np.loadtxt(sys.argv[1])
    nucB = np.loadtxt(sys.argv[2])
    
    t0 = np.abs(float(sys.argv[3]))
    dt = float(sys.argv[4])
    tStep = int(sys.argv[5])
    
    # ignore z positions for now; project to pancakes at fixed z-coordinate
    zCoord = np.max((t0 - tStep*dt,0.0))
    nucA[:,2] = -zCoord
    nucB[:,2] = zCoord
    
    #all_nucleons = nucA
    #all_nucleons = nucB
    all_nucleons = np.concatenate((nucA,nucB))
        
    camera_angle = (0.0, np.pi/4.0)
    normal = np.array([np.sin(camera_angle[1])*np.cos(camera_angle[0]),
                       np.sin(camera_angle[1])*np.sin(camera_angle[0]),
                       np.cos(camera_angle[1])])
    camera_position = 50.0*normal # say
    
    distances = np.array([distance_to_plane(normal, point-camera_position)
                          for point in all_nucleons])
    #print(distances)
    all_nucleons = all_nucleons[distances.argsort()[::-1]]

    datacube, cookie_cutter = nucleon(X, Y, Z, all_nucleons[0])
    #print(np.amin(datacube), np.amax(datacube))
    #print(1/0)
    for center in all_nucleons[1:]:
        density, cookie_cutter = nucleon(X, Y, Z, center)
        #print(np.amin(density), np.amax(density),\
        #      np.amin(cookie_cutter), np.amax(cookie_cutter),\
        #      np.amin(datacube), np.amax(datacube))
        datacube = (1.0-cookie_cutter)*datacube + density
        # sign of density (and therefore color) based on sign of z-coordinate
    #datacube = nucleon(X,Y,Z,np.array([0.0,0.0,0.0]))
    #print(datacube)
        
    #print(np.amin(datacube))    
    #print(np.amax(datacube))    
    #print(1/0)
    
    
    # this is where the image array is produced
    image = volume_renderer.render_volume((x,y,z), datacube, camera_angle, N=500, \
                                          transferFunction=constantTransferFunction,
                                          max_opacity=0.5, scale_min=-1.0, scale_max=1.0,
                                          fill_value=0.0)

    # Plot Volume Rendering
    plt.figure(figsize=(2,2), dpi=500)
    
    # z-axis in image points up by default
    # swap axes to get conventional heavy-ion orientation
    image = np.swapaxes(image, 0, 1)

    ax = plt.gca()

    im = ax.imshow(image, cmap=chosen_colormap)
    plt.axis('off')
    
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)
    
    # Save figure
    #plt.savefig('all_frames/pre_collision_frames/frame_'+str(tStep)+'.png',\
    #            dpi=500, bbox_inches='tight', pad_inches = 0)
    plt.imsave(fname='all_frames/pre_collision_frames/frame_'+str(tStep)+'.png', \
               arr=image, cmap=chosen_colormap, format='png')
    #plt.show()
        
    return 0
    


  
if __name__== "__main__":
  main()


