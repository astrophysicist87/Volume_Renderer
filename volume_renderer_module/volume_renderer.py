import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm                                                                          
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#chosen_colormap = cm.get_cmap('plasma', 256)
#chosen_colormap = cm.get_cmap('magma', 256)
chosen_colormap = cm.get_cmap('inferno', 256)

def enhance_channel(a, f, no_clip=False):
    '''
    f enhances RGB colors
    '''
    return a*f if no_clip else np.clip(a*f, 0.0, 1.0)

def defaultTransferFunction(x0, **kwargs):
    '''
    transferFunction takes normalized array of densities and converts
    each cell to a corresponding RGBA color.
    '''
    # x0 must be in range [0,1]
    # frac allows to rescale low-density colormap
    frac         = kwargs.get("frac")         if "frac"         in kwargs else 0.0
    num_contours = kwargs.get("num_contours") if "num_contours" in kwargs else 10
    max_opacity  = kwargs.get("max_opacity")  if "max_opacity"  in kwargs else 0.5

    x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
    r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
    delta = 1.0/num_contours
    a = max_opacity*np.exp( -(x - 1.0)**2/delta**2 )
    for center in np.linspace(0, 1, num_contours, endpoint=False):
        a += max_opacity*x**2*np.exp( -(x - center)**2/delta**4 )
    return r,g,b,a



def render_volume(points, datacube, angles, **kwargs):
    '''
    render_volume... renders... the volume...
    '''
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    
    print(datacube.shape)

    datacube += 1e-15
    minimum = np.amin(datacube)
    maximum = np.amax(datacube)
    logmin = np.amin(np.log(datacube))
    logmax = np.amax(np.log(datacube))

    phi, theta = angles
    N = kwargs.get("N") if "N" in kwargs else 180
    c = np.linspace(-20.0, 20.0, N)
    qx, qy, qz = np.meshgrid(c,c,c)
    qxR  = qx
    qyR  = qy * np.cos(theta) - qz * np.sin(theta) 
    qzR  = qz * np.cos(theta) + qy * np.sin(theta)
    qxRR = qxR * np.cos(phi) - qyR * np.sin(phi) 
    qyRR = qyR * np.cos(phi) + qxR * np.sin(phi) 
    qzRR = qzR
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    camera_grid = interpn(points, datacube, qi, method='linear',\
                                  bounds_error=False, fill_value=np.amin(datacube)\
                                 ).reshape((N,N,N))
    
    # Do Volume Rendering
    image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

    # allow user to pass custom transfer function
    transferFunction = kwargs.get("transferFunction") if "transferFunction" in kwargs \
                                                      else defaultTransferFunction
    
    # allow to use log of density to determine colors
    use_log_densities = kwargs.get("use_log_densities") if "use_log_densities" in kwargs \
                                                        else False
    
    if use_log_densities:
        for dataslice in camera_grid:
            log_dataslice = np.log(dataslice)
            normed_log_dataslice = (log_dataslice-logmin)/(logmax-logmin)
            r,g,b,a = transferFunction(normed_log_dataslice)
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
    else:
        for dataslice in camera_grid:
            normed_dataslice = (dataslice - minimum)/(maximum - minimum)
            r,g,b,a = transferFunction(normed_dataslice)
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
            
    print(image.shape)
        
    return np.clip(image, 0.0, 1.0).T

