import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm                                                                          
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#chosen_colormap = cm.get_cmap('plasma', 256)
#chosen_colormap = cm.get_cmap('magma', 256)
chosen_colormap = cm.get_cmap('inferno', 256)

def theta(scale, location, x):
    if x < 1e-3:
        return 0.0
    elif x > 1.0-1e-3:
        return 1.0
    else:
        return 0.5*(1.0+np.tanh(a*(x-b)/(x*(1.0-x))))

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
    
    datacube += 1e-15

    cutoff      = kwargs.get("cutoff")      if "cutoff"      in kwargs else 0.0
    fill_value  = kwargs.get("fill_value")  if "fill_value"  in kwargs else np.amin(datacube)
    max_opacity = kwargs.get("max_opacity") if "max_opacity" in kwargs else 0.5
    
    # allow user to pass custom transfer function
    transferFunction = kwargs.get("transferFunction") if "transferFunction" in kwargs \
                                                      else defaultTransferFunction
    
    # allow to use log of density to determine colors
    use_log_densities = kwargs.get("use_log_densities") if "use_log_densities" in kwargs \
                                                        else False

    phi, theta = angles
    N = kwargs.get("N") if "N" in kwargs else 180
    upsample_factor = 1
    newN = upsample_factor*N+1 if upsample_factor > 1 else N
    c = np.linspace(-20.0, 20.0, newN)
    #c = np.sort(np.unique(np.concatenate((np.linspace(-20,20,newN),np.linspace(-1,1,newN)))))
    print(c,len(c))
    qx, qy, qz = np.meshgrid(c,c,c)
    qxR  = qx
    qyR  = qy * np.cos(theta) - qz * np.sin(theta) 
    qzR  = qz * np.cos(theta) + qy * np.sin(theta)
    qxRR = qxR * np.cos(phi) - qyR * np.sin(phi) 
    qyRR = qyR * np.cos(phi) + qxR * np.sin(phi) 
    qzRR = qzR
    qi = np.array([qxR[:,:-1:upsample_factor,:-1:upsample_factor].ravel(), \
                   qyR[:,:-1:upsample_factor,:-1:upsample_factor].ravel(), \
                   qzR[:,:-1:upsample_factor,:-1:upsample_factor].ravel()]).T \
         if upsample_factor > 1 else np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    #camera_grid = np.zeros((N,N,N))
    #if use_log_densities:
    #    camera_grid = np.exp(interpn(points, np.log(datacube), qi, method='linear',\
    #                                  bounds_error=False, fill_value=fill_value\
    #                                 )).reshape((N,N,N))
    #else:
    #    camera_grid = interpn(points, datacube, qi, method='linear',\
    #                                  bounds_error=False, fill_value=fill_value\
    #                                 ).reshape((N,N,N))
    
    #print('Image center (before)',flush=True)
    
    #camera_grid = interpn(points, datacube, qi, method='linear',\
    #                      bounds_error=False, fill_value=fill_value\
    #                     ).reshape((newN,N,N))
    cN = len(c)
    camera_grid = interpn(points, datacube, qi, method='linear',\
                          bounds_error=False, fill_value=fill_value\
                         ).reshape((cN,cN,cN))
    
    #print('Image center (after)',flush=True)


    # Do Volume Rendering
    image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
    
    # Allow to plot log densities instead
    if use_log_densities:
        #logmin  = np.log(kwargs.get("scale_min")) if "scale_min" in kwargs else np.log(np.amin(datacube))
        #logmax  = np.log(kwargs.get("scale_max")) if "scale_max" in kwargs else np.log(np.amax(datacube))
        logmin  = np.log(kwargs.get("scale_min")) if "scale_min" in kwargs else np.amin(np.log(camera_grid))
        logmax  = np.log(kwargs.get("scale_max")) if "scale_max" in kwargs else np.amax(np.log(camera_grid))
        normed_log_cutoff = (np.log(cutoff)-logmin)/(logmax-logmin)
        #print("Scales:",logmin,logmax,normed_log_cutoff,flush=True)
        for dataslice in camera_grid:
            log_dataslice = np.log(dataslice)
            normed_log_dataslice = (log_dataslice-logmin)/(logmax-logmin)
            r,g,b,a = transferFunction(normed_log_dataslice, cutoff=normed_log_cutoff, \
                                       max_opacity=max_opacity)
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
            #print(r[(image.shape[0]-1)//2,(image.shape[1]-1)//2],\
            #      g[(image.shape[0]-1)//2,(image.shape[1]-1)//2],\
            #      b[(image.shape[0]-1)//2,(image.shape[1]-1)//2],\
            #      a[(image.shape[0]-1)//2,(image.shape[1]-1)//2],\
            #      image[(image.shape[0]-1)//2,(image.shape[1]-1)//2])
    else:
        #minimum = kwargs.get("scale_min") if "scale_min" in kwargs else np.amin(datacube)
        #maximum = kwargs.get("scale_max") if "scale_max" in kwargs else np.amax(datacube)    
        minimum = kwargs.get("scale_min") if "scale_min" in kwargs else np.amin(camera_grid)
        maximum = kwargs.get("scale_max") if "scale_max" in kwargs else np.amax(camera_grid)    
        normed_cutoff = (cutoff-minimum)/(maximum-minimum)
        #print(minimum, maximum, normed_cutoff)
        for dataslice in camera_grid:
            normed_dataslice = (dataslice - minimum)/(maximum - minimum)
            #print('Dataslice:', np.amin(dataslice),np.amax(dataslice),\
            #      np.amin(normed_dataslice),np.amax(normed_dataslice))
            r,g,b,a = transferFunction(normed_dataslice, cutoff=normed_cutoff, \
                                       max_opacity=max_opacity)
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]

                                        
    return np.swapaxes( np.clip(image, 0.0, 1.0), 0, 1)

