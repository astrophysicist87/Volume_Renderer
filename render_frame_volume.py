import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import h5py as h5
import sys
from scipy.interpolate import interpn
from matplotlib import cm                                                                          
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#chosen_colormap = cm.get_cmap('plasma', 256)
chosen_colormap = cm.get_cmap('magma', 256)

def enhance_channel(a, f):
	return np.clip(a*f,0.0,1.0)

def transferFunction(x0):
	#frac = 0.45
	#x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
	#r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
	#a = 0.1*np.exp( -(x0 - 1.0)**2/0.1 ) \
	#	+ 0.0*np.exp( -(x0 - 0.9)**2/0.0001 ) \
	#	+ 0.0*np.exp( -(x0 - 0.8)**2/0.0001 ) \
	#	+ 0.0*np.exp( -(x0 - 0.7)**2/0.0001 ) \
	#	+ 0.0*np.exp( -(x0 - 0.6)**2/0.0001 ) \
	#	+ 0.0*np.exp( -(x0 - 0.5)**2/0.0001 )
	x = x0
	r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
	#enhancement_factor = 1.5
	#r = enhance_channel(r, enhancement_factor)
	#g = enhance_channel(g, enhancement_factor)
	#b = enhance_channel(b, enhancement_factor)
	steps = 11
	delta = 1.0/(steps-1)
	a = delta*np.exp( -(x0 - 1.0)**2/delta**2 )
	for center in np.linspace(0,1,steps-1,endpoint=False):
		a += 2.0*delta*x**2*np.exp( -(x - center)**2/delta**4 )
	return r,g,b,a



def main():
	# Load Datacube
	#f = h5.File('datacube.hdf5', 'r')
	#datacube = np.array(f['density'])
	
	# Datacube Grid
	#Nx, Ny, Nz = datacube.shape
	Nx, Ny, Nz = 235, 235, 19
	datacube = np.loadtxt(sys.argv[1], usecols=(0,1,2,3)).reshape([Nx,Ny,Nz,4])

	#x = np.linspace(-Nx/2, Nx/2, Nx)
	#y = np.linspace(-Ny/2, Ny/2, Ny)
	#z = np.linspace(-Nz/2, Nz/2, Nz)
	x = datacube[:,0,0,0]
	y = datacube[0,:,0,1]
	z = datacube[0,0,:,2]
	datacube = datacube[:,:,:,-1]
	points = (x, y, z)

	datacube += 1e-15
	minimum = np.amin(datacube)
	maximum = np.amax(datacube)
	logmin = np.amin(np.log(datacube))
	logmax = np.amax(np.log(datacube))

	# Do Volume Rendering at Different Veiwing Angles
	Nangles = 1
	#phi, theta = 0.25*np.pi, 0.25*np.pi
	for i in range(Nangles):
		
		print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles), flush=True)
	
		# Camera Grid / Query Points -- rotate camera view
		#angle = 2.0*i*np.pi / Nangles
		angle = 0.25*np.pi
		N = 500
		c = np.linspace(-20.0, 20.0, N)
		qx, qy, qz = np.meshgrid(c,c,c)
		qxR = qx
		qyR = qy * np.cos(angle) - qz * np.sin(angle) 
		qzR = qy * np.sin(angle) + qz * np.cos(angle)
		#qxR = qx * np.cos(angle) + qy * np.sin(angle)
		#qyR = qy * np.cos(angle) - qx * np.sin(angle) 
		#qzR = qz
		qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
	
		# Interpolate onto Camera Grid
		camera_grid = interpn(points, datacube, qi, method='linear',\
                                      bounds_error=False, fill_value=np.amin(datacube)\
                                     ).reshape((N,N,N))
	
		#np.savetxt('rotated_camera_view.dat',\
		#		interpn(points, datacube, qi, method='linear',\
		#		bounds_error=False, fill_value=np.amin(datacube)), fmt="%10.6f")
	
		# Do Volume Rendering
		image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

		for dataslice in camera_grid:
			#log_dataslice = np.log(dataslice)
			#normed_log_dataslice = (log_dataslice-logmin)/(logmax-logmin)
			#r,g,b,a = transferFunction(normed_log_dataslice)
			normed_dataslice = (dataslice - minimum)/(maximum - minimum)
			r,g,b,a = transferFunction(normed_dataslice)
			image[:,:,0] = a*r + (1-a)*image[:,:,0]
			image[:,:,1] = a*g + (1-a)*image[:,:,1]
			image[:,:,2] = a*b + (1-a)*image[:,:,2]
		
		image = np.clip(image,0.0,1.0)
		
		# Plot Volume Rendering
		plt.figure(figsize=(4,4), dpi=500)
		
		plt.imshow(image)
		plt.axis('off')
		
		# Save figure
		plt.savefig('volumerender' + str(i) + '.png', dpi=500, bbox_inches='tight', pad_inches = 0)
	
	
	
	# Plot Simple Projection -- for Comparison
	plt.figure(figsize=(4,4), dpi=80)
	
	plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
	plt.clim(-5, 5)
	plt.axis('off')
	
	# Save figure
	plt.savefig('projection.png', dpi=240, bbox_inches='tight', pad_inches = 0)
	#plt.show()
	

	return 0
	


  
if __name__== "__main__":
  main()
