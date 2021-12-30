import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
from matplotlib import cm                                                                          
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

chosen_colormap = cm.get_cmap('plasma', 12)

def transferFunction(x):
	return np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])


def main():
	""" Volume Rendering """
	
	# Load Datacube
	f = h5.File('datacube.hdf5', 'r')
	datacube = np.array(f['density'])
	
	# Datacube Grid
	Nx, Ny, Nz = datacube.shape
	x = np.linspace(-Nx/2, Nx/2, Nx)
	y = np.linspace(-Ny/2, Ny/2, Ny)
	z = np.linspace(-Nz/2, Nz/2, Nz)
	points = (x, y, z)
	
	# Do Volume Rendering at Different Veiwing Angles
	Nangles = 10
	for i in range(Nangles):
		
		print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
	
		# Camera Grid / Query Points -- rotate camera view
		angle = 2.0*i*np.pi / Nangles
		N = 180
		c = np.linspace(-N/2, N/2, N)
		qx, qy, qz = np.meshgrid(c,c,c)
		qxR = qx
		qyR = qy * np.cos(angle) - qz * np.sin(angle) 
		qzR = qy * np.sin(angle) + qz * np.cos(angle)
		qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
		
		# Interpolate onto Camera Grid
		camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
		
		# Do Volume Rendering
		image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
	
		for dataslice in camera_grid:
			log_dataslice = np.log(dataslice)
			normed_log_dataslice = (log_dataslice - np.amin(log_dataslice)) \
						/(np.amax(log_dataslice) - np.amin(log_dataslice))
			#print(np.array(transferFunction(normed_log_dataslice)).shape)
			r,g,b,a = transferFunction(normed_log_dataslice)
			image[:,:,0] = a*r + (1-a)*image[:,:,0]
			image[:,:,1] = a*g + (1-a)*image[:,:,1]
			image[:,:,2] = a*b + (1-a)*image[:,:,2]
		
		image = np.clip(image,0.0,1.0)
		
		# Plot Volume Rendering
		plt.figure(figsize=(4,4), dpi=80)
		
		plt.imshow(image)
		plt.axis('off')
		
		# Save figure
		plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
	
	
	
	# Plot Simple Projection -- for Comparison
	plt.figure(figsize=(4,4), dpi=80)
	
	plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
	plt.clim(-5, 5)
	plt.axis('off')
	
	# Save figure
	plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
	#plt.show()
	

	return 0
	


  
if __name__== "__main__":
  main()
