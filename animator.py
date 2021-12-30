import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py as h5
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

chosen_colormap = cm.get_cmap('plasma', 12)

def transferFunction(x0):
	frac = 0.35
	x = np.clip(x0, frac, 1.0)/(1.0-frac)-frac/(1.0-frac)
	r,g,b,a = np.transpose(np.array(chosen_colormap(x)), axes=[2,0,1])
	#a = 0.6*np.exp( -(x - 1.0)**2/1.0 )
	a = 0.6*np.exp( -(x0 - 1.0)**2/0.0001 ) \
		+ 0.3*np.exp( -(x0 - 0.9)**2/0.0001 ) \
		+ 0.1*np.exp( -(x0 - 0.8)**2/0.0001 ) \
		+ 0.01*np.exp( -(x0 - 0.7)**2/0.0001 ) \
		+ 0.01*np.exp( -(x0 - 0.6)**2/0.0001 ) \
		+ 0.01*np.exp( -(x0 - 0.5)**2/0.0001 )
	#a = 0.0
	#steps = 11
	#delta = 1.0/(steps-1)
	#for center in np.linspace(0,1,steps):
	#	a += x**2*np.exp( -(x - center)**2/delta**4 )
	return r,g,b,a

Nangles  = 100
datacube = None
points   = None

def animate(i):
	print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n', flush=True)

	# Camera Grid / Query Points -- rotate camera view
	angle = 2.0*i*np.pi / Nangles
	N = 180
	c = np.linspace(-N/2, N/2, N)
	qx, qy, qz = np.meshgrid(c,c,c)
	qxR = qx
	qyR = qy * np.cos(angle) - qz * np.sin(angle) 
	qzR = qy * np.sin(angle) + qz * np.cos(angle)
	qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
	
	minimum = np.amin(datacube)
	maximum = np.amax(datacube)
	logmin  = np.amin(np.log(datacube))
	logmax  = np.amax(np.log(datacube))

	# Interpolate onto Camera Grid
	camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
		
	# Do Volume Rendering
	image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
	
	for dataslice in camera_grid:
		log_dataslice = np.log(dataslice)
		normed_log_dataslice = (log_dataslice - logmin)/(logmax - logmin)
		r,g,b,a = transferFunction(normed_log_dataslice)
		#normed_dataslice = (dataslice - minimum)/(maximum - minimum)
		#r,g,b,a = transferFunction(normed_dataslice)
		image[:,:,0] = a*r + (1-a)*image[:,:,0]
		image[:,:,1] = a*g + (1-a)*image[:,:,1]
		image[:,:,2] = a*b + (1-a)*image[:,:,2]
		
	image = np.clip(image,0.0,1.0)
		
	plt.imshow(image)
	plt.axis('off')



def main():
	global datacube
	global points

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
	
	# Plot Volume Rendering
	fig = plt.figure(figsize=(4,4), dpi=80)
		
	# Do Volume Rendering at Different Veiwing Angles
	#for i in range(Nangles):
	ani = animation.FuncAnimation(fig, animate, np.arange(Nangles))
	
	#f = "animation.mp4" 
	#writervideo = animation.FFMpegWriter(fps=10) 
	#ani.save(f, writer=writervideo)

	f = "animation.gif" 
	#writergif = animation.PillowWriter(fps=10) 
	ani.save(f, writer='imagemagick', fps=10)

	return 0
	


  
if __name__== "__main__":
  main()
