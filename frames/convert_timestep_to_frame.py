import h5py as h5
import numpy as np
import sys

data = np.loadtxt(sys.argv[1], usecols=(0,1,2,3,4))
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])
dz = float(sys.argv[4])

tau = data[0,0]
data[:,[3,4]] *= 0.197327

Nz = int(np.ceil(2.0*tau/dz))+1
if Nz%2==0:
    Nz += 1

zpts = np.linspace(-tau,tau,num=Nz)
output = np.tile(data,(Nz,1,1))

for iz, zSlice in enumerate(output):
    zSlice[:,0] = np.full_like( zSlice[:,0], zpts[iz] )

# re-shape to (Nx,Ny,Nz,5) and set column order to x, y, z, T, e
output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[1,2,0,3,4]]

# set column order to x, y, z, T, e
#output = (output.reshape((output.size//5,5)))[:,[1,2,0,3,4]]
#np.savetxt( sys.argv[1].replace("timestep", "frame"), \
#            output, fmt="%10.6f" )

outfilename = (sys.argv[1].replace('timestep', 'frame')).replace('.dat','.h5')
print(outfilename)
hf = h5.File(outfilename, 'w')
hf.create_dataset('density', data = output)
hf.close()