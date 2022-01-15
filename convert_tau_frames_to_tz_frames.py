import h5py as h5
import numpy as np
import sys

dxy = float(sys.argv[1])
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])
data = np.array([np.loadtxt(frame) for frame in sys.argv[4:]]) # tau, x, y, e

#tauFinal = 12.43995 # hard-coded for this event!!!
tauFinal = 12.5 # hard-coded for this event!!!
dtau = 0.03585
tFinal = tauFinal
dt = dtau
#zmax = 25.0 # hard-coded for this event!!!
#dz = dxy
#Nz = int(np.ceil(2.0*zmax/dz))+1
#if Nz%2==0:
#    Nz += 1

tauRange = data[:,0,0]
tRange = np.copy(tauRange)

print('tRange=',tRange)
if True:
    exit

# add an extra column to data for t coordinates
print('data.shape =', data.shape)
data = np.concatenate( (np.zeros_like(data[:,:,0])[:,:,np.newaxis], data), axis=2 )
print('data.shape =', data.shape)

# doesn't matter what this is, just declare it
final = None

for iFrame, frame in enumerate(data):
    t = tau = frame[0,0]
    if iFrame == 0:
        # make all three z slices the same
        Nz = 3
        zpts = np.linspace(-0.5*dt,0.5*dt,num=Nz)
        output = np.tile(frame,(Nz,1,1))

        for iz, zSlice in enumerate(output):
            zSlice[:,0] = np.full_like( zSlice[:,0], t )
            zSlice[:,1] = np.full_like( zSlice[:,1], zpts[iz] )
        
        # re-shape to (Nx,Ny,Nz,5) and set column order to t, x, y, z, e
        output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[0,2,3,1,4]]
        final = np.copy(output)
    else:
        unelapsed_ts = tRange[iFrame:]
        print(unelapsed_ts)
        zpts = np.sqrt(unelapsed_ts**2 - tau**2)
        zpts = np.concatenate((-zpts[-1:0:-1],zpts))
        tpts = np.concatenate((unelapsed_ts[-1:0:-1],unelapsed_ts))
        Nz = len(zpts)
        
        output = np.tile(frame,(Nz,1,1))

        print('output.shape = ', output.shape)

        # set 0th column to t coordinate, 1st column to z coordinate
        for iz, zSlice in enumerate(output):
            zSlice[:,0] = np.full_like( zSlice[:,0], tpts[iz] )
            zSlice[:,1] = np.full_like( zSlice[:,1], zpts[iz] )
        
        # re-shape to (Nx,Ny,Nz,5) and set column order to t, x, y, z, e
        output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[0,2,3,1,4]]
        print(final.shape, output.shape)
        final = np.dstack((final, output))
        print(final.shape)
        
# reshape and sort
final = final.reshape([final.size//5,5])
#final = final[final[:, 0].argsort()]

np.savetxt('see_if_it_works.dat', final, fmt="%lf")




#outfilename = (sys.argv[1].replace('timestep', 'frame')).replace('.dat','.h5')
#hf = h5.File(outfilename, 'w')
#hf.create_dataset('x', data = output[:,0,0,0])
#hf.create_dataset('y', data = output[0,:,0,1])
#hf.create_dataset('z', data = output[0,0,:,2])
#hf.create_dataset('energy_density', data = output[:,:,:,3])
#hf.close()
