import h5py as h5
import numpy as np
import sys

print("Loading data", flush=True)

dxy = float(sys.argv[1])
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])
data = np.array([np.loadtxt(frame) for frame in sys.argv[4:]]) # tau, x, y, e

dt = dtau = 0.03585
tauRange = data[:,0,0]
tRange = np.copy(tauRange)

# add an extra column to data for t coordinates
data = np.concatenate( (np.zeros_like(data[:,:,0])[:,:,np.newaxis], data), axis=2 )

# doesn't matter what this is, just declare it
final = None

for iFrame, frame in enumerate(data):
    t = tau = frame[0,1]
    print('Processing tau =', tau, flush=True)
    if iFrame == 0:
        # add 2 extra z slices for initial timestep
        Nz = 2
        zpts = np.linspace(-0.5*dt,0.5*dt,num=Nz)
        output = np.tile(frame,(Nz,1,1))

        for iz, zSlice in enumerate(output):
            zSlice[:,0] = np.full_like( zSlice[:,0], t )
            zSlice[:,1] = np.full_like( zSlice[:,1], zpts[iz] )
        
        # re-shape to (Nx,Ny,Nz,5) and set column order to t, x, y, z, e
        output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[0,2,3,1,4]]
        final = np.copy(output)

    print("\t - set up", flush=True)
    unelapsed_ts = tRange[iFrame:]
    zpts = np.sqrt(unelapsed_ts**2 - tau**2)
    zpts = np.concatenate((-zpts[-1:0:-1],zpts))
    tpts = np.concatenate((unelapsed_ts[-1:0:-1],unelapsed_ts))
    Nz = len(zpts)
    
    print("\t - tiling", flush=True)
    output = np.tile(frame,(Nz,1,1))

    print("\t - filling", flush=True)
    # set 0th column to t coordinate, 1st column to z coordinate
    #for iz, zSlice in enumerate(output):
    #    zSlice[:,0] = np.full_like( zSlice[:,0], tpts[iz] )
    #    zSlice[:,1] = np.full_like( zSlice[:,1], zpts[iz] )
    output[:,:,0] = tpts[:,np.newaxis]
    output[:,:,1] = zpts[:,np.newaxis]
    
    # re-shape to (Nx,Ny,Nz,5) and set column order to t, x, y, z, e
    print("\t - reshaping", flush=True)
    output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[0,2,3,1,4]]

    print("\t - stacking", flush=True)
    final = np.dstack((final, output))
        
# reshape and sort
print('Reshape, sort, and split final array', flush=True)
final = final.reshape([final.size//5,5])
final = final[np.lexsort((final[:,3], final[:,2], final[:,1], final[:,0]))]

final = np.split(final, (np.where(np.diff(final[:,0])>1e-6)[0]+1).tolist())

print('Saving results', flush=True)


#for iTimeslice, timeslice in enumerate(final):
#    print(iTimeslice, timeslice.shape)
#    np.savetxt('all_frames/post_collision_frames_vs_t/frame_' \
#               + str(iTimeslice) + '.dat', timeslice, fmt="%lf")


for iTimeslice, timeslice in enumerate(final):
    outfilename = 'all_frames/post_collision_frames_vs_t/frame_' \
                  + str(iTimeslice).zfill(4) + '.h5'
    hf = h5.File(outfilename, 'w')
    timeslicesize = len(timeslice)
    Nz = timeslicesize//(Nx*Ny)
    output = timeslice.reshape(Nx,Ny,Nz,5)
    hf.create_dataset('t', data = timeslice[0,0])
    hf.create_dataset('x', data = output[:,0,0,1])
    hf.create_dataset('y', data = output[0,:,0,2])
    hf.create_dataset('z', data = output[0,0,:,3])
    hf.create_dataset('energy_density', data = output[:,:,:,4])
    hf.close()
    
print('Done!')
