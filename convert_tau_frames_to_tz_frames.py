import h5py as h5
import numpy as np
import sys

print("Loading data", flush=True)

dxy = float(sys.argv[1])
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])


def output_to_text(iFrame, data_in, alldata):
    print("Output to text")
    data = data_in.reshape([data_in.size//4,4])
    print(data.shape)
    data = np.c_[ data, np.zeros((len(data),3)) ]
    print(data.shape)
    print(data)
    bigdata = np.swapaxes(np.tile(data,(Nx*Ny,1,1)), 0, 1)
    print("bigdata.shape=",bigdata.shape)
    print(alldata.shape)
    #for tauslice in bigdata:
    #    tauslice[
    exit(1)
    
    
    
    
    data = data[np.lexsort((data[:,3], data[:,2], data[:,1], data[:,0]))]
    np.savetxt('all_frames/post_collision_frames_vs_t/frame_' \
               + str(iFrame) + '.dat', data, fmt="%lf")
    
def output_to_hdf5(iFrame, data):
    print("\t\t (A)",flush=True)
    print(data.shape,flush=True)
    data = data.reshape([data.size//5,5])
    print(data.shape,flush=True)
    data = data[np.lexsort((data[:,3], data[:,2], data[:,1], data[:,0]))]
    outfilename = 'all_frames/post_collision_frames_vs_t/frame_' \
                  + str(iFrame).zfill(4) + '.h5'
    print("\t\t (B)",flush=True)
    hf = h5.File(outfilename, 'w')
    datasize = len(data)
    Nz = datasize//(Nx*Ny)
    output = data.reshape(Nx,Ny,Nz,5)
    hf.create_dataset('t', data = data[0,0])
    hf.create_dataset('x', data = output[:,0,0,1])
    hf.create_dataset('y', data = output[0,:,0,2])
    hf.create_dataset('z', data = output[0,0,:,3])
    hf.create_dataset('energy_density', data = output[:,:,:,4])
    hf.close()
    print("\t\t (Ca)",flush=True)



data = np.array([np.loadtxt(frame) for frame in sys.argv[4:]]) # tau, x, y, e

dt = dtau = 0.03585
tauRange = data[:,0,0]
tRange = np.copy(tauRange)

# add an extra column to data for t coordinates
#data = np.concatenate( (np.zeros_like(data[:,:,0])[:,:,np.newaxis], data), axis=2 )

# doesn't matter what this is, just declare it
final = None

for iFrame, frame in enumerate(data):
    t = tau = frame[0,1]
    print('Processing tau =', tau, flush=True)
    if iFrame == 0:
        # add 2 extra z slices for initial timestep
        Nz = 2
        zpts = np.linspace(-0.5*dt,0.5*dt,num=Nz)
        #output = np.tile(frame,(Nz,1,1))
        output = np.full((Nz,4), float(iFrame))

        output[:,0] = 0
        output[:,1] = t
        output[:,2] = zpts
        #for iz, zSlice in enumerate(output):
        #    zSlice[:,0] = np.full_like( zSlice[:,0], 0 )
        #    zSlice[:,1] = np.full_like( zSlice[:,1], t )
        #    zSlice[:,2] = np.full_like( zSlice[:,2], zpts[iz] )
        
        # re-shape to (Nx,Ny,Nz,5) and set column order to t, x, y, z, e
        #output = np.swapaxes(output.reshape((Nz,Ny,Nx,5)), 0, 2)[:,:,:,[0,2,3,1,4]]
        final = np.copy(output)

    print("\t - set up", flush=True)
    unelapsed_ts = tRange[iFrame:]
    unelapsed_tinds = np.arange(len(tRange))[iFrame:]
    print(unelapsed_ts)
    print(unelapsed_tinds)
    print(tau)
    zpts = np.sqrt(unelapsed_ts**2 - tau**2)
    zpts = np.concatenate((-zpts[-1:0:-1],zpts))
    tpts = np.concatenate((unelapsed_ts[-1:0:-1],unelapsed_ts))
    tinds = np.concatenate((unelapsed_tinds[-1:0:-1],unelapsed_tinds))
    #print('tpts =',tpts)
    #print('zpts =',zpts)
    Nz = len(zpts)
    
    print("\t - tiling", flush=True)
    #output = np.tile(frame,(Nz,1,1))
    output = np.full((Nz,4), float(iFrame))

    print("\t - filling", flush=True)
    # set 0th column to t coordinate, 1st column to z coordinate
    print(output.shape, tpts.shape, zpts.shape)
    output[:,0] = tinds
    output[:,1] = tpts
    output[:,2] = zpts
    
    print("\t - stacking", flush=True)
    # stack along first axis
    final = np.vstack((final, output))
    
    print("\t - printing", flush=True)
    print(final.shape)
    elements_to_print = (final[:,0] == iFrame)
    #print("1",elements_to_print.shape)
    #print("2a",final[elements_to_print].shape)
    output_to_text(iFrame, final[elements_to_print], data)
    #output_to_hdf5(iFrame, final[elements_to_print])
    #print("2b",final[elements_to_print].shape)
    print("\t\t (Cb)",flush=True)
    final = final[np.logical_not(elements_to_print)]
    print("\t\t (D)",flush=True)
    #final = final.reshape([final.size//(Nx*Ny*5),Nx*Ny,5])
    print("\t\t (E)",flush=True)
    #print("3",final.shape)
    print(final.shape)
        



'''
# reshape and sort
print('Reshape, sort, and split final array', flush=True)
final = final.reshape([final.size//5,5])
final = final[np.lexsort((final[:,3], final[:,2], final[:,1], final[:,0]))]

final = np.split(final, (np.where(np.diff(final[:,0])>1e-6)[0]+1).tolist())

print('Saving results', flush=True)

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
    hf.close()'''
    
print('Done!')
