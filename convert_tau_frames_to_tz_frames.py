import h5py as h5
import numpy as np
import sys

print("Loading data", flush=True)

dxy = float(sys.argv[1])
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])


def output_to_text(iFrame, data_in, alldata):
    data = data_in.reshape([data_in.size//4,4])
    data = np.c_[ data, np.zeros((len(data),3)) ]
    bigdata = np.swapaxes(np.tile(data,(Nx*Ny,1,1)), 0, 1)
    for iTauslice, tauslice in enumerate(bigdata):
        tauslice[:,4:] = alldata[int(data[iTauslice,3],),:,1:]        
    bigdata = (bigdata.reshape([bigdata.size//7,7]))[:,[1,4,5,2,6]]
    bigdata = bigdata[np.lexsort((bigdata[:,3], bigdata[:,2], bigdata[:,1], bigdata[:,0]))]
    
    # finally, interpolate to uniform grid in z (including endpoints)
    # with spacing comparable to dxy if possible
    
    
    # save results to text file
    np.savetxt('all_frames/post_collision_frames_vs_t/frame_' \
               + str(iFrame) + '.dat', bigdata, fmt="%lf")
    
def output_to_hdf5(iFrame, data_in, alldata):
    data = data_in.reshape([data_in.size//4,4])
    data = np.c_[ data, np.zeros((len(data),3)) ]
    bigdata = np.swapaxes(np.tile(data,(Nx*Ny,1,1)), 0, 1)
    for iTauslice, tauslice in enumerate(bigdata):
        tauslice[:,4:] = alldata[int(data[iTauslice,3],),:,1:]        
    bigdata = (bigdata.reshape([bigdata.size//7,7]))[:,[1,4,5,2,6]]
    bigdata = bigdata[np.lexsort((bigdata[:,3], bigdata[:,2], bigdata[:,1], bigdata[:,0]))]
    outfilename = 'all_frames/post_collision_frames_vs_t/frame_' \
                  + str(iFrame).zfill(4) + '.h5'
    hf = h5.File(outfilename, 'w')
    datasize = len(bigdata)
    Nz = datasize//(Nx*Ny)
    output = bigdata.reshape(Nx,Ny,Nz,5)
    hf.create_dataset('t', data = bigdata[0,0])
    hf.create_dataset('x', data = output[:,0,0,1])
    hf.create_dataset('y', data = output[0,:,0,2])
    hf.create_dataset('z', data = output[0,0,:,3])
    hf.create_dataset('energy_density', data = output[:,:,:,4])
    hf.close()



data = np.array([np.loadtxt(frame) for frame in sys.argv[4:]]) # tau, x, y, e

dt = dtau = 0.03585
tauRange = data[:,0,0]
tRange = np.copy(tauRange)

# add an extra column to data for t coordinates
#data = np.concatenate( (np.zeros_like(data[:,:,0])[:,:,np.newaxis], data), axis=2 )

# doesn't matter what this is, just declare it
final = None

for iFrame, frame in enumerate(data):
    t = tau = frame[0,0]
    print('Processing tau =', tau, flush=True)
    if iFrame == 0:
        # add 2 extra z slices for initial timestep
        Nz = 2
        zpts = np.linspace(-0.5*dt,0.5*dt,num=Nz)
        output = np.full((Nz,4), float(iFrame))

        output[:,0] = 0
        output[:,1] = t
        output[:,2] = zpts
        
        final = np.copy(output)

    #print("\t - set up", flush=True)
    unelapsed_ts = tRange[iFrame:]
    unelapsed_tinds = np.arange(len(tRange))[iFrame:]
    zpts = np.sqrt(unelapsed_ts**2 - tau**2)
    zpts = np.concatenate((-zpts[-1:0:-1],zpts))
    tpts = np.concatenate((unelapsed_ts[-1:0:-1],unelapsed_ts))
    tinds = np.concatenate((unelapsed_tinds[-1:0:-1],unelapsed_tinds))
    Nz = len(zpts)
    
    #print("\t - tiling", flush=True)
    output = np.full((Nz,4), float(iFrame))

    #print("\t - filling", flush=True)
    # set 0th column to t coordinate, 1st column to z coordinate
    output[:,0] = tinds
    output[:,1] = tpts
    output[:,2] = zpts
    
    # stack along first axis
    #print("\t - stacking", flush=True)
    final = np.vstack((final, output))
    
    #print("\t - printing", flush=True)
    elements_to_print = (final[:,0] == iFrame)
    #output_to_text(iFrame, final[elements_to_print], data)
    output_to_hdf5(iFrame, final[elements_to_print], data)
    final = final[np.logical_not(elements_to_print)]
            
print('Done!')
