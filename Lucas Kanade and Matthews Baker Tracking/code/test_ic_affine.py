import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from InverseCompositionAffine import InverseCompositionAffine
from file_utils import mkdir_if_missing

data_name = 'race'      # could choose from (car1, car2, landing, race, ballet) 

# load data name
data = np.load('../data/%s.npy' % data_name)

# obtain the initial rect with format (x1, y1, x2, y2)
if data_name == 'car1':
    initial = np.array([170, 130, 290, 250])   
elif data_name == 'car2':
    initial = np.array([59,116,145,151])    
elif data_name == 'landing':
    initial = np.array([440, 80, 560, 140]) 
elif data_name == 'race':
    initial = np.array([200, 310, 350, 380])
elif data_name == 'ballet':
    initial = np.array([700, 210, 775, 300])      
else:
    assert False, 'the data name must be one of (car1, car2, landing, race, ballet)'

numFrames = data.shape[2]
w = initial[2] - initial[0]
h = initial[3] - initial[1]

# load over frames
rects = []
rects.append(initial)
fig = plt.figure(1)
ax = fig.add_subplot(111)
for i in range(numFrames-1):
    print("frame****************", i)
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = rects[i]

    # run algorithm
    M = InverseCompositionAffine(It, It1, rect)
    print('M ', M)

    # transform the old rect to new one
    corners = np.array([[rect[0], rect[1], 1], 
                        [rect[2], rect[3], 1]]).transpose()
    newRect = np.matmul(M, corners).transpose().reshape((4, ))
    rects.append(newRect)

    rect = newRect

    # Show image
    print("Plotting: ", rect)
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(It1, cmap='gray')
    save_path = "../results/ic_affine/%s/frame%06d.jpg" % (data_name, i+1)
    mkdir_if_missing(save_path)
    plt.savefig(save_path)
   # plt.pause(0.01)
    ax.clear()