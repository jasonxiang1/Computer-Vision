import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    #############################
    # TO DO ...
    # Generate testpattern here
    
    #assume origin at the center of the patch
    #use uniform distribution for test patterns
    
    lower_bound = 0.0
    upper_bound = (patch_width*patch_width)-1
    
    compareX_float = np.random.uniform(low=lower_bound,high=upper_bound,size=(nbits))
    compareY_float = np.random.uniform(low=lower_bound,high=upper_bound,size=(nbits))
    
    compareX = np.array(compareX_float,dtype=np.int32)
    compareY = np.array(compareY_float,dtype=np.int32)
    
    
    return  compareX, compareY

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    # TO DO ...
    # compute locs, desc here
    
    patch_width = 9
    patch_halfwidth = np.int(9/2)
    test_len = 256
    
    #compute length of locsDoG
    len_DoG = len(locsDoG)
    
    #initialize locs and desc vectors
    locs_table = np.zeros((1,3))
    desc_table = np.zeros((1,test_len))
    
    for i in range(len_DoG):

        locs_row = np.int(locsDoG[i,1])
        locs_col = np.int(locsDoG[i,0])
        
        #compute locs_channel to be a constant
        locs_channel=2
        # locs_channel = np.int(locsDoG[i,2])
        
        #check if can create 9x9 patch at interest point
        #compute image lengths at gaussian scale
        len_rows = len(gaussian_pyramid[:,:,locs_channel])
        len_cols = len(gaussian_pyramid[0,:,locs_channel])
        
        
       #check if you can create a 9x9 patch 
        if locs_row+patch_halfwidth>len_rows-1:
            continue
        elif locs_row-patch_halfwidth<0:
            continue
        elif locs_col+patch_halfwidth>len_cols-1:
            continue
        elif locs_col-patch_halfwidth<0:
            continue
        
        #create patch at the interest points at the gaussian scale
        patch = gaussian_pyramid[locs_row-patch_halfwidth:locs_row+patch_halfwidth+1,
                                 locs_col-patch_halfwidth:locs_col+patch_halfwidth+1,locs_channel]
        
        #flatten patch
        patch_flatten = patch.flatten()
        
        #use compareX and compareY to compare value in the patch
        patch_flatten_compX = patch_flatten[compareX]
        patch_flatten_compY = patch_flatten[compareY]
        
        #compare compX with compY
        patch_flatten_compare = patch_flatten_compX < patch_flatten_compY
        patch_flatten_compare = np.array(patch_flatten_compare,dtype=np.int32)
        patch_flatten_compare = patch_flatten_compare[np.newaxis,:]
        
        #store locs to locs_table
        locsDoG_temp = locsDoG[i]
        locsDoG_temp = locsDoG_temp[np.newaxis,:]
        locs_table = np.append(locs_table,locsDoG_temp,axis=0)
        
        #store desc to desc_table
        desc_table = np.append(desc_table,patch_flatten_compare,axis=0)
        
        
    
    locs = locs_table[1:,:]
    desc = desc_table[1:,:]
    
    return locs, desc



def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    # TO DO ...
    
    #compute run DoGDecctor function
    k = np.sqrt(2)
    levels = [-1, 0, 1, 2, 3, 4]
    locsDoG,gaussian_pyramid = DoGdetector(im,k=k,levels=levels)
    
    
    #pass im and other variables to computeBrief function
    locs,desc = computeBrief(im,gaussian_pyramid,locsDoG,k,levels,compareX,compareY)
    
    
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio 
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    # plt.waitforbuttonpress(0)
    plt.close(fig)
    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    #construct for loop to plot rotating image to mirrored reference point
    im_ref = cv2.imread('../data/model_chickenbroth.jpg')
    im_rot = cv2.imread('../data/model_chickenbroth.jpg')
    
    #compute comparison between incline L and R
    im1 = cv2.imread("../data/incline_L.png")
    im2 = cv2.imread("../data/incline_R.png")
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
        
    
