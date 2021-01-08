import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import BRIEF

# TODO

'''
Algorithm:
    - declare array of rotations to compute [i.e. 0 to 360 in 10 deg increments]
    - declare blank array to record number of matches
    - create for loop
    - at every for loop, call up function in BRIEF.py
    - record the number of matches at each increments 
    - at the end of for loop plot number of correct matches vs degree of rotation
'''

#declare array with 10 deg increments from 0 to 360 deg
rot_arr = np.arange(0,190,10,dtype=int)
len_rot_arr = len(rot_arr)

#declare blank array to collect number of matches
mat_arr = []

#declare variable for reference image
im_ref = cv2.imread('../data/model_chickenbroth.jpg')
locs_ref, desc_ref = BRIEF.briefLite(im_ref)  

#declare test image
im_test = cv2.imread('../data/chickenbroth_01.jpg')
#compute rows and columns of the test image
im_test_rows = len(im_test)
im_test_cols = len(im_test[0])


for i in range(len_rot_arr):
    
    #get rotation matrix around the center at the degree specificed by rotation array
    rot_mat = cv2.getRotationMatrix2D((np.int32(im_test_cols/2),np.int32(im_test_rows/2)),rot_arr[i],1)
    
    #rotate test image by degree specified by rotation array
    rot_im_test = cv2.warpAffine(im_test,rot_mat,(im_test_cols,im_test_rows))
    
    # #plot the rotated image for testing purposes
    # plt.imshow(rot_im_test);plt.show()

    #compute the interest locations of the rotated test image
    locs_test, desc_test = BRIEF.briefLite(rot_im_test) 
    
    #compute matches between rotated image and reference image
    matches = BRIEF.briefMatch(desc_ref, desc_test)
    
    #append length of matches to matches array declared earlier
    mat_arr = np.append(mat_arr,len(matches))
    

#plot bar graph of number of matches vs rotation iteration
plt.bar(rot_arr,mat_arr);plt.show()

print('done')