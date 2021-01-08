import cv2
import numpy as np

#import myEdgeFilter
from myEdgeFilter import myEdgeFilter
#import myHoughTransform
from myHoughTransform import myHoughTransform

def myHoughLines(H, nLines):
    
    rho_resolution = 1
    theta_resolution = 1
    
    #declare variables of Img1
    rho_max = len(H)*rho_resolution
    theta = np.arange(0,180,theta_resolution)
    theta_max = np.max(theta)
    rho = np.arange(-rho_max,rho_max,rho_resolution)
        
    #use 3x3 NMS filter
    #assume odd NMS filter size
    NMS_size = 3
    NMS_midpoint = np.int((NMS_size/2))
    NMS_filter_ones = np.int_(np.ones((NMS_size * NMS_size)).reshape((NMS_size,NMS_size)))
    NMS_filter_ones[NMS_midpoint,NMS_midpoint] = 0
    H_bordered = cv2.copyMakeBorder(H,NMS_midpoint,NMS_midpoint,NMS_midpoint,NMS_midpoint,cv2.BORDER_REPLICATE)
    H_bordered_row = len(H_bordered)
    H_bordered_col = len(H_bordered[0])
    
    H_NMS_array = np.zeros_like(H_bordered)
    count = 0
    for i in range(NMS_midpoint,H_bordered_row-NMS_midpoint):
        for j in range(NMS_midpoint,H_bordered_col-NMS_midpoint):
            temp_filt = NMS_filter_ones * H_bordered[i-NMS_midpoint:(i+NMS_midpoint+1),j-NMS_midpoint:(j+NMS_midpoint+1)]
            temp_max = np.max(temp_filt)
            if (H_bordered[i,j] > temp_max):
                H_NMS_array[i,j] = 1
                count +=1    
    
    H_NMS_array = H_NMS_array * H_bordered
    H_NMS_array_unbordered = H_NMS_array[NMS_midpoint:(H_bordered_row-NMS_midpoint),NMS_midpoint:(H_bordered_col-NMS_midpoint)]
    

    nonzero_indices_row,nonzero_indices_col = np.nonzero(H_NMS_array_unbordered>0)
    #nonzero_indices_row,nonzero_indices_col = np.nonzero(H>0)


    rho_indices_row = (nonzero_indices_row*rho_resolution)
    theta_indices_col = (nonzero_indices_col*theta_resolution)
    
    #H_NMS_array_unbordered[H_NMS_array_unbordered<50] = 0
    
    nLines_rho = H_NMS_array_unbordered[rho_indices_row,theta_indices_col]
    #print(nLines_rho.shape)
    
    rho_small_to_big = np.argsort(nLines_rho)
    #print(rho_small_to_big[::-1])
    rho_indices_row = rho_indices_row[rho_small_to_big[::-1]]
    theta_indices_col = theta_indices_col[rho_small_to_big[::-1]]
    
    rhos = rho_indices_row[:(nLines)]
    thetas = theta_indices_col[:(nLines)]

    return rhos,thetas