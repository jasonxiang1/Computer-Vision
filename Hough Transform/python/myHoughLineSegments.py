import glob
import os.path as osp
import numpy as np
import cv2


def myHoughLineSegments(img_in, edgeimage, peakRho, peakTheta, rhosscale, thetasscale):
	# Your implemention
    
    edgeimage = edgeimage[:,:,np.newaxis]
    rhos = rhosscale[peakRho]
    thetas = thetasscale[peakTheta]
    
    edgeimage_rows = len(edgeimage)
    edgeimage_cols = len(edgeimage[0])
    
    edge_nz_row_indices,edge_nz_col_indices = np.nonzero(edgeimage[:,:,0])
    
    edge_nz_len = len(edge_nz_row_indices)
    
    for i in range(0,edge_nz_len):
        x0_comp = edge_nz_col_indices[i]
        y0_comp = np.int_((rhos - (x0_comp*np.cos(np.deg2rad(thetas))))/np.sin(np.deg2rad(thetas)))
        
        diff = np.absolute(y0_comp-edge_nz_row_indices[i])
        
        if (np.any(diff<=4)):
            img_in[edge_nz_row_indices[i],edge_nz_col_indices[i],:] = (0,255,0)
            
    img_output = img_in

    return img_output