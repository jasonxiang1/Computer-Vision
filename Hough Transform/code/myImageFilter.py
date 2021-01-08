import cv2
import numpy as np

def myImageFilter(img0, hfilt):
    # Your implemention
    #calculate rows and columns of img0
    img0_row = len(img0)
    img0_col = len(img0[0])
    img0_tot = img0_row * img0_col    
    hfilt_halfrow = int(len(hfilt)/2)
    hfilt_halfcol = int(len(hfilt[0])/2)
    
    i=0
    j=0
    
    img1_unsqueezed = np.zeros((img0_row,img0_col,1))
    img0_bordered = cv2.copyMakeBorder(img0, hfilt_halfrow, hfilt_halfrow, hfilt_halfcol, hfilt_halfcol,cv2.BORDER_REPLICATE)
    
    for i in range(hfilt_halfcol,len(img0_bordered)-hfilt_halfcol):
        for j in range(hfilt_halfrow,len(img0_bordered[0])-hfilt_halfrow):
            imag_mat = img0_bordered[i-hfilt_halfcol:(i+hfilt_halfcol+1),j-hfilt_halfrow:(j+hfilt_halfrow+1)]
            conv_mat = hfilt[::-1,::-1]
            img1_unsqueezed[i-hfilt_halfcol,j-hfilt_halfrow] = np.sum(conv_mat*imag_mat)
            
    img1 = np.squeeze(img1_unsqueezed)
    return img1