import cv2
import numpy as np

def myImageFilterX(img0, hfilt):
    # Your implemention
   

    img0_row = len(img0)
    img0_col = len(img0[0])
    hfilt_halfrow = int(len(hfilt)/2)
    hfilt_halfcol = int(len(hfilt[0])/2)
    img0_bordered = cv2.copyMakeBorder(img0, hfilt_halfrow, hfilt_halfrow, hfilt_halfcol, hfilt_halfcol,cv2.BORDER_REPLICATE)
    img0_bordered_row = len(img0_bordered)
    img0_bordered_col = len(img0_bordered[0])
    
    
    x_reshape= np.int_(np.zeros((1,(2*hfilt_halfcol+1))))
    conv_mat = hfilt[::-1,::-1]
    
    
    for i in range(hfilt_halfrow,len(img0_bordered)-hfilt_halfrow):
        x_temp = np.transpose(img0_bordered[i-hfilt_halfrow:i+hfilt_halfrow+1,:])
        x_reshape = np.concatenate((x_reshape,x_temp),axis=0)
    
    
    diag_diag_mat = np.dot(x_reshape[1:,:],hfilt)
    
    hfilt_eye = np.eye(2*hfilt_halfrow+1)
    idx = np.arange(diag_diag_mat.shape[0]-hfilt_eye.shape[0]+1)[:,None] + np.arange(hfilt_eye.shape[0])
    out = diag_diag_mat[idx]*hfilt_eye
    
    
    sum_2D = np.sum(out,axis=1)
    sum_1D = np.sum(sum_2D,axis=1)
    end_zero_vect = np.zeros((2*hfilt_halfcol))
    out_bordered = np.append(sum_1D,end_zero_vect)
    img2_bordered = out_bordered.reshape((img0_row,(img0_col+(2*hfilt_halfrow)),1))
    img1_unsqueezed = img2_bordered[:,:(-2*hfilt_halfrow)]
    img1 = np.squeeze(img1_unsqueezed)
    return img1