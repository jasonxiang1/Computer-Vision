import cv2
import numpy as np

#additional imports
import numpy.matlib as npm
import math

from GaussianKernel import Gauss2D
from myImageFilter import myImageFilter



def myHoughTransform(InnputImage, rho_resolution, theta_resolution):
	# Your implemention
    
    InnputImage = InnputImage[:,:,np.newaxis]
    
    Img1_rows = len(InnputImage)
    Img1_cols = len(InnputImage[0])
    Img1_val_mat = np.array([Img1_rows, Img1_cols])
    rho_max = np.int_(np.ceil(np.sqrt(np.dot(np.transpose(Img1_val_mat),Img1_val_mat))))
    
    theta_max = 360
    theta = np.arange(0,theta_max,theta_resolution)
    
    rho = np.arange(0,rho_max,rho_resolution)
    
    theta_vec_len = np.int(np.ceil(theta_max*(1/theta_resolution)))
    rho_vec_len = np.int(np.ceil((rho_max)*(1/rho_resolution)))
    
    accum_mat = np.zeros((theta_vec_len,rho_vec_len),dtype=int)
    
    Img1_reshaped_1D = InnputImage.reshape((Img1_rows*Img1_cols,1))
    
    row_Img1_xvec_init = np.arange((Img1_cols),dtype=int)
    row_Img1_xvec_repeat = np.tile(row_Img1_xvec_init,Img1_rows)

    row_Img1_yvec_init = np.arange((Img1_rows),dtype=int)
    row_Img1_yvec_repeat = np.repeat(row_Img1_yvec_init,Img1_cols)

    Img1_vectored = np.concatenate((row_Img1_xvec_repeat,row_Img1_yvec_repeat))
    Img1_vectored_2D_horiz = Img1_vectored.reshape((2,np.int(Img1_vectored.size/2)))
    Img1_vectored_2D_vert = np.transpose(Img1_vectored_2D_horiz)

    #create vector of cos and sin, respectively
    cos_vectored = np.cos(np.deg2rad(theta))
    sin_vectored = np.sin(np.deg2rad(theta))
    
    #combine cos and sin vectors
    theta_cos_sin = np.concatenate((cos_vectored,sin_vectored))
    theta_cos_sin_reshape = theta_cos_sin.reshape((2,np.int(len(theta_cos_sin)/2)))

    #multiple together
    rho_vectored = np.dot(Img1_vectored_2D_vert,theta_cos_sin_reshape)

    #reshape rho_vectored back to img dimensions and multiply with image
    rho_vectored_image_shape = rho_vectored.reshape(Img1_rows,Img1_cols,theta_max)
    
    rho_on_image = (rho_vectored_image_shape * InnputImage)

    rho_on_image[rho_on_image<0] = -1 * rho_on_image[rho_on_image<0]

    rho_on_image_rounded = np.int_(np.around(rho_on_image))

    for i in range(0,len(theta)):
        unique, counts = np.unique(rho_on_image_rounded[:,:,i],return_counts=True)
        accum_mat[i,unique-1] = accum_mat[i,unique-1] + counts
        
    accum_mat[:,0] = 0
    accum_mat[:,-1] = 0

    H = np.transpose(accum_mat)
    rhos = rho
    thetas = theta

    return H, rhos, thetas
