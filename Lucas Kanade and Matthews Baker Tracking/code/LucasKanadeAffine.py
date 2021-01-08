import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here



    
    '''
    Goal: Track landing tape in video by translating the rectangle through time
    
    Algorithm:
        1. create while loop
        2. include end conditional to while loop with maxIters
        3. warp given frame I with inital warp parameters p
        4. compute the error impave between the template image and warped frame image
        5. warp the x and y directional gradients with the given warp parameters
        6. compute the jacobian for all pixel locations
        7. compute the steepest descent image based on the computed warped gradiant and jacobian
        8. compute the hessian matrix with the given descent image
        9. compute teh delta p using the hessian matrix, steepest descent, and the error
        10. add delta p onto p and output p to update rect
    '''


    ''' 
    Notes:
        - Use RectBivariateSpline.ev function to create a finer image to move the template image in more detail
    '''

    #define magnitude of delta p to be well above the threshold to start
    mag_del_p = threshold + 10
    #define iteration number to be zero to start
    i = 0
    
    Iy_shape,Ix_shape = It.shape
    
    if x1 > x2:
        temp = x2
        x1 = x2
        x1 = temp
        
    if y1 > y2:
        temp = y2
        y2 = y1
        y1 = temp
    
    if x1 > Ix_shape:
        x1 = Ix_shape
    elif x1 < 0:
        x1 = 0

    if x2 > Ix_shape:
        x2 = Ix_shape
    elif x2 < 0:
        x2 = 0
        
    if y1 > Iy_shape:
        y1 = Iy_shape
    elif y1 < 0:
        y1 = 0

    if y2 > Iy_shape:
        y2 = Iy_shape
    elif y2 < 0:
        y2 = 0
    
    # #compute interpolated warp of image\
    # to compute arrays out of interp variables It_interp(np.arange(1000),np.arange(1100))
    It1_interp = RectBivariateSpline(np.arange(Iy_shape),np.arange(Ix_shape),It1)
    #compute interpolated warp of template image
    It_interp = RectBivariateSpline(np.arange(Iy_shape),np.arange(Ix_shape),It)
    
    
    rect_y_range = np.linspace(np.int32(y1),np.int32(y2),num=np.int32(y2-y1))
    rect_x_range = np.linspace(np.int32(x1),np.int32(x2),num=np.int32(x2-x1))
    It_rect = It[np.int32(y1):np.int32(y2)+1,np.int32(x1):np.int32(x2)+1]
    #It_rect = It_interp(rect_y_range,rect_x_range)
    
    onesI_mat = np.ones_like(It_rect)
    Iindices = np.array(np.where(onesI_mat>=0))
    #offset rows by y value of rect
    Iindices[0,:] = Iindices[0,:] + y1
    #offset columns by x value of rect
    Iindices[1,:] = Iindices[1,:] + x1
    #convert indices array to homogeneous and to float32 format
    #first row is y - values, second row are the x - values
    Iindices_homo = np.array(np.append(Iindices,np.ones(len(Iindices[0]))[np.newaxis,:],axis=0),dtype=np.float32)
    Iindices_homo = np.concatenate((Iindices_homo[1,:][np.newaxis,:],Iindices_homo[0,:][np.newaxis,:],Iindices_homo[2,:][np.newaxis,:]),axis=0)

    #initialize the 

    #create while loop
    while mag_del_p > threshold:
        #create max_iter conditional
        if i == maxIters:
            break
        
        #warp image by warp parameters
        # reshape the output affine matrix
        warp_mat = np.array([[1.0+p[0], p[1],    p[2]],
                      [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
        #TODO: resolve warping and error calculations
        # #perform warp on interpolated image
        I_Wxp = np.matmul(warp_mat,Iindices_homo)
        
        #output intensity of image at the warped coordinates
        warped_It1 = It1_interp.ev(np.array(I_Wxp[1,:],dtype=float),np.array(I_Wxp[0,:],dtype=float))
        
        #compute error between the template at the original pixel coordinates and the imaged at the warped pixel coordinates
        #in the boudning box
        error_warpedtemp = It_rect.flatten()[np.newaxis,:] - warped_It1     
        
        #compute gradient of the warped image
        #compute the x and y gradient of the original image It1
        Ix = It1_interp.ev(np.array(I_Wxp[1,:],dtype=float),np.array(I_Wxp[0,:],dtype=float),dy=1)
        Iy = It1_interp.ev(np.array(I_Wxp[1,:],dtype=float),np.array(I_Wxp[0,:],dtype=float),dx=1)
        #warp the gradiaent according to warp parameters
        #assumed warped coordinates for gradients at the same as I_Wxp
        
        #evaluate jacobian for each pixel point
        #create zero matrix 
        len_win = len(I_Wxp[0])
        J_Wp = np.zeros((2,6,len_win))
        #compute jacobian
        J_Wp[0,0,:] = I_Wxp[0,:]
        J_Wp[0,1,:] = I_Wxp[1,:]
        J_Wp[0,2,:] = np.ones((len_win))
        J_Wp[1,3,:] = I_Wxp[0,:]
        J_Wp[1,4,:] = I_Wxp[1,:]
        J_Wp[1,5,:] = np.ones((len_win))
        
        
        #compute the steepest descent image
        grad_vect = np.append(Ix[np.newaxis,np.newaxis,:],Iy[np.newaxis,np.newaxis,:],axis=1)
        steep_descent = np.einsum('ijk,ikl->ijl',grad_vect.transpose(2,0,1),J_Wp.transpose(2,0,1))
        steep_descent = steep_descent.transpose(1,2,0)
        
        
        #compute hessian
        hess_unsumed = np.einsum('ijk,ikl->ijl',steep_descent.transpose(2,1,0),steep_descent.transpose(2,0,1))
        #sum hessian along one axis
        hess_summed = np.sum(hess_unsumed,axis=0)
        
        #compute deltap
        inv_hess = np.linalg.inv(hess_summed)
        steep_des_trans = steep_descent.transpose(1,0,2)
        delp_unsum = steep_des_trans*error_warpedtemp[np.newaxis,:,:]
        delp_sum = np.matmul(inv_hess,np.sum(delp_unsum,axis=2))
        
        #update p
        p = p + delp_sum
        
        #update mag of p
        mag_del_p = np.linalg.norm(delp_sum)
        
        #increment iteration number
        i += 1
        
    M = warp_mat

    
    return M
