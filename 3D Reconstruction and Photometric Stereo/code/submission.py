# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #


# Insert your package here
from skimage.color import rgb2xyz
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import lsqr as splsqr
import pdb
from utils import integrateFrankot

import numpy as np
import scipy.linalg
import scipy.ndimage
#import skimage.io
import cv2
import helper
import matplotlib.pyplot as plt
import utils

'''
Q3.2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    #pass
    ''' 
    Algorithm:
        1. normalize x-coordinated by dividing by max of M x-direction
        2. normalize y-coordinates by dividing by max of M y-direction
        3. use all points to compute the main matrix
        4. perform SVD and select the last column of the V matrix
        5. output a F matrix -> perform SVD again to enforce rank 2
        5. pass the fundamental matrix through the refineF helper function
        6. output reshaped 3x3 F matrix from the function
    '''
    #Outputted Fundamental Matrix will transfer a point on the first camera image
    #to a line on the second camera image
    
    #Remember that the x-coordinate is the column entry
    #Remember that the y-coordinate is the row entry
    
    
    #construct normalizatin vector T_nonhomo (2x2)
    T_nonhomo = np.array([1/M,0,0,1/M]).reshape((2,2))
    
    #multiply T_nonhomo with pts1 and pts2
    #reshape back into Nx2 form
    pts1_homo = np.transpose(np.matmul(T_nonhomo,np.transpose(pts1)))
    pts2_homo = np.transpose(np.matmul(T_nonhomo,np.transpose(pts2)))
    
    #comput the Nx9 matrix based on the 8-point algorithm
    #construct Nx9 matrix column by column
    A_col1 = (pts1_homo[:,0]*pts2_homo[:,0])[:,np.newaxis]
    A_col2 = (pts1_homo[:,0]*pts2[:,1])[:,np.newaxis]
    A_col3 = pts1_homo[:,0][:,np.newaxis]
    A_col4 = (pts1_homo[:,1]*pts2_homo[:,0])[:,np.newaxis]
    A_col5 = (pts1_homo[:,1]*pts2_homo[:,1])[:,np.newaxis]
    A_col6 = pts1_homo[:,1][:,np.newaxis]
    A_col7 = pts2_homo[:,0][:,np.newaxis]
    A_col8 = pts2_homo[:,1][:,np.newaxis]
    A_col9 = np.ones((len(pts1_homo),1))
    #column stack the columns together
    A_mat = np.column_stack((A_col1,A_col2,A_col3,A_col4,A_col5,A_col6,A_col7,A_col8,A_col9))
    
    #compute SVD of Nx9 matrix
    U, E, V_transpose = scipy.linalg.svd(A_mat)
    
    #select last column of the V matrix
    f_unranked_unshaped = np.transpose(V_transpose)[:,-1]
    f_unranked = f_unranked_unshaped.reshape((3,3))
    
    #perform SVD on matrix again
    U_ranked, E_ranked, V_transpose_ranked = scipy.linalg.svd(f_unranked)
    
    #change last diagonal matrix to be equal to 0
    E_ranked[-1] = 0
    E_ranked = np.diagflat(E_ranked)
    
    #multiply matrix together again
    F_norm_ranked = np.linalg.multi_dot([U_ranked, E_ranked, V_transpose_ranked])
    
    # #refine F
    # F_norm_ranked = helper.refineF(F_norm_ranked,pts1,pts2)
    
    
    #un-normalize F matrix
    #construct homogeneous T matrix (3x3)
    T_homo_flattened = np.array([1/M,1/M,1])
    T_homo = np.diagflat(T_homo_flattened)
    #create F matrix
    F = np.linalg.multi_dot([np.transpose(T_homo),F_norm_ranked,T_homo])

    return F    
    
    

'''
Q3.2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pass
    



'''
Q3.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    #pass

    '''
    Algorithm:
        1. Multiply the F matrix by K2^T on the left side and K on the right side
    '''
    
    #treat K1 as K
    #treat K2 as K'

    E = np.linalg.multi_dot([np.transpose(K2), F, K1])

    return E

'''
Q3.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    #pass

    '''
    Algorithm:
        1. Declare blank 3D coordinate array and error array
        2. Build for loop
        3. Each iteratin of for loop, construct A matrix
        4. Perform SVD on A matrix
        5. Extract last column of V matrix
        6. Append column as 3D coordinate to the blank 3D coordinate array
        7. Out of for loop, compute reprojection error based on equation in hw prompt
    '''

    #define blank arrays to be appended through the for loop
    P_arr = np.zeros((1,4))
    
    #compute length of correspondences for the for loop
    corr_len = len(pts1)
    
    #define camera matrix vectors for C1 and C2
    c1_p1_trans = C1[0,:]
    c1_p2_trans = C1[1,:]
    c1_p3_trans = C1[2,:]
    c2_p1_trans = C2[0,:]
    c2_p2_trans = C2[1,:]
    c2_p3_trans = C2[2,:]

    #build for loop for each point correspondence
    for i in range(corr_len):
        #define x and y values for each camera matrix
        pts1_x = pts1[i,0]
        pts1_y = pts1[i,1]
        pts2_x = pts2[i,0]
        pts2_y = pts2[i,1]
        
        #construct A matrix based on least squares algorithm in class
        A_mat = np.vstack((pts1_y*c1_p3_trans-c1_p2_trans,c1_p1_trans-pts1_x*c1_p3_trans,pts2_y*c2_p3_trans-c2_p2_trans,c2_p1_trans-pts2_x*c2_p3_trans))
        
        #perform SVD on A matrix
        U, E, V_trans = scipy.linalg.svd(A_mat)
        
        #extract last column of V matrix
        point_3d_temp = np.transpose(V_trans)[:,-1]
        
        # #normalize 3d point based on last point
        #point_3d_temp_norm = (point_3d_temp/point_3d_temp[-1])[np.newaxis,:]
        point_3d_temp_norm = point_3d_temp[np.newaxis,:]
        
        #append 3d point to the end of P_arr
        P_arr = np.append(P_arr,point_3d_temp_norm,axis=0)
        
    P_arr = P_arr[1:,:]
    
    
    #calculate reprojection error
    #compute homogeneous vector of P_arr
    #P_arr_homo = np.append(P_arr, np.ones((corr_len,1)),axis=1)
    P_arr_homo = P_arr
    x_2d_c1_est = np.matmul(C1,np.transpose(P_arr_homo))
    x_2d_c2_est = np.matmul(C2,np.transpose(P_arr_homo))
    #nonhomo and normalize estimated coordinates
    x_2d_c1_est_norm = x_2d_c1_est[:2,:]/x_2d_c1_est[-1,:]
    x_2d_c2_est_norm = x_2d_c2_est[:2,:]/x_2d_c2_est[-1,:]
    #compute difference between point correspondence and estimated values
    c1_diff = pts1 - np.transpose(x_2d_c1_est_norm)
    c2_diff = pts2 - np.transpose(x_2d_c2_est_norm)
    #compute maginutde along one direction of the arrays
    c1_diff_mag = np.linalg.norm(c1_diff,axis=1)
    c2_diff_mag = np.linalg.norm(c2_diff,axis=1)
    #compute sum
    err_arr = np.sum(c1_diff_mag) + np.sum(c2_diff_mag)
    
    P = P_arr[:,:3]/P_arr[:,-1][:,np.newaxis]
    err = err_arr
    
    return P, err
    

'''
Q3.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    #pass
    
    
    #focus on vectorizing the checking code along the epipolar line
    #this function is for one point on the first camera image
    '''
    Algorithm:
        1. Declare a set window size
        3. Compute the epipole line on im2 based on the point from im1
        4. W/o a for loop, output all x and correspond y coordinates from the outputted line
        5. build for loop
        6. iterate through each x coordinate and compute gaussian window around the x,y coordinate
    '''
    
    #declare size for the window
    win_size = 45
    pad_size = np.int(win_size/2)
    #declare sigma for gaussian weighting
    sigma = 15
    #create 2d array whose center is spiked at 1 to create gaussian filter
    win_gauss = np.zeros((win_size,win_size))
    win_gauss[pad_size,pad_size] = 1
    #perform gaussian blur on window to create gaussian weighting window
    win_gauss = scipy.ndimage.gaussian_filter(win_gauss,sigma)[:,:,np.newaxis]
    win_gauss = np.concatenate((win_gauss,win_gauss,win_gauss),axis=2)
    
    #create homogeneous coordinate for x1 and y1
    im1_homo_coord = np.array([x1,y1,1]).reshape((3,1))
    
    #compute epipolar line coefficients on the second camera
    im2_line_coef = np.matmul(F,im1_homo_coord)
    #line is of the form ax+by+c = 0
    im2_a = im2_line_coef[0]
    im2_b = im2_line_coef[1]
    im2_c = im2_line_coef[2]
    
    #compute coordinates for im2 along the epipolar line
    #compute max dimensions of im2
    im2_max_x = len(im2[0])
    im2_max_y = len(im2)
    #compute range vectors in x and y directions (i.e. columns and rows)
    im2_x_vector = np.arange(im2_max_x)
    im2_y_vector = np.arange(im2_max_y)
    #compute corresponding y-coordinates based on equation:
    #y = -c/b-(a/b)x
    im2_calc_y_vector = -(im2_c/im2_b) - (im2_a/im2_b)*im2_x_vector
    im2_round_y_vector = im2_calc_y_vector.astype(np.int)
    #combine x-vector and y-coordinates together
    #format is (x,y)
    im2_line_y_vector = np.append(im2_x_vector[:,np.newaxis],im2_round_y_vector[:,np.newaxis],axis=1)
    im2_line_y_vector = im2_line_y_vector[im2_line_y_vector[:,1]>0]
    im2_line_y_vector = im2_line_y_vector[im2_line_y_vector[:,1]<im2_max_y-1]
    #x= -c/a-(b/a)y
    im2_calc_x_vector = -(im2_c/im2_a) - (im2_b/im2_a)*im2_y_vector
    im2_round_x_vector = im2_calc_x_vector.astype(np.int)
    im2_line_x_vector = np.append(im2_round_x_vector[:,np.newaxis],im2_y_vector[:,np.newaxis],axis=1)
    im2_line_x_vector = im2_line_x_vector[im2_line_x_vector[:,0]>0]
    im2_line_x_vector = im2_line_x_vector[im2_line_x_vector[:,0]<im2_max_y-1]   
    
    #compute the intensity of the window around the first camera image
    im1_win = im1[y1-pad_size:y1+pad_size+1,x1-pad_size:x1+pad_size+1,:]
    im1_win_gauss = im1_win*win_gauss
    
    #define blank norm value
    min_norm = 1000
    
    #define temp x and y placeholder variables
    x_temp = 0
    y_temp = 0
    
    #define buffer around the x1,y1 coordinate point in image 2 to locate images
    
    #because lines are parallel along the y axis
    #build for loop to ride along calculated x vector rather than calculated y vectors
    for i in range(len(im2_line_x_vector)):
        #compute window at y,x
        x_i = im2_line_x_vector[i,0]
        y_i = im2_line_x_vector[i,1]
        
        #skip through if y is less than the value of the pad size
        if (y_i<pad_size) or (y_i+pad_size>im2_max_y-1) or (y_i-y1>100):
            continue
        
        #create window around x_i and y_i
        win_i = im2[y_i-pad_size:y_i+pad_size+1,x_i-pad_size:x_i+pad_size+1,:]
        
        #perform gaussian filter
        win_i_gauss = win_i*win_gauss
        
        #calculate intensity difference between windows
        inten_euc = np.linalg.norm(im1_win_gauss-win_i_gauss)
        
        #if this the minimal dist then record the x and y coordiantes
        if inten_euc<min_norm:
            min_norm = inten_euc
            x_temp = x_i
            y_temp = y_i
            
    x2 = x_temp
    y2 = y_temp
            
    return x2,y2
    


'''
Q3.5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    #pass
    
    '''
    Algorithm:
        find 2 random point correspondences
        fit a line on that model
        compute the normal distance between each point and the model
            based on (tol) units of length between the line and the point
        store inliers into an array and compare 
    '''
    
    #define num_iter and tol values
    num_iter = 2000
    tol = 10
    
    # #normalize pts based on value M
    # pts1_norm = pts1/M
    # pts2_norm = pts2/M
    
    #compute length of pts
    len_pts = len(pts1)
    
    #create arange vector that is the same length as the pts
    matches_indices = np.arange(len_pts)
    
    #declare number of inliers to be equal to 0
    num_inliers = 0
    
    
    #create homogeneous vector for pts1
    pts1_homo = np.append(pts1,np.ones((len_pts,1)),axis=1)
    #create homogeneous vector for pts2
    pts2_homo = np.append(pts2,np.ones((len_pts,1)),axis=1)
    
    #goal compare the outputs of the fundamnental matrix 
    
    #create for loop
    for i in range(num_iter):
        #select eight random pts rows
        ran_rowindices = np.random.choice(matches_indices,8)
        
        #output random selections from pts1 and pts2
        pts1_ran = pts1[ran_rowindices,:]
        pts2_ran = pts2[ran_rowindices,:]
        
        #compute Fundamental Matrix F
        F_i = eightpoint(pts1_ran, pts2_ran, M)
        
        #compute line coefficients for all x1 and y1 values
        line_coef2 = np.matmul(F_i,np.transpose(pts1_homo))
        #line is of the form ax+by+c = 0
        a_vect = line_coef2[0,:]
        b_vect = line_coef2[1,:]
        c_vect = line_coef2[2,:]
        
        #compute estimated x-values from given y-values in pts2
        #x = -c/a -b/a*y
        pts2_x_est = -(c_vect/a_vect)-(b_vect/a_vect)*pts2[:,1]
        
        # #convert to nonhomo coordinates
        # pts2_x_est = pts2_x_est[:,:2]/pts2_x_est[:,-1][:,np.newaxis]
        
        #compute distance to line
        dist_num = a_vect*pts2[:,0] + b_vect*pts2[:,1] + c_vect
        dist_num = np.abs(dist_num)
        dist_denum = a_vect*a_vect + b_vect*b_vect
        dist_denum = np.sqrt(dist_denum)
        dist_line = dist_num/dist_denum
        
        #compute difference between x-values in pts2 and estimated x-values
        x_val_diff = pts2[:,0] - pts2_x_est
        #compute absolute values of the difference
        x_val_diff = np.abs(x_val_diff)
        
        #compute the amount of inliers based on tolerance factor
        inliers_vect = np.where(x_val_diff<tol,1,0)
        
        #sum up all inliers elements
        inliers_sum = np.sum(inliers_vect)
        
        line_dist_vect = np.where(dist_line<tol,1,0)
        line_dist_sum = np.sum(line_dist_vect)
        
        #compared if it is the higest number of inliers
        #if want to use x-value distance, change line_dist_sum with inliers_sum
        if line_dist_sum > num_inliers:
            val = pts2_x_est
            num_inliers = inliers_sum
            F_choose = F_i
            inliers_choose = inliers_vect
            
    F = F_choose
    inliers = inliers_choose
    
    return F, inliers
    
        
        
        


'''
Q3.5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    #pass
    
    '''
    Algorithm:
        1. Compute the different rows of the matrix based on lecture slides
    '''

    #assuming that the epipole e is the rodrigues vector
    
    #compute magnitude of r
    r1_mag = np.linalg.norm(r)
    
    #compute r1_transpose based on algorithm
    r1_trans = r/r1_mag
    
    #compute r2_transpose based on algorithm
    r2_trans = np.array([-r[1,0], r[0,0], 0])[:,np.newaxis]
    r2_mag = np.linalg.norm(r2_trans)
    r2_trans = r2_trans/r2_mag
    
    #compute r3_transpose based on algorithm
    r3_trans = np.cross(np.transpose(r1_trans),np.transpose(r2_trans))
    
    R = np.concatenate((np.transpose(r1_trans),np.transpose(r2_trans),r3_trans),axis=0)
    
    return R
    

'''
Q3.5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    #pass
    
    '''
    Algorithm:
        1. compute intermediary A matrix
        2. compute parameters based off of A matrix
        3. compute conditional based on article
        4. compute S1/2 and u vectors
        5. compute conditional based on article
        6. compute sin conditional based on article
    '''
    
    #compute A matrix
    A= (R-np.transpose(R))/2
    #compute parameters from A matrix
    p = np.transpose(np.array([A[2,1],A[0,2],A[1,0]]))
    s = np.linalg.norm(p)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    
    theta = np.arctan2(s,c)
    
    #first conditional
    if (s==0.0) and (c==1.0):
        r = np.zeros((3,1))
    elif (s==0.0) and (c==-1.0):
        R_nz = R+np.identity(3)
        #compute norm of columns
        R_col_nz = np.linalg.norm(R_nz,axis=0)
        #find the index of the nonzero element
        col_nz_indices = np.nonzero(R_col_nz)
        #select the first index from list
        nz_index_select = col_nz_indices[0]
        #extract column from the rotation matrix
        v = R_nz[:,nz_index_select]
        u = v/np.linalg.norm(v)
        r = u*np.pi
        if (np.round(np.linalg.norm(r),2)==3.14) & (r[0] == 0.0) & (r[1]==0.0) & (r[2] < 0):
            r = -r
        elif (r[0]==0.0) & (r[1]<0):
            r = -r
        elif r[0]<0:
            r=-r
        else:
            r = r
    elif np.sin(theta) != 0:
        u = p/s
        theta = np.arctan2(s,c)
        r = u*theta        
        
    
        
    return r
        
        


'''
Q3.5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original 
            and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    #pass
    
    '''
    Algorithm:
        1. extract t2 and r2 from the x input variable
        2. compute rodrigues matrix from r2
        3. compute camera matrices of 1 and 2
        4. compute estimated projects onto 1 and 2
        5. compute residuals between p1 and p2 and estimated p1 and p2
    '''
    
    #note: do not need to optimize intrinsics
    #already has p1 and p2 projections
    #need to compute the estimated 1 and 2 projections
    
    #break down x array into its components
    t2 = x[-3:][:,np.newaxis]
    r2 = x[-6:-3][:,np.newaxis]
    P = x[:-6].reshape([-1,3])
    
    #compute Rodrigues rotation matrix from r2
    R_rect = rodrigues(r2)
    
    #construct M2 from R_rect and t2
    M2_const = np.append(R_rect,t2,axis=1)
    
    #Compute C1 and C2 camera matrices
    C1 = np.matmul(K1,M1)
    C2 = np.matmul(K2,M2_const)
    
    #convert 3d points to homogeneous coordinates in 3d
    P_homo = np.append(P,np.ones((len(P),1)),axis=1)
    
    #Computer estimated p1 and p2 projects based on camera matrices and 3d points
    p1_est = np.matmul(C1,np.transpose(P_homo))
    p2_est = np.matmul(C2,np.transpose(P_homo))
    
    #normalize projected coordinates based on third coordinate
    p1_est_norm = p1_est[:2,:]/p1_est[-1,:][np.newaxis,:]
    p2_est_norm = p2_est[:2,:]/p2_est[-1,:][np.newaxis,:]
    
    #compute residual
    p1_hat = np.transpose(p1_est_norm)
    p2_hat = np.transpose(p2_est_norm)
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]),(p2-p2_hat).reshape([-1])])
    
    return residuals[:,np.newaxis]


'''
Q3.5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    #pass
    
    '''
    Algorithm:
        1. compute maximum M value between p1 and p2
        2. compute F and inliers from the ransacF function
        3. compute inliers of P_init
        4. extract rotation matrix from M2_init
        5. perform invRodrigues to get r2 vector from M2 rotation
        6. concatenate r2 and t2 with the inliers of P_init to get x vector
        7. run inputs through scipy leastsq to optimize the residuals function
    '''
    
    #compute maximum value M value between p1 and p2
    combined_var = np.append(p1,p2,axis=1)
    M = np.amax(combined_var)
    
    # #compute F and inliers from the ransacF function
    # F, inliers = ransacF(p1,p2,M)
    
    # #extract values that are only the inliers
    # P_in = P_init[np.nonzero(inliers)]
    # p1_in = p1[np.nonzero(inliers)]
    # p2_in = p2[np.nonzero(inliers)]
    
    P_in = P_init
    p1_in = p1
    p2_in = p2
    
    #extract rotation matrix from M2_init variable
    R2 = M2_init[:,:3]
    #extract translation matrix from M2_init variable
    t2 = M2_init[:,-1]
    
    #output r2 vector using invRodrigues function    
    r2 = invRodrigues(R2)
    
    #concatenate r2,t2, and flattened P_in together into x variable
    P_in_flat = P_in.reshape([-1])
    x_var = np.concatenate([P_in_flat, r2, t2])
    
    #pass variable into scipy leastsq optimization
    #credit Erica Weng for suggesting code snippet below
    obj = lambda x: (rodriguesResidual(K1,M1,p1_in,K2,p2_in,x).flatten())
    x_star = scipy.optimize.leastsq(obj,x_var)[0]
    
    #extract out the r2 and t2 vectors from x_star output
    t2_star = x_star[-3:][:,np.newaxis]
    r2_star = x_star[-6:-3][:,np.newaxis]
    P_star = x_star[:-6].reshape([-1,3])

    #create updated M2 matrix
    R2_star = rodrigues(r2_star)
    M2_star = np.append(R2_star,t2_star,axis=1)
    
    #return values
    M2 = M2_star
    w = P_star
    
    return M2, w


def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Q4.1

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    # Replace pass by your implementation
    #pass
    
    #remember camera is looking at sphere from top down view
    
    #compute width(x) and height (y) of image
    len_y = res[0]
    len_x = res[1]
    
    #break down lighting vectors
    light1 = light[:,0][:,np.newaxis]
    light2 = light[:,1][:,np.newaxis]
    light3 = light[:,2][:,np.newaxis]
    
    #create image array of zeros based on picture size
    sphere_im = np.zeros((res))
    
    #convert half length of x and y to m
    halflen_x = (len_x/2)*pxSize
    halflen_y = (len_y/2)*pxSize
    
    for i in range(len_y):
        #eq to use is (x-1250)^2+(y-1500)^2+z^2 = r^2
        #compute x and y in m
        x_i_m = np.arange(len_x)*pxSize
        y_i_m = i*np.ones(len_x)*pxsize
        
        #verify elements where is within the range of the sphere
        rand_cond_check = np.sqrt(np.square(x_i_m-halflen_x)+np.square(y_i_m-halflen_y))
        
        #create bool matrix
        rand_check_indices = rand_cond_check<rad
        rand_check_indices = np.nonzero(rand_check_indices)
        rand_cond_check = rand_cond_check[rand_check_indices]
        
        #update x_i_m and y_i_m values
        x_i_m = x_i_m[rand_check_indices]
        y_i_m = y_i_m[rand_check_indices]
        
        if len(rand_cond_check) == 0:
            continue
        
        #compute z_i_m vector
        z_i_m = np.sqrt(np.square(rad)-np.square(x_i_m-halflen_x)-np.square(y_i_m-halflen_y))
        
        #compute surface normal in m
        surf_norm_m = np.array([x_i_m,y_i_m,z_i_m])
        
        #compute intensity at each point along row
        inten_1 = np.matmul(np.transpose(light1),surf_norm_m)
        inten_2 = np.matmul(np.transpose(light2),surf_norm_m)
        inten_3 = np.matmul(np.transpose(light3),surf_norm_m)
        
        # #confirm if intensity is positive of note
        # inten_1 = inten_1.clip(min=0)
        # inten_2 = inten_2.clip(min=0)
        # inten_3 = inten_3.clip(min=0)
        
        inten = inten_1 + inten_2 + inten_3
    
        sphere_im[i,rand_check_indices] = inten
        
    #convert to float32
    sphere_im_float32 = sphere_im.astype(np.float32)
    
    #normalize
    sphere_im_float32 = sphere_im_float32*(50/np.amax(sphere_im_float32))
    
    #return the sphere image
    image = sphere_im_float32
    

    return image
    

def loadData(path = "../data/"):

    """
    Q4.2.1

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # Replace pass by your implementation
    #pass
    
    #load all images from the folder
    num_images = 7
    
    #blank image array
    lum_arr = []
    
    for i in range(num_images):
        #load image
        im_name = "input_" + str(i+1)+".tif"
        im_temp = cv2.imread(path+im_name)
        
        #plt.imshow(im_temp);plt.show()
    
        #convert images to xyz color space
        im_temp_xyz = utils.lRGB2XYZ(im_temp)
        
        #extract the luminance channel from the image (i.e. Y-channel)
        y_channel = im_temp_xyz[:,:,1]
        
        #flatten the y channel
        y_channel_flat = y_channel.flatten()[np.newaxis,:]
        
        #append the variable to the end of 
        lum_arr = np.append(lum_arr,y_channel_flat)
        
    #compute image shape
    im_len,im_wid,im_channels = im_temp.shape
    
    lum_arr = lum_arr[np.newaxis,:].reshape((7,im_len*im_wid))
        
    #load the lighting sources file
    light_sources = np.load(path+"sources.npy")
    light_sources = np.transpose(light_sources)

    I = lum_arr
    L = light_sources
    s = (im_len,im_wid)
        
    return I, L, s
    
    


def estimatePseudonormalsCalibrated(I, L):

    """
    Q4.2.2

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # Replace pass by your implementation
    #pass
    
    #in the form Ax=y
    #L^TB = I
    #least squares equation is B = inv(A^TA)@A^T
     
    A = np.transpose(L)
    y = I
    
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),I)

    return B
    

def estimateAlbedosNormals(B):

    '''
    Q4.2.3

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    # Replace pass by your implementation
    # pass

    #from the psuedo normal, extract the magnitude and unit vector
    #magnitude is the albedo
    #unit vector is the normal
    
    #compute the magnitude of the B vector per column
    mag = np.linalg.norm(B,axis=0)[np.newaxis,:]
    
    #compute the norm vector
    norm_vect = B/mag
    
    #return variables
    albedos = mag
    normals = norm_vect
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Q4.2.4

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    # Replace pass by your implementation
    # pass

    #reshape albedos to be the same size at the image shape
    alb_im = albedos.reshape(s)
    
    #reshape normals to be s x 3
    norm_im1 = normals[0,:].reshape(s)[:,:,np.newaxis]
    norm_im2 = normals[1,:].reshape(s)[:,:,np.newaxis]
    norm_im3 = normals[2,:].reshape(s)[:,:,np.newaxis]
    norm_im = np.concatenate((norm_im1,norm_im2,norm_im3),axis=2)
    
    #rescle normal to be from 0 to 1
    norm_im += 1
    norm_im = norm_im/2.0

    return alb_im,norm_im

def estimateShape(normals, s):

    """
    Q4.3.1

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    # Replace pass by your implementation
    pass


def plotSurface(surface):

    """
    Q4.3.1

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    # Replace pass by your implementation
    pass



if __name__ == '__main__':
    #Q4.1
    #define the center of the sphere
    #center is at the origin and 10 cm below the camera
    center = np.array([0, 0, -0.1])
    #define radius of the sphere in m
    rad = 0.005
    #define the pixel size
    pxsize = 0.000005
    #define the resolution of the image
    res = np.array([3000, 2500])
    #define lighting directions
    light1 = np.array([1., 1., 1.])/np.sqrt(3)
    light2 = np.array([1., -1., 1.])/np.sqrt(3)
    light3 = np.array([-1., -1., 1.])/np.sqrt(3)
    #concatentate all lighting directions into one variable
    lighting = np.concatenate((light1[:,np.newaxis],light2[:,np.newaxis],light3[:,np.newaxis]),axis=1)
    # #input variables into renderNDotLSphere function
    # sphere_image = renderNDotLSphere(center,rad,lighting,pxsize,res)
    
    # #out the sphere image
    # plt.imshow(sphere_image);plt.show()
    
    #Q4.2.1
    path = "../data/"
    #input path into the loadData function
    I, L, s = loadData(path)
    
    #estimate pseudonormals
    B = estimatePseudonormalsCalibrated(I, L)
    
    #run through estimate albedos code
    albedos, normals = estimateAlbedosNormals(B)
    
    #output images of the albedo and normals
    albedo_im, normal_im = displayAlbedosNormals(albedos, normals, s)
    
    #display images
    #plt.imshow(albedo_im,cmap="gray");plt.show()
    plt.imshow(normal_im,cmap="rainbow");plt.show()