import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

import matplotlib.pyplot as plt

def computeH(p1, p2):
    """
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2
    assert p2.shape[0] == 2
    #############################
    # TO DO ...
    
    '''
    Algorithm:
        Devise method to transfer information in each array into an array that is twice as long at the original array for each of the two
        Element multiply arrays of both p1 and p2 and get 2N by 9 array of points
        use linalg module to compute SVD of resulting array
        find the column of the array of V in which it has the small value
        declare that column to be the homography values
    '''
    
    #compute row length and column width of input arrays
    rows,cols = p1.shape
    
    #compute p2 vector 2Nx9 matrix
    #first create Nx18 array then reshape to 2Nx9
    p2_1to2 = np.transpose(p2)*-1
    p2_3 = np.ones((cols,1))*-1
    p2_4to6 = np.zeros((cols,3))
    p2_7to8 = np.transpose(p2)
    p2_9= np.ones((cols,1))
    p2_10to12 = np.zeros((cols,3))
    p2_13to14 = np.transpose(p2)*-1
    p2_15 = np.ones((cols,1))*-1
    p2_16to17 = np.transpose(p2)
    p2_18 = np.ones((cols,1))
    
    p2_concat = np.hstack((p2_1to2,p2_3,p2_4to6,p2_7to8,p2_9,p2_10to12,p2_13to14,p2_15,p2_16to17,p2_18)).reshape((2*cols,9))
    
    #compute p1 vector 2Nx9 matrix
    p1_1to3 = np.ones((cols,3))
    p1_4to6 = np.zeros((cols,3))
    p1_7to9 = np.hstack((np.transpose(p1[0,:])[:,np.newaxis],np.transpose(p1[0,:])[:,np.newaxis],np.transpose(p1[0,:])[:,np.newaxis]))
    p1_10to12 = np.zeros((cols,3))
    p1_13to15 = np.ones((cols,3))
    p1_16to18 = np.hstack((np.transpose(p1[1,:])[:,np.newaxis],np.transpose(p1[1,:])[:,np.newaxis],np.transpose(p1[1,:])[:,np.newaxis]))
    
    p1_concat = np.hstack((p1_1to3,p1_4to6,p1_7to9,p1_10to12,p1_13to15,p1_16to18)).reshape((2*cols,9))
    
    #combine p1 and p2 matrices together
    result_mat = p2_concat*p1_concat
    
    #perform svd on 2Nx9 matrix
    svd_mat = np.linalg.svd(result_mat)
    
    #the third index is the V^T matrix
    #tranpose the V matrix
    v_svd_mat = np.transpose(svd_mat[2])
    
    #output the column vector with the smallest value
    # v_svd_smallestvalue_index = np.argwhere(np.abs(v_svd_mat)==np.min(np.abs(v_svd_mat)))
    # #first value is rows, second value is columns
    # v_svd_smallcol = v_svd_mat[:,v_svd_smallestvalue_index[0][1]]
    
    #last column of V is the most optimal h array
    #see Least Squares by SVD lecture slides for proof
    #select the last column to be the chosen matrix
    v_svd_smallcol = v_svd_mat[:,-1]
    
    #output result as return value to function
    H2to1 = v_svd_smallcol.reshape((3,3))
    #H2to1_test, status_test = cv2.findHomography(p2.T,p1.T)
    #import pdb; pdb.set_trace()
    
    
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    """
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    """
    ###########################
    # TO DO ...
    
    #note perform RANSAC only on points that are matched together
    '''
    Algorithm:
        find 2 random point correspondences
        fit a line on that model
        compute the normal distance between each point and the model
            based on (tol) units of length between the line and the point
        store inliers into an array and compare 
    '''
    
    #remember locs is (x,y) <=> (col,row)
    #define matches between locs1 and locs2
    locs1_matches = locs1[matches[:,0], 0:2]
    locs2_matches = locs2[matches[:,1], 0:2]
    
    #compute length of matches
    match_rows,match_cols = matches.shape
    #rows dictate the amount of matches there are
    #create arange vector that is the length of matches array
    match_indices = np.arange(match_rows)
    
    #declare number of inliers varibles
    num_inliers = 0
    
    for i in range(num_iter):
        #select 4 random points from list of matches
        ran_rowindices = np.random.choice(match_indices,4)
        
        #output random indices from matches array
        matches_ran = matches[ran_rowindices,:]
        
        #output randomized locs1 and locs2 with the random matches array
        locs1_ran_matches = np.transpose(locs1[matches_ran[:,0],0:2])
        locs2_ran_matches = np.transpose(locs2[matches_ran[:,1],0:2])
        
        #input randomly selected matches into computeH function
        H_est_2to1 = computeH(locs1_ran_matches,locs2_ran_matches)
        
        #preprocess locs2 array to be [x,y,1]^T array that is N long
        locs2_homogeneous = np.vstack((np.transpose((locs2_matches[:,0:2])),np.ones((1,match_rows))))
    
        #multiply homography array using matrix multiplication
        locs1_estimated_homo = np.matmul(H_est_2to1,locs2_homogeneous)
        
        #convert homogeneous to nonhomogeneous coordinates
        locs1_estimated_nonhomo = locs1_estimated_homo[0:2,:]/locs1_estimated_homo[2,:]
        #round estimated coordinates
        locs1_estimated_nonhomo = np.round(locs1_estimated_nonhomo)
        
        #compute difference between locs1_estimated with locs1
        locs1_diff = -locs1_estimated_nonhomo+np.transpose(locs1_matches)
        
        #compute magnitude of locs1 difference along axis=0
        locs1_diff_norm = np.linalg.norm(locs1_diff,axis=0)
        
        #if normal of the difference is less than or equation to the tolerance then make 1
        #else make zero
        
        #process points that are within the tolerance
        locs1_diff_tol = np.where(locs1_diff_norm<=tol,1,0)
        
        #sum the total ones together
        temp_inliers = np.sum(locs1_diff_tol)
        
        #if value is the largest number of inliers then store
        #else drop and move to next iteration
        if temp_inliers > num_inliers:
            num_inliers=temp_inliers
            bestH = H_est_2to1
        
    print(num_inliers, "\t",(num_inliers/match_rows)*100)
    return bestH



def compositeH(H, template, img):
    """
    Returns final warped harry potter image. 
    INPUTS
        H - homography 
        template - desk image
        img - harry potter image
    OUTPUTS
        final_img - harry potter on book cover image  
    """
    # TODO
    '''
    Algorithm:
        perform warpPerspective function on harry potter cover image
        combine harry potter cover with cv desk image
    '''
    
    #format of warpPerspective
    #inputs: image to be warped, homography matrix, size of image
    #select the size of the desk image to the size of the image
    
    #compute shape of template image
    temp_rows = len(template)
    temp_cols = len(template[0])
    #computer shape of img
    img_rows = len(img)
    img_cols = len(img[0])
    
    #perform warp perspective on the harry potter image and 
    warped_img = cv2.warpPerspective(img,H,(temp_cols,temp_rows))
    
    #the following website was used to combine both images together
    #https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html
    
    temp_img = template
    
    #declare region of interest
    warped_rows = len(warped_img)
    warped_cols = len(warped_img[0])
    roi = temp_img[0:warped_rows,0:warped_cols]
    
    #conver image to gray for processing
    img2gray = cv2.cvtColor(warped_img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    #black out are of logo in ROI
    temp_img_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    
    #take only region of logo from logo image
    warped_img_fg = cv2.bitwise_and(warped_img,warped_img,mask=mask)
    
    #put warped image in ROI and modify main image
    dst = cv2.add(temp_img_bg,warped_img_fg)
    temp_img[0:warped_rows,0:warped_cols] = dst
    
    #plt.imshow(temp_img);plt.show()

    final_img = temp_img    

    return final_img


if __name__ == "__main__":
    im1 = cv2.imread("../data/model_chickenbroth.jpg")
    im2 = cv2.imread("../data/chickenbroth_01.jpg")
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    #return value of ransacH is to return the inliers from the matches
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    print('done')
    

