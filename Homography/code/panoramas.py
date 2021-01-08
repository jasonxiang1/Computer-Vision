import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches

import matplotlib.pyplot as plt


def imageStitching(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    """
    #######################################
    im1_pano = np.zeros((im1.shape[0] + 80, im1.shape[1] + 750, 3), dtype=np.uint8)
    im1_pano[: im1.shape[0], : im1.shape[1], : im1.shape[2]] = im1
    im1_pano_mask = im1_pano > 0

    # TODO ...
    # warp im2 onto pano
    # pano_im=....
    # pano_im_mask = ...
    
    #compute similar format compositeH function in planarH.py file
    #warp im2 to be shape (80,750)
    
    # warped_rows = 80
    # warped_cols = 750
    
    #create warped image and mask
    warped_im2 = cv2.warpPerspective(im2,H2to1,(im1_pano.shape[1],im1_pano.shape[0]))
    warped_rows = len(warped_im2)
    warped_cols = len(warped_im2[0])
    
    pano_im = warped_im2
    pano_im_mask = warped_im2 > 0

    # TODO
    # dealing with the center where images meet.
    # im_center_mask = ...

    im_full = pano_im + im1_pano

    im_R = im_full * np.logical_not(im1_pano_mask)
    im_L = im_full * np.logical_not(pano_im_mask)
    # TODO produce im center, mix of pano_im and im1_pano
    # im_center = ...
    im_center = warped_im2 * np.logical_and(im1_pano_mask,pano_im_mask)
    return im_R + im_L + im_center


def imageStitching_noClip(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    """
    ######################################
    # TO DO ...
    s = 1
    tx = 0
    # clip
    # establish corners
    # create new corners
    # ty = ... used for M_translate matrix
    
    #TODO: determine aspect ratio based on im1 or the base image
    
    #compute constants
    im1_rows = len(im1)
    im1_cols = len(im1[0])
    im2_rows = len(im2)
    im2_cols = len(im2[0])
    
    #compute location of the four corners of both images
    im1_topleft = np.array([0,0])[:,np.newaxis]
    im1_topright = np.array([im1_cols,0])[:,np.newaxis]
    im1_bottomleft = np.array([0,im1_rows])[:,np.newaxis]
    im1_bottomright = np.array([im1_cols,im1_rows])[:,np.newaxis]
                                           
    im2_topleft = np.array([0,0])[:,np.newaxis]
    im2_topright = np.array([im2_cols,0])[:,np.newaxis]
    im2_bottomleft = np.array([0,im2_rows])[:,np.newaxis]
    im2_bottomright = np.array([im2_cols,im2_rows])[:,np.newaxis]

    #comput nonhomogeneous vectors for im1 and im2 corners                                          
    im1_vector = np.hstack((im1_topleft,im1_topright,im1_bottomleft,im1_bottomright))
    im2_vector = np.hstack((im2_topleft,im2_topright,im2_bottomleft,im2_bottomright))
    
    #convert nonhomogeneous to homogeneous coordinates
    im1_vect_homo = np.vstack((im1_vector,np.ones((1,4))))
    im2_vect_homo = np.vstack((im2_vector,np.ones((1,4))))
    
    #compute the homographied coordinates of both vectors
    im1_vect_homographed_homo = np.matmul(H2to1,im1_vect_homo)
    im2_vect_homographed_homo = np.matmul(H2to1,im2_vect_homo)
    
    #stacks the two homographed coordinates for comaprison
    im_stacked_homographed_homo = np.hstack((im1_vect_homographed_homo,im2_vect_homographed_homo))
    #normalize coordinates and convert to nonhomogeneous coordinates
    im_stacked_homographed_nonhomo = im_stacked_homographed_homo[0:2,:]/im_stacked_homographed_homo[2,:]
    
    #im nonhomogeneous size differences
    im_stacked_size_height = np.array([im_stacked_homographed_nonhomo[1,2]-im_stacked_homographed_nonhomo[1,0],im_stacked_homographed_nonhomo[1,3]-im_stacked_homographed_nonhomo[1,1],im_stacked_homographed_nonhomo[1,6]-im_stacked_homographed_nonhomo[1,4],im_stacked_homographed_nonhomo[1,7]-im_stacked_homographed_nonhomo[1,5]])
    im_stacked_size_width = np.array([im_stacked_homographed_nonhomo[0,1]-im_stacked_homographed_nonhomo[0,0],im_stacked_homographed_nonhomo[0,3]-im_stacked_homographed_nonhomo[0,2],im_stacked_homographed_nonhomo[0,5]-im_stacked_homographed_nonhomo[0,4],im_stacked_homographed_nonhomo[0,7]-im_stacked_homographed_nonhomo[0,6]])
    
    max_height = np.int(np.amax(im_stacked_size_height))
    max_width = np.int(np.amax(im_stacked_size_width))

    tx = np.amin(im_stacked_homographed_nonhomo[0,:])
    ty = np.amin(im_stacked_homographed_nonhomo[1,:])*-1
    
    # you actually dont need to use M_scale for the pittsburgh city stitching.
    M_scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    M = np.matmul(M_scale,M_translate)

    # TODO fill in the arguments
    # pano_im2 = cv2.warpPerspective(im2, ....)
    # pano_im1 = cv2.warpPerspective(im1, ....)
    
    pano_im1 = cv2.warpPerspective(im1,M,(np.int(max_width*2),max_height+20))
    pano_im2 = cv2.warpPerspective(im2,np.matmul(M,H2to1),(np.int(max_width*2),max_height+20))

    im1_pano_mask = pano_im1 > 0
    im2_pano_mask = pano_im2 > 0

    # TODO
    # should be same line as what you implemented in line 32, in imagestitching
    # im_center_mask = ...
    pano_im_full = pano_im1 + pano_im2

    im_R = pano_im_full * np.logical_not(im1_pano_mask)
    im_L = pano_im_full * np.logical_not(im2_pano_mask)
    # should be same line as what you implemented in line 39, in imagestitching
    # im_center = ...
    im_center = pano_im2 * np.logical_and(im1_pano_mask,im2_pano_mask)
    return im_center + im_R + im_L


def generatePanorama(im1, im2):
    H2to1 = np.load("bestH.npy")
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == "__main__":
    im1 = cv2.imread("../data/incline_L.png")
    im2 = cv2.imread("../data/incline_R.png")
    
    #downscale im1 and im2 for quicker debugging
    #specify scale percentage
    scale_percent = 30
    width_im1 = np.int(im1.shape[1]*scale_percent/100)
    height_im1 = np.int(im1.shape[0]*scale_percent/100)
    dim_im1 = (width_im1,height_im1)
    width_im2 = np.int(im2.shape[1]*scale_percent/100)
    height_im2 = np.int(im2.shape[0]*scale_percent/100)
    dim_im2 = (width_im2,height_im2)
    im1_resize = cv2.resize(im1,dim_im1,interpolation = cv2.INTER_AREA)
    im2_resize = cv2.resize(im2,dim_im2,interpolation = cv2.INTER_AREA)
    print(im1.shape)
    
    locs1, desc1 = briefLite(im1_resize)
    locs2, desc2 = briefLite(im2_resize)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    #imagestitching
    image_stitched = imageStitching(im1_resize,im2_resize,H2to1)
    cv2.imwrite("../results/7_1.jpg",image_stitched)
    np.save("../results/q7_1.npy",H2to1)
    np.save("bestH.npy",H2to1)
    
    # TODO
    # save bestH.npy
    pano_im = generatePanorama(im1_resize, im2_resize)
    plt.imshow(pano_im);plt.show()
    plt.imshow(cv2.cvtColor(pano_im,cv2.COLOR_BGR2RGB));plt.show()
    
    cv2.imwrite("../results/7_3.jpg",pano_im)
    
    #generate panorama for hi_L and hi_R
    im1_hiL = cv2.imread("../data/hi_L.jpg")
    im2_hiR = cv2.imread("../data/hi_R.jpg")

    #downscale im1 and im2 for quicker debugging
    #specify scale percentage
    scale_percent = 30
    width_im1 = np.int(im1_hiL.shape[1]*scale_percent/100)
    height_im1 = np.int(im1_hiL.shape[0]*scale_percent/100)
    dim_im1 = (width_im1,height_im1)
    width_im2 = np.int(im2_hiR.shape[1]*scale_percent/100)
    height_im2 = np.int(im2_hiR.shape[0]*scale_percent/100)
    dim_im2 = (width_im2,height_im2)
    im1_resize = cv2.resize(im1_hiL,dim_im1,interpolation = cv2.INTER_AREA)
    im2_resize = cv2.resize(im2_hiR,dim_im2,interpolation = cv2.INTER_AREA)


    locs1, desc1 = briefLite(im1_resize)
    locs2, desc2 = briefLite(im2_resize)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save("bestH.npy",H2to1)
    
    pano_im = generatePanorama(im1_resize, im2_resize)
    plt.imshow(pano_im);plt.show()
    cv2.imwrite("../results/7_3_hi.jpg",pano_im)
    
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # print(H2to1)
    # cv2.imwrite("../results/panoImg.png", pano_im)
    # cv2.imshow("panoramas", pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
