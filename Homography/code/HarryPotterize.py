import numpy as np
import cv2
import skimage.io
import planarH
import BRIEF



# warp harry potter onto cv desk image
# save final image as final_image
# TODO

#load images for question 6.1
img_cv_desk = cv2.imread("../data/pf_desk.jpg")
img_cv_ref = cv2.imread("../data/pf_scan_scaled.jpg")
img_hp_cover = cv2.imread("../data/hp_cover.jpg")

#run brieflite on all images
locs_cv_desk,desc_cv_desk = BRIEF.briefLite(img_cv_desk)
locs_cv_ref,desc_cv_ref = BRIEF.briefLite(img_cv_ref)
#locs_hp_cover,desc_hp_cover = BRIEF.briefLite(img_hp_cover)

#run matching with hp cover image with the cv desk image
matches_1 = BRIEF.briefMatch(desc_cv_desk,desc_cv_ref)

#compute homography matrix with the outputted matches
#homography intends to map the coordinates from the harry potter cover image to the cv desk image
H_1 = planarH.ransacH(matches_1,locs_cv_desk,locs_cv_ref,num_iter=20000,tol=3)

#warp harry potter cover image based on homography matrix using cv2 warpPerspective
#pass images and homography matrix to compositeH function

#reshape harry potter image before passing to compositeH
img_hp_cover_scaled = cv2.resize(img_hp_cover,(img_cv_ref.shape[1],img_cv_ref.shape[0]), interpolation=cv2.INTER_AREA)

final_img_1 = planarH.compositeH(H_1,img_cv_desk,img_hp_cover_scaled)

num_iterations = np.array([50, 500, 5000, 20000, 40000, 60000])
num_tots = np.array([1,2,3,4,7])

for i in num_iterations:
    for j in num_tots:
        img_cv_desk = cv2.imread("../data/pf_desk.jpg")
        #compute new H_1
        print("Iteration: num_iter: ",i,"  num_tots: ",j)
        H_1ij = planarH.ransacH(matches_1,locs_cv_desk,locs_cv_ref,num_iter=i,tol=j)
        #compute new final image
        final_img_1 = planarH.compositeH(H_1ij,img_cv_desk,img_hp_cover_scaled)

