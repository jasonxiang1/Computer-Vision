import numpy as np
import cv2

#import matplotlib to analyze images
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(
        im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    cv2.imshow("Pyramid of image", im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    """
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    """
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]

    '''
    Algorithm:
    create for loop spanning across L-1 levels of the gaussian pyramid
    subtract the current gaussian image from the previous one
    append on dog to the dog array
    np.stack the pyramid to be the same structure as gaussian
    '''
    for i in DoG_levels:
        DoG_temp = gaussian_pyramid[:, :, i-1] - gaussian_pyramid[:, :, i]
        #DoG_temp[DoG_temp<0] = 0

        DoG_pyramid.append(DoG_temp)

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    """
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    """
    principal_curvature = None

    gxx = []
    gyy = []
    gxy = []
    gyx = []

    for l in range(DoG_pyramid.shape[2]):
        # Computing 1st order derivatives
        gx = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )
        gy = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )

        # Computing 2nd order derivatives
        gxx.append(
            cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gxy.append(
            cv2.Sobel(gx, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyx.append(
            cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyy.append(
            cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )

    gxx = np.stack(gxx, axis=-1)
    gxy = np.stack(gxy, axis=-1)
    gyx = np.stack(gyx, axis=-1)
    gyy = np.stack(gyy, axis=-1)

    principal_curvature = np.divide(
        np.square(np.add(gxx, gyy)), (np.multiply(gxx, gyy) - np.multiply(gxy, gyx))
    )

    return principal_curvature


def getLocalExtrema(
    DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12
):
    """
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    """
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here

    #set constants
    num_neigh = 2
    pad_size = np.int(num_neigh/2)
    num_channels = 2
    len_rows = len(DoG_pyramid)
    len_cols = len(DoG_pyramid[0])
    len_channels = len(DoG_levels)

    #declare temp DoG table
    locsDoG_table = np.zeros((1,4),dtype=int)

    #pad DoG_image
    DoG_pyramid_padded = cv2.copyMakeBorder(DoG_pyramid,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=0)

    #build for loop
    for i in range(pad_size,(len_rows+pad_size)):
        for j in range(pad_size,(len_cols+pad_size)):
            for w in range(1,len_channels):
                #create comparison window
                win_mat = DoG_pyramid_padded[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1,w-1:w+2]
                #print(win_mat.shape)

                #find max value in window
                max_val = np.amax(np.abs(win_mat))

                #calculate max value
                max_index = np.where(np.abs(win_mat) == max_val)

                if len(max_index[0]) != 1:
                    continue
                elif len(max_index[1]) != 1:
                    continue
                elif len(max_index[2]) != 1:
                    continue
                channel_offset = np.int(max_index[2])
                row_offset = np.int(max_index[0])
                col_offset = np.int(max_index[1])

                #check max_val with DoG threshold
                # print(str(row_offset) + " " + str(col_offset) + " " + str(channel_offset))
                # print(str(i) + " " + str(j) + " " + str(w))               
                # print(win_mat)
                
                #check if index is already in the array
                
                
                if (max_val <th_contrast):
                   continue
                elif (np.abs(principal_curvature[i-2*pad_size+row_offset,j-2*pad_size+col_offset,w-1+channel_offset]))> 12:
                    continue
                elif np.any(np.all(np.isin(locsDoG_table,np.array([j-2*pad_size+col_offset,i-2*pad_size+row_offset,principal_curvature[i-2*pad_size+row_offset,j-2*pad_size+col_offset,w-1+channel_offset],w-1+channel_offset])),axis=1)):
                    continue
                else:
                    #print(np.array(max_index))
                    locs_index = np.array([j-2*pad_size+col_offset,i-2*pad_size+row_offset,principal_curvature[i-2*pad_size+row_offset,j-2*pad_size+col_offset,w-1+channel_offset],w-1+channel_offset])
                    locs_index = locs_index[np.newaxis,:]
                    locsDoG_table = np.append(locsDoG_table,locs_index,axis=0)
                    
    
    #when i have threshold and threshold contrast
    
    #sort array based on top curvature points
    #sort curve list in desending curvature order
    image_ext_sorted = locsDoG_table[locsDoG_table[:,2].argsort()][::-1]
    
    #determine percentage of points to use to highlight on image
    #num_curv = 500
    perc_curv = 1.0
    am_ext = np.int(len(image_ext_sorted)*perc_curv)
    image_ext_perc = image_ext_sorted[:am_ext,:]
    
    #output array without the 3rd column
    locsDoG = np.transpose(np.vstack((image_ext_perc[:,0],image_ext_perc[:,1],image_ext_perc[:,3])))
    
    
    
    return locsDoG


def DoGdetector(
    im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12
):
    """
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    """
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    
    #check if image if grayscale
    if (im.ndim==3):
        image_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    else:
        image_gray = np.array(im)
    
    #check if image is normalized
    if np.amax(image_gray) > 1:
        image_norm = cv2.normalize(image_gray,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        image_norm = np.array(image_gray)
        
    #create gaussian pyramid of image
    image_pyr = createGaussianPyramid(image_norm,sigma0,k,levels)
    
    #create DoG
    image_DoG,image_DoG_levels = createDoGPyramid(image_pyr,levels)
    
    #create curvature array
    image_curv = computePrincipalCurvature(image_DoG)
    
    #determine local extrema
    image_ext = getLocalExtrema(image_DoG,image_DoG_levels,image_curv,th_contrast,th_r)

    
    #highlight and plot interest points 
    image_rgb = cv2.cvtColor(image_gray,cv2.COLOR_GRAY2RGB)
    image_rgb[np.array(image_ext[:,1],dtype=int),np.array(image_ext[:,0],dtype=int),:] = [0,255,0]
    locsDoG = np.array(image_ext,dtype=np.int32)
    gauss_pyramid = image_pyr
    
    return locsDoG, gauss_pyramid


if __name__ == "__main__":
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    #im = cv2.imread("../data/pf_stand.jpg")
    im = cv2.imread("../data/model_chickenbroth.jpg")
    #im = cv2.imread("../data/chickenbroth_01.jpg")
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    #breakpoint()
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

