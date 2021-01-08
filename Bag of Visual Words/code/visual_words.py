import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import os

import cv2
import scipy.ndimage
import skimage.color
import glob

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
    
    #define sigmas and filter size to use
    sigma_scales = np.array([1,2,4,8,8*np.sqrt(2)],dtype=float)
    amount_filt = 4
    num_filters = len(sigma_scales)*amount_filt
    

    #check if image is grayscale
    #stack image if is grayscale
    if ((len(image[0,0])<3) or (image.ndim)==2):
        img_rgb = np.stack((image,image,image),axis=2)
    else:
        img_rgb = np.array(image)
        
    #check if the image is normalized between 0 and 1
    if (np.amax(img_rgb)>1.0):
        img_rgb_normal = cv2.normalize(img_rgb,None,alpha=0.0,beta=1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    else:
        img_rgb_normal = np.array(img_rgb)
        
    #convert image to lab colored image
    img_rgb_lab = skimage.color.rgb2lab(img_rgb_normal)

    #define parameters of iamge
    len_rows = len(img_rgb_lab)
    len_cols = len(img_rgb_lab[0])
    len_channels = len(img_rgb_lab[0,0])
    filt_img = np.zeros((len_rows,len_cols,len_channels*num_filters),dtype=float)
    
    #perform multiple filter convolutions
    for i in range(0,len(sigma_scales)):
        #gaussian filter with sigma = sigma_scales[i]
        #padding = reflect values of edge pixel
        filt_img[:,:,i*12] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,0],sigma=sigma_scales[i],mode='reflect')
        filt_img[:,:,i*12+1] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,1],sigma=sigma_scales[i],mode='reflect')
        filt_img[:,:,i*12+2] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,2],sigma=sigma_scales[i],mode='reflect')
    
        #laplacian of gaussian with sigma = sigma_scales[i]
        #padding = relect values of edge pixel
        filt_img[:,:,i*12+3] = scipy.ndimage.gaussian_laplace(img_rgb_lab[:,:,0],sigma=sigma_scales[i],mode='reflect')
        filt_img[:,:,i*12+4] = scipy.ndimage.gaussian_laplace(img_rgb_lab[:,:,1],sigma=sigma_scales[i],mode='reflect')
        filt_img[:,:,i*12+5] = scipy.ndimage.gaussian_laplace(img_rgb_lab[:,:,2],sigma=sigma_scales[i],mode='reflect')
    
        #derivatie of Gaussian in the x direction
        #padding = reflect values of edge pixel
        filt_img[:,:,i*12+6] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,0],sigma=sigma_scales[i],order=(0,1),mode='reflect')
        filt_img[:,:,i*12+7] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,1],sigma=sigma_scales[i],order=(0,1),mode='reflect')
        filt_img[:,:,i*12+8] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,2],sigma=sigma_scales[i],order=(0,1),mode='reflect')
        
        #derivative of Guassian in the y direction
        #padding = reflect values of edge pixel
        filt_img[:,:,i*12+9] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,0],sigma=sigma_scales[i],order=(1,0),mode='reflect')
        filt_img[:,:,i*12+10] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,1],sigma=sigma_scales[i],order=(1,0),mode='reflect')
        filt_img[:,:,i*12+11] = scipy.ndimage.gaussian_filter(img_rgb_lab[:,:,2],sigma=sigma_scales[i],order=(1,0),mode='reflect')
     
    filter_responses = np.array(filt_img)
    
    return filter_responses

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    
    #acquire filter responses from image
    filt_img = extract_filter_responses(image)
    
    #calculate constants
    len_rows = len(filt_img)
    len_cols = len(filt_img[0])
    
    #reshape filt_img to be a (# of pixels,3F) array
    filt_img_reshape = filt_img.reshape((len_rows*len_cols,60))
    
    #declare word map array
    img_wordmap = np.zeros((len_rows*len_cols,1))
    
    #for loop to create word map
    for i in range(len_rows*len_cols):
        temp = scipy.spatial.distance.cdist(filt_img_reshape[i,:][np.newaxis,:],dictionary,'euclidean')
        img_wordmap[i] = np.argmin(temp)
    
    #reshape wordmap to be the image
    wordmap = img_wordmap.reshape(len_rows,len_cols)
    
    return wordmap

def get_harris_points(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (alpha, 2) that contains interest points
    '''

    # ----- TODO -----
    
    #convert image (if 3D) to grayscale image
    image_float = np.array(image,dtype=np.float32)
    
    if(image_float.ndim==2):
        img_grayscale = image_float
    else:
        img_grayscale = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
    
    img_gray = np.asarray(img_grayscale)
    
    #assign variables to total rows and columns of image
    len_rows = len(img_gray)
    len_cols = len(img_gray[0])
    
    #define sobel filter window and run image through x and y sobels
    kern_size_sobel = 3
    img_gray_xsob = cv2.Sobel(img_gray,cv2.CV_32F,dx=1,dy=0,ksize=kern_size_sobel)
    img_gray_ysob = cv2.Sobel(img_gray,cv2.CV_32F,dx=0,dy=1,ksize=kern_size_sobel)

    #define covariance matrix window
    ksize_sob_conv = 3
    
    #define border for sobel images to computer covariance matrix
    bord_buffer = np.int(ksize_sob_conv/2)
    xsob_bordered = cv2.copyMakeBorder(img_gray_xsob, bord_buffer, bord_buffer, bord_buffer, bord_buffer,cv2.BORDER_REPLICATE)
    ysob_bordered = cv2.copyMakeBorder(img_gray_ysob, bord_buffer, bord_buffer, bord_buffer, bord_buffer,cv2.BORDER_REPLICATE)

    #computer w/ border rows and columns of output sobel
    sobel_rows = len(xsob_bordered)
    sobel_cols = len(xsob_bordered[0])
    
    #define R matrix to store r values
    R_2D_vector = np.zeros_like(img_gray)
    
    #for loop to computer covariance matrix
    for i in range(bord_buffer,sobel_rows-bord_buffer):
        for j in range(bord_buffer,sobel_cols-bord_buffer):
            #index xsobel and ysobel windows
            xsob_window = xsob_bordered[i-bord_buffer:i+bord_buffer+1,j-bord_buffer:j+bord_buffer+1]
            ysob_window = ysob_bordered[i-bord_buffer:i+bord_buffer+1,j-bord_buffer:j+bord_buffer+1]
            
            #compute each elements in the covariance matrix
            h_covar_11 = np.sum(xsob_window*xsob_window,dtype=float)
            h_covar_12 = np.sum(xsob_window*ysob_window,dtype=float)
            h_covar_21 = np.sum(ysob_window*xsob_window,dtype=float)
            h_covar_22 = np.sum(ysob_window*ysob_window,dtype=float)
            
            #compute det(H) = ad-bc
            det_H = h_covar_11*h_covar_22 - h_covar_12*h_covar_21
            
            #compute trace of H = a + d
            trace_H = h_covar_11 + h_covar_22
            
            #compute R
            R_2D_vector[(i-bord_buffer),(j-bord_buffer)] = det_H - k*np.square(trace_H)
            
    #reshape r matrix to 1d array to organize
    R_1D_vector = R_2D_vector.flatten()
    
    #use argsort to find the indices of max vaues of r matrix
    r_matrix_topindices = np.array(np.argsort(R_1D_vector)[::-1])
    print(r_matrix_topindices.dtype)
    return_value_indices = r_matrix_topindices[:np.int(alpha)]
        
    #organize top indices matrix to top alpha values where other values are null
    r_matrix_topalpha = np.arange(len_rows*len_cols)
    r_matrix_topalpha = np.where(np.isin(r_matrix_topalpha,return_value_indices),1,0)
    
    #reshape matrix to 2D array
    r_matrix_topalpha_2D = r_matrix_topalpha.reshape((len_rows,len_cols))
    
    #output indices and organize output
    topalpha_y,topalpha_x = np.nonzero(r_matrix_topalpha_2D)
    topalpha_y = topalpha_y[:,np.newaxis]
    topalpha_x = topalpha_x[:,np.newaxis]
    topalpha_indices = np.concatenate((topalpha_y,topalpha_x),axis=1).reshape((np.int(alpha),2))
    
    
    return topalpha_indices



def compute_dictionary_one_image(args):
    '''
    Extracts alpha samples of the dictionary entries from an image. Use the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    # ----- TODO -----
    

    #load image
    img = cv2.imread('../data/' + str(image_path))
    img_rgb = np.asarray(img)
    
    #compute filtered responses for image
    filt_img_map = extract_filter_responses(img_rgb)
    filt_img = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    filt_img = np.asarray(filt_img,dtype=np.float32)
        
    #computer harris alpha points of interest
    img_alpha_topindices = get_harris_points(filt_img,alpha)
    
    #computer (alpha, 3F) sampled responses
    sampled_response = filt_img_map[img_alpha_topindices[:,0],img_alpha_topindices[:,1],:]

    #save sampled responses and index to tempfile
    indices_repeat = np.repeat(i,len(sampled_response))
    indices_repeat = indices_repeat[:,np.newaxis]
    
    #save outputs to temp file in current directory
    np.save('../data/temp_file_folder/' + str(i) + "_" + str(image_path[-20:-4]),np.append(sampled_response, indices_repeat,axis=1))
    
    
    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''
    import util
    #os.environ["OPENBLAS_MAIN_FREE"] = "1"
    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    cpu_cores = num_workers
    
    #define null row of alpha points and indices
    alpha_points = np.zeros((1,60))
    alpha_indices = np.zeros((1))
    
    
    #compute length of data file arrays
    test_len = len(train_data['files'])
    
    #define alpha value
    alpha = 500
    
    #for loop to acquire filtered responses at alpha points
    for i in range(test_len):
        print('iteration: ' + str(i))
        temp_image_path = train_data['files'][i]
        temp_index = train_data['labels'][i]
        
        args = np.array([temp_index, alpha, temp_image_path])
    
        
        #pass args to compute_dictionary_one_image
        
        
        compute_dictionary_one_image(args)
        
        
        
        #load temp file
        load_temp_file = np.load('../data/one_image_temp_file.npy')
        
        #append temp file to dictionary
        alpha_points = np.append(alpha_points,load_temp_file[:,:-1],axis=0)
        alpha_indices = np.append(alpha_indices,load_temp_file[:,-1])
        
    alpha_points = alpha_points[1:,:]
    alpha_indices = alpha_indices[1:]
    
    #save points to final dictionary image
    np.save('../data/dictionary.npy',alpha_points)
    np.save('../data/dictionary_indices.npy',alpha_indices)
    
    
    image_paths_col = train_data['files'][:,np.newaxis]
    index_col = train_data['labels'][:,np.newaxis]
    alpha_col = np.repeat(alpha,test_len)[:,np.newaxis]
    args = np.concatenate((index_col,alpha_col,image_paths_col),axis=1)
    
    p = multiprocessing.Pool(num_workers-2)
    p.map(compute_dictionary_one_image,args)
    p.close()
    p.join()

    
    #load training data
    imglist = []
    for filename in glob.glob('../data/temp_file_folder/*.npy'):
        imglist.append(filename)

    imglist.sort()
    
    for harris_path in imglist:
        harris_array = np.load(harris_path)
        
        alpha_points = np.append(alpha_points,harris_array[:,:-1],axis=0)
        alpha_indices = np.append(alpha_indices,harris_array[:,-1])
        
    alpha_points = alpha_points[1:,:]
    alpha_indices = alpha_indices[1:]
    
    print('starting KMeans now!')
    K = 300
    kmeans = sklearn.cluster.KMeans(n_clusters = K,n_jobs = 15).fit(alpha_points)
    dictionary = kmeans.cluster_centers_
    
    
    #save outputs to temp file in current directory
    np.save('../data/dictionary.npy', dictionary)

    pass
