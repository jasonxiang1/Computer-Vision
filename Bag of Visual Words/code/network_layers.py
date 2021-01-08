import numpy as np
import scipy.ndimage
import os

import skimage.transform

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''
    #--Implementation-- 

    #compile string list of variables
    string_mat = []
    for vgg_name in range(len(vgg16_weights)-2):
        string_mat = np.append(string_mat,vgg16_weights[vgg_name][0])
        
    len_strings = len(string_mat)
    

    #determine if image is a 3 channel image or not, change accordingly if not
    if ((len(x[0,0])<3) or (x.ndim)==2):
        x = x.astype('double')[:,:,np.newaxis]
        img_rgb = np.array(np.stack((x,x,x),axis=2),dtype=np.double)
    else:
        img_rgb = np.array(x,dtype=np.double)
    
    image_normal = img_rgb
    
    x = image_normal
    
    for index in range(len_strings):
        if string_mat[index] == 'conv2d':
            weight = vgg16_weights[index][1]
            bias = vgg16_weights[index][2]
            
            x = multichannel_conv2d(x,weight,bias)
            print("conv2d: DONE - Max - " + str(np.amax(x)))
        elif string_mat[index] == 'relu':
            x = relu(x)
            print("reLU: DONE - Max - " + str(np.amax(x)))
        elif string_mat[index] == 'maxpool2d':
            size = vgg16_weights[index][1]
            
            x = max_pool2d(x,size)
            print("maxpool: DONE - Max - " + str(np.amax(x)))
        elif string_mat[index] == 'linear':
            W = vgg16_weights[index][1]
            b = vgg16_weights[index][2]
            x = linear(x,W,b)
            print("linear: DONE - Max - " + str(np.amax(x)))
    
    return x


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''

    #--Implementation--
    #assuming x size of 224 x 244 if the first convolution or a variant thereafter
    #assuming weight is array of (64,3,3,3) or a variant thereafter
    #assuming bias ia (64) vector or a variant thereafter
    
    #acquire constants of image
    image_rows = len(x)
    image_cols = len(x[0])
    channel_len = len(x[0][0])
    conv_len = len(weight)
    
    
    #define array to store all these convolutions (64, )
    mc_conv_array = np.zeros((image_rows,image_cols,conv_len),dtype=np.double)
    
    for conv_num in range(conv_len):
        temp_biases = bias[conv_num]
        for channel_index in range(channel_len):
            temp_weights = weight[conv_num][channel_index]
            #print(temp_weights)
            
            conv_channel = scipy.ndimage.correlate(x[:,:,channel_index],temp_weights,mode='constant',cval=0)
            #conv_channel = conv_channel + temp_biases
            
            mc_conv_array[:,:,conv_num] += conv_channel
        mc_conv_array[:,:,conv_num] += temp_biases
        
    
    #print("conv2d:DONE - " + str(mc_conv_array.shape))
    
    return mc_conv_array

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''

    #--Implementation
    #relu_array = np.where(x<0,0,x)
    
    relu_array = x
    relu_array[relu_array<0] = 0
    
    #print("ReLU:DONE - " + str(relu_array.shape))

    return relu_array

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''

    #--Implementation--
    
    #define max indices to be divisble by the image dimensions
    #i.e. assuming that the rows and columns of the input image are equal
    in_image_rowscols = len(x)
    in_image_channels = len(x[0,0])
    in_image_elements = in_image_rowscols*in_image_rowscols
    mp_max_index = in_image_rowscols//size
    
    #define blank size
    mp_array = np.zeros((mp_max_index,mp_max_index,in_image_channels),dtype=np.double)
    
    #for loop to max pool across all channels
    for i in range(in_image_channels):
        for j in range(mp_max_index):
            for w in range(mp_max_index):
                mp_array[j,w,i] = np.amax(x[size*j:size*j+size,size*w:size*w+size,i])
    
    mp_out_array = mp_array
    
    
    #print("max_pool2d:DONE - " + str(mp_out_array.shape))

    return mp_out_array

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''

    #--Implementation--
    if x.ndim == 3:
        x = x.transpose((2,0,1))
        
    x_flat = x.flatten()[:,np.newaxis]
    
    y_out = np.matmul(W,x_flat) + b[:,np.newaxis]

    #print("Linear:DONE - " + str(y_out.shape))

    return y_out

