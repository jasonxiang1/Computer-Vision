import cv2
import numpy as np
import numpy.matlib as npm
import math

from GaussianKernel import Gauss2D
from myImageFilter import myImageFilter
from myImageFilterX import myImageFilterX



def myEdgeFilter(img0, sigma):
    
    #declare sobel filters
    ysobel = (1/8)*np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    xsobel = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    
    #create gaussian matrix
    hsize = 2*math.ceil(3*sigma)+1
    gfilt = Gauss2D((hsize,hsize),sigma)

    #perform gaussian blur on image
    img_gray_blurred = myImageFilterX(img0,gfilt)
    img_gray_blurred = img_gray_blurred[:,:,np.newaxis]
    
    #perform sobel filter convulution on image in both directions
    Ix = myImageFilterX(img_gray_blurred,xsobel)
    Ix = Ix[:,:,np.newaxis]
    Iy = myImageFilterX(img_gray_blurred,ysobel)
    Iy = Iy[:,:,np.newaxis]

    #calculate Im using pythagorem theorem equation
    Im = np.int_(np.sqrt(np.square(Ix)+np.square(Iy)))
    
    #calculate Io_int using arctan
    Io_float = np.arctan(np.divide(Iy,Ix,out=(np.ones_like(Iy)*math.pi/2),where=Ix!=0))*(180/math.pi)
    #for Iy/Ix values that are towards negative infinity, change to -90 degrees
    #Io_float[(np.int_(np.absolute((Io_float-90)))==0) & ((Io_float) <0)] = -1*90
    #convert Io_int from float to integeter to output as a grayscale image
    Io_int = np.int_(Io_float)
    #translate range of Io_int from [-90,90] to [0,180]
    Io_int[Io_int<0] = 180 + Io_int[Io_int<0]
    
    i = 0
    j = 0
    Im_NMS = np.zeros_like(Im)
    bin_matrix = np.zeros(4)
    half_ang = 45/2
    
    for i in range(0,len(Im)-1):
        for j in range(0,len(Im[0])-1):
            if((Io_int[i,j]>=0) & (Io_int[i,j]<(half_ang))):
                Im_NMS[i,j] = np.where(((Im[i,j] <= Im[i,j+1]) or (Im[i,j] <= Im[i,j-1])), 0, 1)
                bin_matrix[0] +=1
            elif((Io_int[i,j]>=(45-half_ang)) & (Io_int[i,j]<(90-half_ang))):
                Im_NMS[i,j] = np.where(((Im[i,j] <= Im[i+1,j+1]) or (Im[i,j] <= Im[i-1,j-1])), 0, 1)
                bin_matrix[1] +=1
            elif((Io_int[i,j]>=(90-half_ang)) & (Io_int[i,j]<(135-half_ang))):
                Im_NMS[i,j] = np.where(((Im[i,j] <= Im[i+1,j]) or (Im[i,j] <= Im[i-1,j])), 0, 1)
                bin_matrix[2] +=1
            elif((Io_int[i,j]>=(135-half_ang)) & (Io_int[i,j]<(180-half_ang))):
                Im_NMS[i,j] = np.where(((Im[i,j] <= Im[i-1,j+1]) or (Im[i,j] <= Im[i+1,j-1])), 0, 1)
                bin_matrix[3] +=1
            elif((Io_int[i,j]>=(180-half_ang))):
                Im_NMS[i,j] = np.where(((Im[i,j] <= Im[i,j+1]) or (Im[i,j] <= Im[i,j-1])), 0, 1)
                bin_matrix[0] +=1
    
    Img1_nonthreshold = Im_NMS*Im
    Img1 = Img1_nonthreshold/np.max(Img1_nonthreshold)
    Io = Im_NMS*Io_int
    
    Img1 = np.squeeze(Img1)
    
    return Img1,Io,Ix,Iy





