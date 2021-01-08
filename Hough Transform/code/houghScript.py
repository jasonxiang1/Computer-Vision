import cv2
import glob
import numpy as np
import os.path as osp

from GaussianKernel import Gauss2D

from myImageFilter import myImageFilter
from myImageFilterX import myImageFilterX
from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines
from myHoughLineSegments import myHoughLineSegments

from matplotlib import pyplot as plt
import matplotlib


print('You are using opencv verstion: {}'.format(cv2.__version__))
THIS_PATH = osp.dirname(__file__)

datadir     = '../data'    # the directory containing the images
resultsdir  = '../results/'  # the directory for dumping results

# parameters | Do NOT change chese parameters
sigma     = 1.5
rhoRes    = 1
thetaRes  = 1
nLines    = 50
# end of parameters

imglist = []
for filename in glob.glob(datadir+'/*.jpg'):
	imglist.append(filename)

imglist.sort()

for img_name in imglist:
    img = cv2.imread(img_name)
    img_rgb = np.asarray(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = np.asarray(img_gray)
    hfilt = Gauss2D((5,5),sigma)
    
	# Q3.1 Convolution
    # you can change the parameters for Gauss2D
    #plt.imshow(img_gray, cmap="gray");plt.show()
    Im1 = myImageFilter(img_gray, hfilt)
    #plt.imshow(Im1,cmap="gray");plt.show()
    
    #plt.imshow(img_gray, cmap="gray");plt.show()
    Im1X = myImageFilterX(img_gray, hfilt)
    # cv2.normalize(Im1X,Im1X, 0, 1, cv2.NORM_MINMAX)
    #plt.imshow(Im1X,cmap="gray");plt.show()

    
	# Q3.3
    ImEdge,Io,Ix,Iy = myEdgeFilter(img_gray, sigma)
    #plt.imshow(ImEdge,cmap="gray");plt.show()
    

    Thresh = 0.2 # change it as needed
    ThrImEdge = 1.0 * (ImEdge > Thresh)
    # plt.imshow(ThrImEdge,cmap="gray");plt.show()

 	# Q3.4
    H,rhoScale,thetaScale = myHoughTransform(ThrImEdge, rhoRes, thetaRes)
    #plt.imshow(H);plt.show()
    


 	# Q3.5
    Peakrhos, Peakthetas = myHoughLines(H, nLines)

    
    rhos = Peakrhos * rhoRes
    rho_max = np.max(rhos)
    thetas = Peakthetas * thetaRes
    #print(rhos.shape)
    #print(thetas.shape)
    #derive (x1,y1) and (x2,y2) and plot on image based on m and b
    for i in range(0,len(thetas)-1):
        
        a = np.cos(np.deg2rad(thetas[i]))
        b = np.sin(np.deg2rad(thetas[i]))
        
        x0 = a * rhos[i]
        y0 = b * rhos[i]
        
        x1 = int(x0 + (500*rho_max)*(-b))
        y1 = int(y0 + (500*rho_max)*a)
        
        x2 = int(x0 - (500*rho_max)*(-b))
        y2 = int(y0 - (500*rho_max)*a)
        
        cv2.line(img_gray,(x1,y1),(x2,y2),(255,0,0),1)
        
        
    plt.imshow(img_gray);plt.show()
    



    # Q3.6
    OutputImage  = myHoughLineSegments(img_rgb, ThrImEdge, Peakrhos, Peakthetas, rhoScale, thetaScale)
    plt.imshow(cv2.cvtColor(OutputImage, cv2.COLOR_BGR2RGB));plt.show()

 	# everything below here just saves the outputs to files
    fname = resultsdir + img_name.split('/')[-1][:-4]+'_01Blur.png'
    matplotlib.image.imsave(fname,  Im1,cmap="gray")

    fname = resultsdir + img_name.split('/')[-1][:-4]+'_02Blur.png'
    matplotlib.image.imsave(fname,  Im1X,cmap="gray")

    fname = resultsdir + img_name.split('/')[-1][:-4]+'_03edge.png'
    matplotlib.image.imsave(fname, ImEdge,cmap="gray")

    fname = resultsdir + img_name.split('/')[-1][:-4]+'_04threshold.png'
    matplotlib.image.imsave(fname,   ThrImEdge,cmap="gray")

 	# plot hough space, brighter spots have higher votes
    fname = resultsdir + img_name.split('/')[-1][:-4]+'_05Hough.png'
    matplotlib.image.imsave(fname, H)

    fname = resultsdir + img_name.split('/')[-1][:-4]+'_06Final.png'
    matplotlib.image.imsave(fname, cv2.cvtColor(OutputImage, cv2.COLOR_BGR2RGB))


