import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

    # plt.imshow(im1)
    # plt.show()
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.show()
    #find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    '''
    algorithm:
    initialize with the first row belonging to the 0th row
    initialize with all corner parameters before the first loop
    start at the 0th row (code is already sorted top to bottom of the image)
    contain an same row index starting at 1
    
    check bounds of the current image with the last image
    if different then += to the row count
    '''

    #define length of the bboxes
    len_bboxes = len(bboxes)


    #define first bounding box
    tlr_comp, tlc_comp, brr_comp, brc_comp = bboxes[0,:]

    #initialize row count array
    #first element is zero for the 0th row
    row_collection = np.zeros(1)
    current_row = 0


    for i in range(1,len_bboxes):
        tlr_curr, tlc_curr, brr_curr, brc_curr = bboxes[i,:]
        if tlr_curr < brr_comp and brr_curr > tlr_comp:
            row_collection = np.append(row_collection, current_row)
        else:
            #this means there is a change of row
            current_row += 1
            row_collection = np.append(row_collection,current_row)
        tlr_comp, tlc_comp, brr_comp, brc_comp = [tlr_curr, tlc_curr, brr_curr, brc_curr]


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################

    #set up for loop through inlier boxes
    #acuiqre the longer side between the two lengths
    #crop out box as a square -> pad by 1 -> transpose -> flatten
    #store cropped image into dataset list

    #create dataset list that is 32x32
    desired_flat_size = np.int(32*32)
    flat_dataset = np.zeros((1,desired_flat_size))
    pad_size = 13


    for i in bboxes:

        #make sure to add before and after not just after
        #apply padding to i
        row_top = np.int(i[0]-pad_size)
        row_bot = np.int(i[2]+pad_size+1)
        col_left = np.int(i[1]-pad_size)
        col_right = np.int(i[3]+pad_size+1)

        image_cropped = bw[row_top:row_bot,col_left:col_right]
        #plt.imshow(image_cropped)

        #find image and pad to be a square crop
        len_rows = len(image_cropped)
        len_cols = len(image_cropped[0])
        val_pad = np.int(np.abs((len_rows-len_cols)/2))

        if len_rows>len_cols:
            image_square = skimage.util.pad(image_cropped, ((0,0),(val_pad,val_pad)),mode='constant',constant_values=0)
        elif len_cols > len_rows:
            image_square = skimage.util.pad(image_cropped, ((val_pad,val_pad),(0,0)), mode='constant', constant_values=0)

        #invert image
        image_square = ~image_square

        #image_cropped = skimage.transform.rescale(np.float32(image_square),32/len_rows,anti_aliasing=False)
        image_cropped = skimage.transform.resize(np.float32(image_square), (32,32),anti_aliasing=True)
        image_cropped = np.transpose(image_cropped)
        #image_cropped = np.where(image_cropped<0.95,0,1)
        image_cropped[image_cropped<0.95] = image_cropped[image_cropped<0.95]*0.6

        # #only for spyder ide
        # plt.imshow(image_cropped);plt.show()

        #flatten image and store in dataset
        image_flat = image_cropped.flatten()[np.newaxis,:]

        #store flatten image into dataset
        flat_dataset = np.append(flat_dataset, image_flat,axis=0)

    flat_dataset = flat_dataset[1:,:]

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################

    #run through data
    h1 = forward(flat_dataset, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    letters_predict_index = np.argmax(probs,axis=1)

    letters_predict = letters[letters_predict_index]
    print(letters_predict)
    