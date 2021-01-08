import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words

import cv2
import glob


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    
    
    os.environ["OPENBLAS_MAIN_FREE"] = "1"

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----

    #declare SPM_layer_num
    SPM_layer_num = 3

    #compute length of data file arrays
    test_len = len(train_data['files'])
    dict_size = len(dictionary)
    hist_len = (dict_size*(4**(SPM_layer_num)-1))/3
    

    
    image_paths_col = train_data['files'][:,np.newaxis]
    index_col = train_data['labels'][:,np.newaxis]
    layer_num_col = np.repeat(SPM_layer_num,test_len)[:,np.newaxis]
    args = np.concatenate((image_paths_col,index_col,layer_num_col),axis=1)
    print(args.shape)
    
    p = multiprocessing.Pool(num_workers-2)
    p.map(custom_file_visual_words,args)
    p.close()
    p.join()
    

    #create arrays for labels and img_histograms
    label_array = []
    feature_array = np.zeros((1,np.int32(hist_len)))
    
    #load files
    #load training data
    imglist = []
    for filename1 in glob.glob('../data/feat_hist_traindata_folder/*.npz'):
        imglist.append(filename1)

    imglist.sort()
    print(len(imglist))
    count = 0
    #populate label and 
    for npz_path in imglist:
        count += 1
        #print(count)
        npz_load = np.load(npz_path)
        temp_feat_array = npz_load['feat_histogram']
        temp_label_array = npz_load['label']

        feature_array = np.append(feature_array,temp_feat_array[np.newaxis,:],axis=0)
        label_array = np.append(label_array,np.int32(temp_label_array))

    features = feature_array[1:,:]
    labels = label_array
    
    print(features.shape)
    print(labels.shape)

    np.savez('trained_system.npz',features=features,labels=labels,dictionary=dictionary,SPM_layer_num=SPM_layer_num)

    pass


def custom_file_visual_words(args):
    
    '''
    algorithm:
        w/ image path, load image
        pass image through get_visual_words function
        save file to folder w/ generated name
    
    notes:
        1. does not matter if the image is grayscale or rgb b/c filt_responses
        function will check for that feature
        
    '''
    
    #break down args into its components
    image_path, label, SPM_layer_num = args
    
    #load dictionary
    dictionary = np.load("dictionary.npy")
    
    #compute dictionary size
    dict_size = np.int32(len(dictionary))
    
    #load image
    img = cv2.imread('../data/' + str(image_path))
    img_rgb = np.asarray(img)
    
    #run image through visual_wordmap
    word_img = visual_words.get_visual_words(img_rgb,dictionary)
    
    #run wordmap through hist SPM function
    feat_histogram = get_feature_from_wordmap_SPM(word_img,SPM_layer_num,dict_size)
    
    #save file and labels to .npz file in determined folder
    np.savez('../data/feat_hist_traindata_folder/' "hist_featfile_" + str(image_path[-20:-4]) + ".npz",feat_histogram=feat_histogram,label=label)
    
    #print test
    print('DONE WITH ONE')
    
    pass


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    '''
    algorithm:
        load test images to file
        run histogram of test images
        get image of most 
    '''
    #assigned each type of trained system to variables
    features_trained = trained_system['features']
    labels_trained = trained_system['labels']
    dictionary_trained = trained_system['dictionary']
    SPM_layer_num = trained_system['SPM_layer_num']
    
    #declare size of dictionary
    dict_size = len(dictionary_trained)
        
    #assign test_data arrays to variables
    test_filepaths = test_data['files']
    test_labels = test_data['labels']
    
    #declare size of test_data
    len_test_data = len(test_filepaths)
    
    #define confusion matrix
    conf_matrix = np.zeros((8,8))
    
    for i in range(len_test_data):
        print(i)
        #test with one image
        image1 = cv2.imread('../data/' + str(test_filepaths[i]))
        label1 = test_labels[i]
        image1_rgb = np.asarray(image1)
    
        #run image through visual words
        image1_wordmap = visual_words.get_visual_words(image1_rgb,dictionary_trained)   
    
        #run word map through SPM function to get histogram
        image1_features = get_feature_from_wordmap_SPM(image1_wordmap, SPM_layer_num, dict_size)
        
        #compare image1 features with trained set histogram across 1000 images
        dist_set_image1 = distance_to_set(image1_features, features_trained)
        
        #find the max value from the dist_set variable
        index_maxvalue = np.argmax(dist_set_image1)
        
        #acquire the label value of the trained image chosen
        chosen_trained_label = labels_trained[index_maxvalue]
        
        if chosen_trained_label == label1:
            indice = np.int32(label1)
            conf_matrix[indice,indice] += 1
        else:
            row_indice = np.int32(chosen_trained_label)
            col_indice = np.int32(label1)
            conf_matrix[row_indice,col_indice] += 1

    

    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix.flatten())
    conf = conf_matrix
        
    return conf,accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----

    pass


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    
    '''
    with one image, compare the minimum of two histograms
    no need to manipulate histograms internally
    with the histogram of minimums, sum across all bins,
    now have a total of minimums for one image
    if two images are equal, then summing the minimuim will equal to 1
    
    algorithm:
        1. get the image length of trained histogram
        2. extend word map histogram by the length of the trained histogram
        3. perform minimum of all histograms
        4. reshape output to be length of images tall
        5. sum across rows
    '''
    #define array lengths for use later
    len_hist = len(word_hist)
    len_images = len(histograms)
    
    #tile/extend word_hist array to the length of the images
    word_hist_flat = np.tile(word_hist,len_images)
    
    #flatten histogram array
    histogram_array_flat = histograms.flatten()
    
    #calculate minimum between two flatten arrays
    sim_array = np.minimum(word_hist_flat,histogram_array_flat)
    sim_array = np.reshape(sim_array,(len_images,len_hist))
    
    #sum columns of sim together
    sim = np.sum(sim_array,axis=1)
    
    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    hist_array,bins_array = np.histogram(wordmap,dict_size,range=(0,dict_size),density=True)
    
    hist = hist_array
    
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----

    layer_num = np.int32(layer_num)
    dict_size = np.int32(dict_size)    

    #define length of histogram output
    hist_length = np.int32((dict_size*((4**layer_num)-1))/3)
    
    #define output histogram from calculated length
    hist_array = np.zeros(hist_length)
    
    #for loop to fill histogram output with values
    count = 0
    for i in np.arange(layer_num):
        L = 2**(i)
        
        #define the weights based on the current iteration layer
        if i==0 or i==1:
            weight = 2**(np.float32(-L))
        else:
            weight = 2**(np.float32(i-L-1))
        
        #split image row-wise first
        wordmap_rowsplit = np.array_split(wordmap,L,axis=0)
        for j in np.arange(L):
            wordmap_rowcolsplit_temp = np.array_split(wordmap_rowsplit[j],L,axis=1)
            for w in np.arange(L):
                temp_hist,temp_bins = np.histogram(wordmap_rowcolsplit_temp[w],dict_size,range=(0,dict_size),density=True)
                count += 1
                hist_array[(count-1)*dict_size:count*dict_size] = weight * temp_hist
                
    return hist_array