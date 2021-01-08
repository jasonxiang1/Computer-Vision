import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from skimage import io
import cv2

def evaluate_deep_extractor(img, vgg16):
    '''
    Evaluates the deep feature extractor for a single image.

    [input]
    * image: numpy.ndarray of shape (H,W,3)
    * vgg16: prebuilt VGG-16 network.

    [output]
    * diff: difference between the two feature extractor's result
    '''
    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(img)
    
    feat = network_layers.extract_deep_feature(np.transpose(img_torch.numpy(), (1,2,0)), vgg16_weights)
    
    with torch.no_grad():
        # vgg_test_class = torch.nn.Sequential(*list(vgg16.classifier.children())[0])
        # vgg_test_feat = vgg_test_class(img_torch[None, ])
        
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    print(np.amax(abs(feat)))
    print(feat.shape)
    print(np.amax(abs(vgg_feat_feat.numpy())))
    print(vgg_feat_feat.numpy().shape)
    print(np.abs(vgg_feat_feat.numpy() - feat).shape)
    
    
    return np.sum(np.abs(vgg_feat_feat.numpy()[:,np.newaxis] - feat))


def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----
    
    image_paths_col = train_data['files'][:,np.newaxis]
    index_col = train_data['labels'][:,np.newaxis]
    vgg16_col = np.repeat(vgg16,len(image_paths_col))[:,np.newaxis]
    args = np.concatenate((index_col,image_paths_col,vgg16_col),axis=1)
    
    p = multiprocessing.Pool(num_workers-3)
    output = p.map(get_image_feature,args)
    p.close()
    p.join()
    
    np.savez("trained_system_deep.npz",output=output,index_col=index_col)
    
    pass
    

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    
    # ----- TODO -----
    #load trained system information
    trained_sys = np.load("trained_system_deep.npz")
    trained_output = trained_sys['output']
    trained_labels = trained_sys['index_col']
    
    #load test system information
    test_image_paths = test_data['files'][:,np.newaxis]
    test_images_index = test_data['labels'][:,np.newaxis]
    vgg16_col = np.repeat(vgg16,len(test_image_paths))[:,np.newaxis]
    args = np.concatenate((test_images_index,test_image_paths,vgg16_col),axis=1)
    
    p = multiprocessing.Pool(num_workers-3)
    test_feat_indi = p.map(get_image_feature,args)
    p.close()
    p.join()
    
    conf_matrix = np.zeros((8,8))
    
    for i in range(len(test_image_paths)):
        print(i)
        sum_neg_vect = distance_to_set(test_feat_indi[i],np.squeeze(trained_output,axis=2))
        max_clust_index = np.argmax(sum_neg_vect)
        trained_index = trained_labels[max_clust_index]
        test_index = test_images_index[i]
        if test_index == trained_index:
            conf_matrix[test_index,test_index] += 1
        else:
            conf_matrix[test_index,trained_index] +=1
    
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    
    return conf_matrix,accuracy


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''

    # ----- TODO -----
    
    x = image
    
    #define mean and std
    mean = np.array([0.485,0.456,0.406],dtype=np.double)
    std = np.array([0.229,0.224,0.225],dtype=np.double)  
    
    #determine if image is a 3 channel image or not, change accordingly if not
    if ((len(x[0,0])<3) or (x.ndim)==2):
        x = x.astype('double')[:,:,np.newaxis]
        img_rgb = np.array(np.stack((x,x,x),axis=2),dtype=np.double)
    else:
        img_rgb = np.array(x,dtype=np.double)
    #resize the image to be 224x224
    image_rs = skimage.transform.resize(img_rgb,(224,224))
    
    #normalize the image per channel basd on standard deviations and means
    #declare separate channels within image
    image_rchannel = image_rs[:,:,0]
    image_gchannel = image_rs[:,:,1]
    image_bchannel = image_rs[:,:,2]
    
    image_r_normal = ((image_rchannel-mean[0])/std[0])[:,:,np.newaxis]
    image_g_normal = ((image_gchannel-mean[1])/std[1])[:,:,np.newaxis]
    image_b_normal = ((image_bchannel-mean[2])/std[2])[:,:,np.newaxis]
    
    image_normal = np.concatenate((image_r_normal,image_g_normal,image_b_normal),axis=2)

    img_torch = torch.from_numpy(image_normal)
    
    img_torch_transposed = img_torch.permute(2,0,1)
    
    img_torch_transposed = img_torch_transposed.double()

    return img_torch_transposed


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args

    # ----- TODO -----
    
    pull_image = io.imread('../data/'+str(image_path))
    pull_image = pull_image.astype('float')/255
    
    img_torch = preprocess_image(pull_image)
        
    with torch.no_grad():
        
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    
    
    return (vgg_feat_feat.numpy()[:,np.newaxis])




def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    
    output = -1*np.linalg.norm(train_features-feature.transpose(),ord=2,axis=1)
    
    return output