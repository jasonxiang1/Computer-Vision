import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    #define variance based on input and output dimension sized
    var_w = 2/(in_size+out_size)

    #define lower and upper bounds
    low_bound = -np.sqrt(6)/np.sqrt(in_size+out_size)
    upper_bound = np.sqrt(6)/np.sqrt(in_size+out_size)

    #define upper and lower bound of system
    W = np.random.uniform(low=low_bound,high=upper_bound,size=(in_size,out_size))
    b = np.zeros(out_size)


    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    #res is of a specific size
    res = 1/(1+np.exp(-x));

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################

    #np.matmul(X,W).shape = (40,25)
    #b is a row vector that broadcasts across all rows of the first term
    pre_act = np.matmul(X,W) + b

    #implement sigmoid on the function
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    #res input is (40,4) matrix

    #acquire column of max values of res
    row_maxes = np.amax(x,axis=1)
    row_maxes = row_maxes[:,np.newaxis]

    #subtract row_maxes from res to get numerically stable matrix
    res_stable = x-row_maxes

    #calculate software max of res
    res_stable_denom = np.sum(np.exp(res_stable),axis=1)
    res_stable_denom = res_stable_denom[:,np.newaxis]
    res = np.exp(res_stable)/res_stable_denom

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    #f(x) is probs (40,4)
    #y is y (40,4)

    #compute log of f(x)
    f_log = np.log(probs)

    len_examples = len(y)
    loss_examples = np.zeros(len_examples)
    acc_examples = np.zeros(len_examples)


    for i in range(len_examples):
        loss_examples[i] = np.dot(y[i,:],np.transpose(f_log[i,:]))
        acc_examples[i] = np.argmax(probs[i,:])==np.argmax(y[i,:])


    loss = -np.sum(loss_examples)
    acc = np.sum(acc_examples)/len_examples

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################

    dcdz_weight = delta*activation_deriv(post_act)
    dcdw_weight = (np.matmul(dcdz_weight.T,X).T)

    dcdz_bias = delta*activation_deriv(post_act)
    dcdb_bias = np.matmul(dcdz_bias.T,np.ones((len(dcdz_bias))))

    dcdz_x = delta*activation_deriv(post_act)
    dcdx_x = (np.matmul(dcdz_x,W.T))

    grad_W = dcdw_weight
    grad_b = dcdb_bias
    grad_X = dcdx_x
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]

def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    #make sure arrays are aligned with each other
    #make sure sample lines up with label

    #create one large array of both x and y to conserve list
    organized_combine = np.append(x,y,axis=1)

    #shuffle arrays
    np.random.shuffle(organized_combine[:])

    #split combined array into x and y
    x_shuffled = organized_combine[:,:len(x[0])]
    y_shuffled = organized_combine[:,len(x[0]):]

    len_samples = len(x)
    num_batches = np.int(len_samples/batch_size)

    for i in range(num_batches):
        temp_batch = (x_shuffled[batch_size*i:batch_size*i+batch_size],y_shuffled[batch_size*i:batch_size*i+batch_size])
        batches.append(temp_batch)
    return batches
