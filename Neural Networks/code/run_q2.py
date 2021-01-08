import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *


# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}

# Q 2.1
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))

#expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid
test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())
# implement forward
h1 = forward(x,params,'layer1')
print(h1.shape)


# Q 2.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# positive, ~1, ~1, (40,4)
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 2.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1

# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# # Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 2.4
batches = get_random_batches(x,y,5)
#print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# WRITE A TRAINING LOOP HERE
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        pass
        # forward

        # loss
        # be sure to add loss and accuracy to epoch totals

        # backward

        # apply gradient

        ##########################
        ##### your code here #####
        ##########################

        #implement forward for layer1
        h1_forward = forward(xb,params,'layer1')

        #implement forward for output layer
        probs_forward = forward(h1_forward,params,'output',softmax)

        #compute loss and accuracy of system
        loss_temp, acc_temp = compute_loss_and_acc(yb, probs_forward)

        #implement first backward
        yb_idx = np.argmax(yb,axis=1)
        delta1_backward = probs_forward
        delta1_backward[np.arange(probs_forward.shape[0]), yb_idx] -= 1
        delta2_backward = backwards(delta1_backward, params, 'output', linear_deriv)

        #implement second backwards param
        backwards(delta2_backward, params, 'layer1', sigmoid_deriv)

        #update gradient steps
        params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
        params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
        params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']

        #increment totalloss›
        total_loss +=loss_temp
        avg_acc += acc_temp

    avg_acc = avg_acc/batch_num

    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# Q 2.5 should be implemented in this file
# you can do this before or after training the network.

##########################
##### your code here #####
##########################

# save the old params
import copy
params_orig = copy.deepcopy(params)

eps = 1e-6
for k,v in params.items():
    if '_' in k:
        continue

    # we have a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   compute derivative with central diffs

    ##########################
    ##### your code here #####
    ##########################


    #compute rows and columns of k
    len_rows = len(v)
    if v.ndim==2:
        len_cols = len(v[0])

    #create zero gray array to fill with central differences
    cent_diff = np.zeros_like(v)

    if v.ndim==2:
        #for loop to run forward and loss for each element in k
        for i in range(len_rows):
            for j in range(len_cols):
                #add eps to the i,j element of the array
                params[k][i,j] += eps

                #run the network and find the loss
                h1_forwardcheck = forward(xb, params, 'layer1')
                probs_forwardcheck = forward(h1_forwardcheck, params, 'output', softmax)
                probs_after = probs_forwardcheck
                loss_temp, acc_temp = compute_loss_and_acc(yb, probs_forwardcheck)
                loss_temp_forward = loss_temp

                #subtract eps to the i,u element of the array
                params[k][i,j] -= 2*eps

                #run the network and find the loss
                h1_backwardcheck = forward(xb, params, 'layer1')
                probs_backwardcheck = forward(h1_backwardcheck, params, 'output', softmax)
                probs_before = probs_backwardcheck
                loss_temp, acc_temp = compute_loss_and_acc(yb, probs_backwardcheck)
                loss_temp_backward = loss_temp
                ans = probs_after-probs_before

                #compute the central difference
                cent_diff[i,j] = (loss_temp_forward-loss_temp_backward)/(2*eps)

                #set params back to the original
                params[k][i,j] += eps
    else:
        # for loop to run forward and loss for each element in k
        for i in range(len_rows):
            # add eps to the i,j element of the array
            params[k][i] += eps

            # run the network and find the loss
            h1_forwardcheck = forward(xb, params, 'layer1')
            probs_forwardcheck = forward(h1_forwardcheck, params, 'output', softmax)
            probs_after = probs_forwardcheck
            loss_temp, acc_temp = compute_loss_and_acc(yb, probs_forwardcheck)
            loss_temp_forward = loss_temp

            # subtract eps to the i,u element of the array
            params[k][i] -= 2*eps

            # run the network and find the loss
            h1_backwardcheck = forward(xb, params, 'layer1')
            probs_backwardcheck = forward(h1_backwardcheck, params, 'output', softmax)
            probs_before = probs_backwardcheck
            loss_temp, acc_temp = compute_loss_and_acc(yb, probs_backwardcheck)
            loss_temp_backward = loss_temp
            ans = probs_after-probs_before

            # compute the central difference
            cent_diff[i] = (loss_temp_forward - loss_temp_backward) / (2 * eps)

            # set params back to the original
            params[k][i] += eps

    #save the filled cent_diff in params
    params['grad_'+k] = cent_diff

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
