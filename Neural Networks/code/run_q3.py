import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 10
learning_rate = 0.01
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################

#initalize layer1 (i.e. hidden layer)
#in_size = 1024, out_size = 64
initialize_weights(1024,hidden_size,params,'layer1')

#initialize output (i.e. output layer)
initialize_weights(hidden_size,36,params,'output')

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    img_w = params['Wlayer1'].reshape((32,32,hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.

    plt.show()

#double check sizes
print('Shape of Wlayer1 is: ',params['Wlayer1'].shape)
print('Shape of blayer1 is: ',params['blayer1'].shape)
print('Shape of Woutput is: ',params['Woutput'].shape)
print('Shape of boutput is: ',params['boutput'].shape)

acc_iter = []
loss_iter = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################

        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)

        # implement first backward
        yb_idx = np.argmax(yb, axis=1)
        delta1_backward = probs
        delta1_backward[np.arange(probs.shape[0]), yb_idx] -= 1
        delta2_backward = backwards(delta1_backward, params, 'output', linear_deriv)

        # implement second backwards param
        backwards(delta2_backward, params, 'layer1', sigmoid_deriv)

        # update gradient steps
        params['Wlayer1'] = params['Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['blayer1'] = params['blayer1'] - learning_rate * params['grad_blayer1']
        params['Woutput'] = params['Woutput'] - learning_rate * params['grad_Woutput']
        params['boutput'] = params['boutput'] - learning_rate * params['grad_boutput']

        # increment totalloss
        total_loss += loss
        total_acc += acc


    total_acc = total_acc / batch_num
    acc_iter = np.append(acc_iter,total_acc)
    loss_iter = np.append(loss_iter,total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

plt.plot(np.arange(max_iters)+1,acc_iter)
plt.title('Running Accuracy vs Epoch [%]')
plt.grid(True)
plt.show()

plt.plot(np.arange(max_iters)+1,loss_iter)
plt.title('Running Loss vs Epoch')
plt.grid(True)
plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
##########################
h1_valid = forward(valid_x,params,'layer1')
probs_valid = forward(h1_valid, params, 'output', softmax)
loss, acc = compute_loss_and_acc(valid_y, probs_valid)

valid_acc = acc


print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    img_w = params['Wlayer1'].reshape((32,32,hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.

    plt.show()

# Q3.1.4

fig = plt.figure(1, (6., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

indices = params['cache_output'][2].argmax(axis=0)
images = valid_x[indices]
images = images.reshape(36, 32, 32)

vis = np.zeros((36, 1024))
inps = np.eye(36)
for i,inp in enumerate(inps):
    vis[i] = inp @ params['Woutput'].T @ params['Wlayer1'].T 
vis = vis.reshape(36, 32, 32)

displayed = np.zeros((72, 32, 32))
displayed[::2] = images
displayed[1::2] = vis
for ax, im in zip(grid, displayed):
    ax.imshow(im.T)
plt.savefig("out.jpg")
plt.show()

# Q3.1.5
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

#run test set to calculate confusion matrix
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']

#run network through test set
h1_test = forward(test_x, params, 'layer1')
probs_test = forward(h1_test, params, 'output', softmax)
loss_test, acc_test = compute_loss_and_acc(test_y, probs_test)

len_test = len(test_x)

#create for loop to compare probs and y to create confusion matrix
for i in range(len_test):
    #compute max index of probs at increment
    probs_test_max_index = np.int(np.argmax(probs_test[i,:])) #between 0 and 35

    #compute max index of y at increment
    y_test_max_index = np.int(np.argmax(test_y[i,:])) #between 0 and 35

    confusion_matrix[probs_test_max_index,y_test_max_index] += 1





import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()