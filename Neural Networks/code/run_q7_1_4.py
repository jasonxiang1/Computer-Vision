import numpy as np
import torch
import torchvision
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets

import matplotlib.pyplot as plt

import os
import numpy as np
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        #self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(400,60)
        self.fc2 = nn.Linear(60, 50)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        sigx = nn.Sigmoid()
        softx = nn.Softmax()
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = sigx(x)
        x = self.fc2(x)
        #x = softx(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
emnist = torchvision.datasets.EMNIST(root='../data', split='balanced', download=True,transform=transforms.Compose([transforms.ToTensor()]))

corr_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E','F','G','H','N','Q','R','T',
                'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
                'w','x','y','z']

print(emnist)

#youtube video was used
#https://www.youtube.com/watch?v=i2yPxY2rOzs

#reference was used
#https://github.com/pytorch/tutorials/blob/master/beginner_source/examples_nn/two_layer_net_nn.py

#reference was used
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# #load NIST36 data
# train_data = scipy.io.loadmat('../data/nist36_train.mat')
# valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
# test_data = scipy.io.loadmat('../data/nist36_test.mat')
#
# train_x_np, train_y_np = train_data['train_data'], train_data['train_labels']
# valid_x_np, valid_y_np = valid_data['valid_data'], valid_data['valid_labels']
# test_x_np, test_y_np = test_data['test_data'], test_data['test_labels']
#
# #convert data over from numpy to tensor
# #train
# train_x = torch.from_numpy(train_x_np).float()
# train_y = torch.from_numpy(train_y_np).float()
# #valid
# valid_x = torch.from_numpy(valid_x_np).float()
# valid_y = torch.from_numpy(valid_y_np).float()
# #valid
# test_x = torch.from_numpy(test_x_np).float()
# test_y = torch.from_numpy(test_y_np).float()

#print out the network
net = Net()

optimizer = optim.SGD(net.parameters(),lr=0.6)

optimizer.zero_grad()

#train the network
criterion = nn.CrossEntropyLoss()

# #get length of input data
# len_x = len(train_x[0])
# len_y = len(train_y[0])
#
# #concatenate x and y training sets
# train_cat = torch.cat((train_x,train_y),1)
# #concatenate x and y test sets
# test_cat = torch.cat((test_x,test_y),1)

batch_size = 500

trainloader = torch.utils.data.DataLoader(emnist, batch_size=batch_size,shuffle=True)

epoch_range = 15

acc_iter = []
loss_iter = []

for epoch in range(epoch_range):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trainloader,0):
        # #splice data to get input and labels
        # inputs = data[:,:len_x]
        # labels = data[:,len_x:]
        inputs, labels = data
        #input_2d = inputs.reshape(32,32)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #calculate accuracy between outputs and examples
        acc_batch = torch.argmax(outputs,1)==labels
        running_acc += torch.sum(acc_batch)
    running_acc = (running_acc/112800)*100
    print('[Epoch #: %d] loss: %.3f ; acc: %.3f'%
          (epoch+1,running_loss,running_acc))
    acc_iter = np.append(acc_iter,running_acc)
    loss_iter = np.append(loss_iter,running_loss)

plt.plot(np.arange(epoch_range)+1,acc_iter)
plt.title('Running Accuracy vs Epoch [%]')
plt.grid(True)
plt.show()

plt.plot(np.arange(epoch_range)+1,loss_iter)
plt.title('Running Loss vs Epoch')
plt.grid(True)
plt.show()


print('done with network training')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

    # plt.imshow(im1)
    # plt.show()
    bboxes, bw = findLetters(im1)

    # create dataset list that is 32x32
    desired_flat_size = np.int(28 * 28)
    flat_dataset = np.zeros((1, desired_flat_size))

    pad_size = 13

    for i in bboxes:

        # make sure to add before and after not just after
        # apply padding to i
        row_top = np.int(i[0] - pad_size)
        row_bot = np.int(i[2] + pad_size + 1)
        col_left = np.int(i[1] - pad_size)
        col_right = np.int(i[3] + pad_size + 1)

        image_cropped = bw[row_top:row_bot, col_left:col_right]
        # plt.imshow(image_cropped)

        # find image and pad to be a square crop
        len_rows = len(image_cropped)
        len_cols = len(image_cropped[0])
        val_pad = np.int(np.abs((len_rows - len_cols) / 2))
        extra_pad = 20

        if len_rows > len_cols:
            image_square = skimage.util.pad(image_cropped, ((extra_pad, extra_pad), (val_pad+extra_pad, val_pad+extra_pad)), mode='constant',
                                            constant_values=0)
        elif len_cols > len_rows:
            image_square = skimage.util.pad(image_cropped, ((val_pad+extra_pad, val_pad+extra_pad), (extra_pad, extra_pad)), mode='constant',
                                            constant_values=0)

        # invert image
        image_square = ~image_square

        # image_cropped = skimage.transform.rescale(np.float32(image_square),32/len_rows,anti_aliasing=False)
        image_cropped = skimage.transform.resize(np.float32(image_square), (28, 28), anti_aliasing=True)
        image_cropped = np.transpose(image_cropped)
        # image_cropped = np.where(image_cropped<0.95,0,1)
        image_cropped[image_cropped < 0.95] = image_cropped[image_cropped < 0.95] * 0.6

        # # only for spyder ide
        # plt.imshow(image_cropped);
        # plt.show()

        # flatten image and store in dataset
        image_flat = image_cropped.flatten()[np.newaxis, :]

        # store flatten image into dataset
        flat_dataset = np.append(flat_dataset, image_flat, axis=0)

    flat_dataset = flat_dataset[1:, :]
    #convert data to tensor
    dataset_tensor = torch.from_numpy(flat_dataset).resize(len(bboxes),28,28)
    dataset_tensor = dataset_tensor[:,None,:,:].float()
    # run through data

    outputs = net(dataset_tensor)
