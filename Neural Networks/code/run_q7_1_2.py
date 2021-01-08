import numpy as np
import torch
import torchvision
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

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

#load MNIST
train = torchvision.datasets.MNIST("",train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#print out the network
net = Net()

optimizer = optim.SGD(net.parameters(),lr=0.3)

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

batch_size = 100

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True)

epoch_range = 4

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
    running_acc = (running_acc/60000)*100
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

#
# #declare variables
# N, D_in, H, D_out = 64, 1024, 64, 36
#
# #create random tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in,H),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(H, D_out),
#     torch.nn.Softmax()
# )
#
# loss_fn = torch.nn.MSELoss(reduction='sum')
#
# learning_rate = 1e-4
#
# epochs = 1000
#
# for t in range(epochs):
#     y_pred = model(train_x)
#
#     loss = loss_fn(y_pred,train_y)
#
#     if t%100==99:
#         print(t,loss.item())
#
#     model.zero_grad()
#
#     loss.backward()
#
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad

print('done')