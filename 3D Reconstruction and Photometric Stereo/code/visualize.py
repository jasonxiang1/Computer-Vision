# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import matplotlib.pyplot as plt
import numpy as np
import submission as sub
import helper
import findM2
import submission
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplt


'''
Q3.7:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

#Goal: to output the 3d points P and plot them on a scatterplot

'''
Algorithm:
    1. Load all necessary data files into program
    2. With the previous pts1 and pts2 variables, compute the F matrix
    3. Compute the necessary M1 and M2 matrices
    4. Use the submission.triangulate function to output the 3d triangulated points
'''

#load image and point correspondence data
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

#load camera intrinsics data
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

#load pts1 and pts2 data
pts1 = data['pts1']
pts2 = data['pts2']

#determine max value M
combined_var = np.append(pts1,pts2,axis=1)
M = np.amax(combined_var)

#compute F matrix from pts1 and pts2
F = submission.eightpoint(pts1,pts2,M)



# #use the findM2 function to compute M2 and C2 matrix
# M2, C2, P_old =findM2.test_M2_solution(pts1,pts2,intrinsics)


#load templecoords.npz data
temp_coord = np.load('../data/templeCoords.npz')
#templecoords files inclues x1 and y1 points
#288 points selected
x1 = temp_coord['x1']
y1 = temp_coord['y1']

#compute length of x1 and y1
len_x1 = len(x1)

#declare blank x2 and y2 arrays
x2 = []
y2 = []

#build for loop
for i in range(len_x1):
    #select x1_i and y1_i values
    x1_i = x1[i,0]
    y1_i = y1[i,0]
    
    #compute corresponding x2_i and y2_i
    x2_i, y2_i = submission.epipolarCorrespondence(im1,im2,F,x1_i,y1_i)
    
    #append x2_i and y2_i to their respective arrays
    x2 = np.append(x2,x2_i)
    y2 = np.append(y2,y2_i)


x2 = x2[:,np.newaxis]
y2 = y2[:,np.newaxis]

#create new pts1 and pts2 vectors
pts1_new = np.append(x1,y1,axis=1)
pts2_new = np.append(x2,y2,axis=1)

# #compute M1 and C1 matrices
# #assume M1 to be equal to [I|0]
# #assume C1 = K1*M1
M1_R = np.identity(3)
M1_t = np.zeros((3,1))
M1 = np.append(M1_R,M1_t,axis=1)
C1 = np.matmul(K1,M1)

#compute M2 and C2 and triangulated 3d points
M2, C2, P =  findM2.test_M2_solution(pts1_new,pts2_new,intrinsics)

np.savez("q3_4_2.npz",F=F,M1=M1,M2=M2,C1=C1,C2=C2)

#output 3d points on scatterplot
fig=plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(P[:,0],P[:,1],P[:,2],color="green")
plt.show()

# fig = plt.figure()
# ax = mplt.Axes3D(fig)
# ax.scatter(P[:,0],P[:,1],P[:,2])
# # #View #1
# # ax.view_init(-50)
# # # View #2
# # ax.view_init(40,0)
# #View #3
# ax.view_init(-60,-130)
# plt.show()
