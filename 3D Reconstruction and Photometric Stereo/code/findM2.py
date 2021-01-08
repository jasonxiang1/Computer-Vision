# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import numpy as np
import submission
import helper


'''
Q3.3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_5.npz
'''


def test_M2_solution(pts1, pts2, intrinsics):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :return: M2, the extrinsics of camera 2
    		 C2, the 3x4 camera matrix
    		 P, 3D points after triangulation (Nx3)
    '''
    #pass
    '''
    Note:
        - Perform code within the API, should not bring in other functions from other files in
    Algorithm:
        1. Compute the Fundamental Matrix F from the given pts1 (see submission.py for code)
        2. Compute E matrix from the given F matrix (see submission.py for code)
        3. Decompose E into four solutions
        4. Compute a number of 3d points using triangulation
        5. Determine correct M2 matrix and output
    '''
    
    #define intrinsic matrices of camera 1 and 2
    K1 = intrinsics['K1']
    K2 = intrinsics["K2"]
    
    #determine maximum of the width and heights from pts1 and pts2
    #concatenate x-values of pts1 and pts2 together
    x_combined = np.append(pts1,pts2,axis=1)
    M = np.amax(x_combined)
    
    #compute the Fundamental Matrix F from pts1, pts2, and M
    F = submission.eightpoint(pts1,pts2,M)
    
    #compute the essential matrix from the F matrix and intrinsics
    #remember that the E matrix matches a point in the first camera image to a line in the second camera image
    E =  submission.essentialMatrix(F,K1,K2)
    
    #determine M1 to be equal to the [I|0]
    M1_R = np.identity(3)
    M1_t = np.zeros((3,1))
    M1 = np.append(M1_R,M1_t,axis=1)
    C1 = np.matmul(K1,M1)
    
    #output M2s from camera2 function in helper.py file
    M2s = helper.camera2(E)
    
    #declare vector of ones that if four long
    #will run through tests and change to zeros the options that do not hold true for the tests
    bool_vect = np.ones((4))
    
    #compute the determinant of the R matrices of each option
    #correct choice has a det(R) equal to 1
    for i in range(4):
        #compute determinant of Ri
        M2_i = M2s[:,:,i]
        R_i = M2_i[:,:3]
        t_i = M2_i[:,-1]
        det_i = np.linalg.det(R_i)
        #determinants that are equal to -1 are not the right M2
        if np.round(det_i,0) != 1.0:
            bool_vect[i] = 0
        #compute sample 3D point using triangulation
        P, err = submission.triangulate(C1,pts1,np.matmul(K2,M2_i),pts2)
        #select random P point
        rand_norm_num = np.int(np.random.rand(1)*len(pts1))
        rand_row = P[rand_norm_num,:]
        if rand_row[-1] < 0:
            bool_vect[i] = 0
        #retain 3d point vectors and errors if bool_vect is still 1
        if bool_vect[i] == 1:
            P_choose = P
            err_choose = err
            C2_choose = np.matmul(K2,M2_i)
            M2_choose = M2_i
    
    M2 = M2_choose
    C2 = C2_choose
    P = P_choose
    
    
    return M2, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')

	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
	np.savez('q3_3_3', M2=M2, C2=C2, P=P)
