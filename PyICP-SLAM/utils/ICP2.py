"""
ref: https://github.com/ClayFlannigan/icp/blob/master/icp.py

try this later: https://github.com/agnivsen/icp/blob/master/basicICP.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyflann

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

# def nearest_neighbor(src, dst):
#     '''
#     Find the nearest (Euclidean) neighbor in dst for each point in src
#     Input:
#         src: Nxm array of points
#         dst: Nxm array of points
#     Output:
#         distances: Euclidean distances of the nearest neighbor
#         indices: dst indices of the nearest neighbor
#     '''

#     assert src.shape == dst.shape

#     # create FLANN object and build index
#     flann = pyflann.FLANN()
#     params = flann.build_index(dst, algorithm='kdtree', trees=4)

#     # query nearest neighbors
#     indices, distances = flann.nn_index(src, 1, checks=params['checks'])

#     return distances.ravel(), indices.ravel()


def kiss_icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001, delta=0.1):
    '''
    The KISS-ICP algorithm: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        delta: step size for calculating the Jacobian
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # calculate the weight matrix based on the distances
        # W = np.diag(1.0 / distances)
        W = np.diag(1.0 / (distances + 1e-8))
        
        # calculate the Jacobian matrix
        J = np.zeros((m * len(distances), m + 1))

        for k in range(len(distances)):
            x = src[:m, k]
            y = dst[:m, indices[k]]
            J[m*k:m*(k+1), :m] = np.eye(m) - delta * (y - x).reshape((-1, 1)) @ (y - x).reshape((1, -1))
            J[m*k:m*(k+1), m] = -(y - x)
        print("shape of jacobian", J.shape)
        print("Shape of weights", W.shape)

        # calculate the update step
        dT = np.linalg.lstsq(W @ J, W @ (dst - src))[0]

        # apply the update step
        src += (dT.reshape((-1, 1)) @ np.ones((1, src.shape[1]))).T

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i

