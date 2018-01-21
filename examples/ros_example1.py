import numpy as np
from math import sqrt
import hpat
import hpat.ros
from hpat import prange, stencil

@stencil
def gaussian_blur(a):
    return (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
            a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
            a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
            a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
            a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)

@hpat.jit
def read_example():
    A = hpat.ros.read_ros_images("image_test.bag")
    # crop out dashboard
    B = A[:,:-50,:,:]
    # intensity threshold
    threshold = B.mean() + .004 * B.std()
    n = B.shape[0]
    mask = np.empty(n, np.bool_)
    for i in prange(n):
        im = B[i]
        mask[i] = im.mean() > threshold
    C = B[mask]
    D = np.empty_like(C)
    for i in prange(len(C)):
        D[i,:,:,0] = gaussian_blur(C[i,:,:,0])
        D[i,:,:,1] = gaussian_blur(C[i,:,:,1])
        D[i,:,:,2] = gaussian_blur(C[i,:,:,2])
    # K-means model
    numCenter = 4
    numIter = 10
    dn, dh, dw, dc = D.shape
    centroids = np.random.randint(0, 255, (numCenter, dh, dw, dc)).astype(np.uint8)
    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((D[i]-centroids[j])**2))
                                for j in range(numCenter)] for i in range(dn)])
        labels = np.array([dist[i].argmin() for i in range(dn)])
        for i in range(numCenter):
            mask2 = (labels==i)
            num_points = np.sum(mask2)
            if num_points != 0:
                centroids[i] = np.sum(D[mask2], 0) / num_points
            else:
                centroids[i] = np.random.randint(0, 255, (dh, dw, dc)).astype(np.uint8)

    return centroids

print(read_example().sum())
#hpat.distribution_report()
