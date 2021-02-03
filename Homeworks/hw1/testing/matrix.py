# adding the packages
"""
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2d
import scipy.sparse as sps
from PIL import Image 

# Package for fast equation solving
from sys import platform
import sparseqr
"""

# World parameters
alpha = 35*math.pi/180;

img = cv2.imread('img2.png')
print(type(img))
img = img[:, :, ::-1].astype(np.float32)

nrows, ncols, colors = img.shape
ground = (np.min(img, axis=2) > 110).astype(np.float32)
foreground = (ground == 0).astype(np.float32)

m = np.mean(img, 2)
kern = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
dmdx = conv2d(m, kern, 'same')
dmdy = conv2d(m, kern.transpose(), 'same')

mag = np.sqrt(dmdx**2 + dmdy**2)
mag[0, :] = 0
mag[-1, :] = 0
mag[:, 0] = 0
mag[:, -1] = 0

theta = np.arctan2(dmdx, dmdy)
edges = mag >= 30
edges = edges * foreground

## Occlusion and contact edges
pi = math.pi
vertical_edges = edges*((theta<115*pi/180)*(theta>65*pi/180)+(theta<-65*pi/180)*(theta>-115*pi/180));
horizontal_edges = edges * (1-vertical_edges) 

kern = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
horizontal_ground_to_foreground_edges = (conv2d(ground, kern, 'same'))>0;
horizontal_foreground_to_ground_edges = (conv2d(foreground, kern, 'same'))>0;
vertical_ground_to_foreground_edges = vertical_edges*np.abs(conv2d(ground, kern.transpose(), 'same'))>0


occlusion_edges = edges*(vertical_ground_to_foreground_edges + horizontal_ground_to_foreground_edges)
contact_edges   = horizontal_edges*(horizontal_foreground_to_ground_edges);


E = np.concatenate([vertical_edges[:,:,None], 
                    horizontal_edges[:,:,None], 
                    np.zeros(occlusion_edges.shape)[:,:,None]], 2)


# Plot
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img.astype(np.uint8))
plt.axis('off')
plt.title('Input image')
plt.subplot(2,2,2)
plt.imshow(edges == 0, cmap='gray')
plt.axis('off')
plt.title('Edges')

# Normals
K = 3
ey, ex = np.where(edges[::K, ::K])
ex *= K
ey *= K
plt.figure()
plt.subplot(2,2,3)
plt.imshow(np.max(mag)-mag, cmap='gray')
dxe = dmdx[::K, ::K][edges[::K, ::K] > 0]
dye = dmdy[::K, ::K][edges[::K, ::K] > 0]
n = np.sqrt(dxe**2 + dye**2)
dxe = dxe/n
dye = dye/n
plt.quiver(ex, ey, dxe, -dye, color='r')
plt.axis('off')
plt.title('Normals')
plt.show()



# Edges and boundaries
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img.astype(np.uint8))
plt.axis('off')
plt.title('Input image')


plt.subplot(2,2,2)
plt.imshow(E+(edges == 0)[:, :, None])
plt.axis('off')
plt.title('Edges')


plt.subplot(2,2,3)
plt.imshow(1-(occlusion_edges>0), cmap='gray')
plt.axis('off')
plt.title('Occlusion boundaries')

plt.subplot(2,2,4)
plt.imshow(1-contact_edges, cmap='gray')
plt.axis('off')
plt.title('Contact boundaries');

# testing the correct matrix
total = []
for i in range(-2, 2):
  for j in range(-2, 2):
    for k in range(-2, 2):
      a = np.array([i, j, k])
      total.append(a)

res = []

for i in total:
  for j in total:
    for k in total:
      res.append(np.array([i, j, k]))
# print(np.shape(res[0]))

Nconstraints = nrows*ncols*20
Aij = np.zeros((3, 3, Nconstraints))
ii = np.zeros((Nconstraints, 1));
jj = np.zeros((Nconstraints, 1));
b = np.zeros((Nconstraints, 1));

V = np.zeros((nrows, ncols))
# Create linear contraints
c = 0
for i in range(1, nrows-1):
  for j in range(1, ncols-1):
    if ground[i,j]:
      # Y = 0
      Aij[:,:,c] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
      ii[c] = i
      jj[c] = j
      b[c] = 0
      V[i,j] = 0
      c += 1 # increment constraint counter
    else:
      # Check if current neirborhood touches an edge
      edgesum = np.sum(edges[i-1:i+2,j-1:j+2])
      # Check if current neirborhood touches ground pixels
      groundsum = np.sum(ground[i-1:i+2,j-1:j+2])
      # Check if current neirborhood touches vertical pixels
      verticalsum = np.sum(vertical_edges[i-1:i+2,j-1:j+2])
      # Check if current neirborhood touches horizontal pixels
      horizontalsum = np.sum(horizontal_edges[i-1:i+2,j-1:j+2])
      # Orientation of edge (average over edge pixels in current
      # neirborhood)            
      nx = np.sum(dmdx[i-1:i+2,j-1:j+2]*edges[i-1:i+2,j-1:j+2])/edgesum
      ny = np.sum(dmdy[i-1:i+2,j-1:j+2]*edges[i-1:i+2,j-1:j+2])/edgesum
      
      
      if contact_edges[i, j]:
        # dY/dy = 0
        Aij[:,:,c] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        ii[c] = i
        jj[c] = j
        b[c] = 0
        c += 1 # increment constraint counter
      if verticalsum > 0 and groundsum == 0:
        # dY/Dy = 1/cos a
        Aij[:,:,c] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8;
        ii[c] = i
        jj[c] = j
        b[c] = 1/np.cos(alpha)
        c += 1 # increment constraint counter
      if horizontalsum > 0 and groundsum == 0 and verticalsum == 0: #(x,y belongs to horizontal edge)
        # dY/dt = 0
        Aij[:,:,c] = a
        # Fill out the kernel (need to revise it! 3 by 3 matrix)
        ii[c] = i
        jj[c] = j
        b[c] = 0
        c += 1 # increment constraint counter
      if groundsum == 0:
        # laplacian = 0
        # 0.1 is a weight to reduce the strength of this constraint
        Aij[:,:,c] = 0.1*np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]);
        ii[c] = i
        jj[c] = j
        b[c] = 0
        c += 1 # increment constraint counter
        
        Aij[:,:,c] = 0.1*np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]); # question 4
        ii[c] = i; 
        jj[c] = j;
        b[c] = 0;
        c = c+1; # increment constraint counter

        Aij[:,:,c] = 0.1*np.array([[0, -1, 1], [0, 1, -1], [0, 0, 0]]);
        ii[c] = i; 
        jj[c] = j;
        b[c] = 0;
        c = c+1; # increment constraint counter

def sparseMatrix(i, j, Aij, imsize):
    """ Build a sparse matrix containing 2D linear neighborhood operators
    Input:
        Aij = [ni, nj, nc] nc: number of neighborhoods with contraints
        i: row index
        j: column index
        imsize: [nrows ncols]
    Returns:
        A: a sparse matrix. Each row contains one 2D linear operator
    """
    ni, nj, nc = Aij.shape
    nij = ni*nj
    
    a = np.zeros((nc*nij))
    m = np.zeros((nc*nij))
    n = np.zeros((nc*nij))
    grid_range = np.arange(-(ni-1)/2, 1+(ni-1)/2)
    jj, ii = np.meshgrid(grid_range, grid_range)
    ii = ii.reshape(-1,order='F')
    jj = jj.reshape(-1,order='F')
    
    
    k = 0
    for c in range(nc):
        # Get matrix index
        x = (i[c]+ii) + (j[c]+jj)*nrows
        a[k:k+nij] = Aij[:,:,c].reshape(-1,order='F')
        m[k:k+nij] = c
        n[k:k+nij] = x
        
        k += nij
    
    m = m.astype(np.int32)
    n = n.astype(np.int32)
    A = sps.csr_matrix((a, (m,  n)))
    
    return A

ii = ii[:c]
jj = jj[:c]
Aij = Aij[:,:,:c]
b = b[:c]
A = sparseMatrix(ii, jj, Aij, nrows)

Y = sparseqr.solve( A, b , tolerance=0)

Y = np.reshape(Y, [nrows, ncols], order='F') # Transfrom vector into image

# Recover 3D world coordinates
x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
x = x.astype(np.float32)
y = y.astype(np.float32)
x -= nrows/2
y -= ncols/2

# Final coordinates
X = x
Z = Y*np.cos(alpha)/np.sin(alpha) - y/np.sin(alpha)
Y = -Y
Y = np.maximum(Y, 0);


E = occlusion_edges.astype(np.float32);
E[E > 0] = np.nan;
Z = Z+E; #  remove occluded edges

plt.figure()
plt.subplot(2,2,1)
plt.imshow(img[1:-1, 1:-1].astype(np.uint8))
plt.axis('off')
plt.title('Edges')

plt.subplot(2,2,2)
plt.imshow(Z[1:-1, 1:-1], cmap='gray')
plt.axis('off')
plt.title('Z')


plt.subplot(2,2,3)
plt.imshow(Y[1:-1, 1:-1], cmap='gray')
plt.axis('off')
plt.title('Y')

plt.subplot(2,2,4)
plt.imshow(X[1:-1, 1:-1], cmap='gray')
plt.axis('off')
plt.title('X')




