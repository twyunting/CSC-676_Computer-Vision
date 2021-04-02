# 4.4
# im = np.zeros((800, 800, 3))

def myImageFilter(img0, h):
  # make sure the input h is 3*3 matrix
  if np.shape(h) != np.shape(np.zeros([3, 3])):
    print("The input h should be 3*3 matrix")
    return 
  else:
    # handle the boundaries 
    img0 = cv2.copyMakeBorder(imgtest, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = [255,255,255])
    # X_derivative of Gaussian
    Gx = np.array(h)/8 # dY/dx 
    Edged_imgGx = cv2.filter2D(img0, -1, Gx) 
    
    # Y_Derivative of Gaussian
    Gy = np.array(h)/8 # dY/dy
    Edged_imgGy = cv2.filter2D(img0, -1, Gy)
    
    # magnitude
    #x = ndimage.convolve(im, Gx)
    x = ndimage.convolve(img0, Gx)
    # Perform y convolution
    y = ndimage.convolve(img0, Gy)
    sobel = np.hypot(x, y)
    sobel= np.asarray(sobel, dtype=np.float64)
    plt.imshow(sobel, cmap='gray')
    return sobel
# output
imgtest = cv2.imread("unnormalized_image.jpg", 0)

# myImageFilter(imgtest,[[1, 2, 6] ,[2 ,0 ,0]])
myImageFilter(imgtest, [[0, 0, 3] ,[2 ,0 ,0], [0, 0, 0]])
# print(np.shape(img))