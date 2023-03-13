import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv2.imread('lena2.jpg', 0)

# define filter parameters
ksize = 31 # filter kernel size
sigma = 5 # standard deviation of Gaussian kernel

# calculate Gaussian filter kernel
kernel = cv2.getGaussianKernel(ksize, sigma)
kernel = np.outer(kernel, kernel.transpose())

# apply filter to image
img_filtered = cv2.filter2D(img, -1, kernel)

# plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original image')
ax[1].imshow(img_filtered, cmap='gray')
ax[1].set_title('filtered image')

plt.show()
