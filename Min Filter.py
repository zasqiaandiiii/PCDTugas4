import cv2
import numpy as np

# load image
img = cv2.imread('lena2.jpg', 0)

# define kernel size
kernel_size = 3

# apply min filter
img_filtered = cv2.erode(img, np.ones((kernel_size, kernel_size), np.uint8))

# display images
cv2.imshow('original image', img)
cv2.imshow('filtered image', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
