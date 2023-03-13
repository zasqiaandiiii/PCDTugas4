import cv2
import numpy as np

# load image
img = cv2.imread('lena2.jpg', 0)

# apply median filter
img_filtered = cv2.medianBlur(img, 5)

# display images
cv2.imshow('original image', img)
cv2.imshow('filtered image', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
