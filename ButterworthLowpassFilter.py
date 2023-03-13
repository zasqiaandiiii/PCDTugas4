import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# load image
img = cv2.imread('lena2.jpg', 0)

# define filter parameters
cutoff = 50 # Hz
order = 4

# calculate Butterworth filter coefficients
nyquist = 0.5*img.shape[1]
normal_cutoff = cutoff/nyquist
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# apply filter to image
img_filtered = cv2.filter2D(img, -1, b)

# plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original image')
ax[1].imshow(img_filtered, cmap='gray')
ax[1].set_title('filtered image')

plt.show()
