import cv2
import numpy as np
from matplotlib import pyplot as plt

# load image in grayscale
img = cv2.imread('lena2.jpg', 0)

# calculate the 2D FFT of the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# display the original and magnitude spectrum images
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
