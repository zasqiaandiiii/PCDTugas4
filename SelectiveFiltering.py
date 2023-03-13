import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv2.imread('lena2.jpg', 0)

# perform 2D Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# define highpass filter mask
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply highpass filter to Fourier transform
fshift_filtered = fshift * mask

# perform inverse Fourier transform to obtain filtered image
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered).real

# plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original image')
ax[1].imshow(img_filtered, cmap='gray')
ax[1].set_title('filtered image')

plt.show()
