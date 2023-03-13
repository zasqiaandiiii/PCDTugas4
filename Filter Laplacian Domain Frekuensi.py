import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv2.imread('lena2.jpg', 0)

# perform 2D Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# define Laplacian filter
H = np.zeros_like(img)
H[0, 1] = 1
H[1, 0] = 1
H[1, 2] = 1
H[2, 1] = 1
H[1, 1] = -4

# apply Laplacian filter to Fourier transform
fshift_filtered = fshift * H

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
