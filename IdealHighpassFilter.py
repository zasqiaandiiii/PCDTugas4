import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv2.imread('lena2.jpg', 0)

# perform 2D Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# define highpass filter parameters
D0 = 50 # cutoff frequency
n = 1 # order of filter

# create highpass filter mask
M, N = img.shape
u, v = np.meshgrid(np.arange(-N/2, N/2), np.arange(-M/2, M/2))
D = np.sqrt(u**2 + v**2)
H = 1 - 1/(1 + (D0/D)**(2*n))

# apply highpass filter to Fourier transform
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
