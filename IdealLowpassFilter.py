import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('gambar1.jpg', cv2.IMREAD_GRAYSCALE)

# Generate frequency response of ideal lowpass filter
fc = 0.1 # Cutoff frequency
N = img.shape[0]
M = img.shape[1]
freqs_u = np.fft.fftfreq(N).reshape(-1, 1)
freqs_v = np.fft.fftfreq(M).reshape(1, -1)
dist = np.sqrt(freqs_u ** 2 + freqs_v ** 2)
H = np.zeros((N, M), dtype=complex)
H[dist <= fc] = 1

# Plot frequency response
fig, ax = plt.subplots()
ax.imshow(np.fft.fftshift(np.abs(H)))
ax.set_xlabel('Frequency v')
ax.set_ylabel('Frequency u')
plt.show()

# Compute frequency spectrum of image
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
magnitude_spectrum = 20 * np.log(np.abs(F_shifted))

# Apply frequency response of filter
F_filtered_shifted = H * F_shifted
F_filtered = np.fft.ifftshift(F_filtered_shifted)
img_filtered = np.real(np.fft.ifft2(F_filtered))

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original image')
axs[0, 1].imshow(magnitude_spectrum, cmap='gray')
axs[0, 1].set_title('Magnitude spectrum')
axs[1, 0].imshow(np.abs(img_filtered), cmap='gray')
axs[1, 0].set_title('Filtered image')
axs[1, 1].imshow(np.fft.fftshift(np.abs(F_filtered_shifted)), cmap='gray')
axs[1, 1].set_title('Filtered spectrum')
plt.show()
