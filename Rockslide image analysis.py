import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('your_image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Segment the image into rocks and soil portions based on suitable thresholds
rock_threshold = 120
soil_threshold = 60

rock_mask = (gray_image >= rock_threshold).astype(np.uint8) * 255
soil_mask = (gray_image < soil_threshold).astype(np.uint8) * 255

# Add noise to the subset rockslide image
rockslide_noise = np.random.normal(loc=0, scale=10, size=rock_mask.shape).astype(np.uint8)
rockslide_noisy_image = cv2.add(rock_mask, rockslide_noise)

# Apply a suitable filter for periodic noise removal (e.g., median filter)
filtered_image = cv2.medianBlur(rockslide_noisy_image, 5)

# Display the original, noisy, and filtered images
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(rockslide_noisy_image, cmap='gray')
plt.title('Noisy Rockslide Image')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.show()
