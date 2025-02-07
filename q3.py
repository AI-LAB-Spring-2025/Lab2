# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image and convert it into a numpy array
img = Image.open('sample.jpeg')
img_array = np.asarray(img)

# Print image details
print(img_array)
print("Data Type:", type(img_array))
print("Shape:", img_array.shape)

# Plot the original image
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")

# Rotate image by 90 degrees
rotated_img = np.rot90(img_array)

plt.subplot(1, 4, 2)
plt.imshow(rotated_img)
plt.title("Rotated 90°")
plt.axis("off")

# Flip image left-right
flipped_img = np.fliplr(img_array)

plt.subplot(1, 4, 3)
plt.imshow(flipped_img)
plt.title("Flipped Left-Right")
plt.axis("off")

# Convert to grayscale using the formula
gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

plt.subplot(1, 4, 4)
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# Show all plots
plt.tight_layout()
plt.show()
