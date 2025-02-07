from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('starry.jpeg')
numpydata = asarray(img)

#print(numpydata)

plt.figure(figsize=(6,6))
plt.title("original")
plt.show()

#original
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(numpydata)
plt.title("Original Image")
plt.axis("off")

#90 deg rotation
rotated_img = np.rot90(numpydata)
plt.subplot(1, 4, 2)
plt.imshow(rotated_img)
plt.title("Rotated 90°")
plt.axis("off")

#flip
flipped_img = np.fliplr(numpydata)
plt.subplot(1, 4, 3)
plt.imshow(flipped_img)
plt.title("Flipped Image")
plt.axis("off")

#grayscale 
gray_img = np.dot(numpydata[..., :3], [0.299, 0.587, 0.114])
plt.subplot(1, 4, 4)
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.show()
