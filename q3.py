import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imgio 

image_path = "test.jpg"
image_data = imgio.imread(image_path)

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(image_data)
plt.title("Original")
plt.axis("off")


rotated_image = np.rot90(image_data)
plt.subplot(1, 4, 2)
plt.imshow(rotated_image)
plt.title("Rotated")
plt.axis("off")


mirrored_image = np.fliplr(image_data)
plt.subplot(1, 4, 3)
plt.imshow(mirrored_image)
plt.title("Flipped")
plt.axis("off")

grayscale_image = np.dot(image_data[..., :3], [0.299, 0.587, 0.114])
plt.subplot(1, 4, 4)
plt.imshow(grayscale_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")


plt.tight_layout()
plt.show()