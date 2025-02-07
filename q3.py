from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import numpy as np
img = Image.open("Sample.png")
data = asarray(img)
print(data)

plt.imshow(data)
plt.axis('off')
plt.show()

rotated_data = np.rot90(data)

flipped_data = np.fliplr(data)

plt.imshow(rotated_data)
plt.axis('off')
plt.title('Rotated Image')
plt.show()


plt.imshow(flipped_data)
plt.axis('off')
plt.title('Flipped Image')
plt.show()

gray_img = np.dot(data[..., :3], [0.299, 0.587, 0.114])
plt.imshow(gray_img)
plt.axis('off')
plt.title('Gray Image')
plt.show()
