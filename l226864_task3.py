#take any image and convert it into numpy array
from PIL import Image
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


img=Image.open('cloud.jpg')
np_img = np.array(img)

plt.imshow(np_img)
plt.axis('off')  # Hide axes
plt.show()

#rotating image
rotated_img=np.rot90(np_img, k = 1, axes = (0, 1))
plt.imshow(rotated_img)
plt.axis('off')  # Hide axes
plt.show()

#flipped image
flipped_img = np.fliplr(rotated_img)
plt.imshow(flipped_img)
plt.axis('off')  # Hide axes
plt.show()

gray_img = np.dot (np_img[..., :3], [0.299, 0.587, 0.114])
plt.imshow(gray_img)
plt.axis('off')  # Hide axes
plt.show()
