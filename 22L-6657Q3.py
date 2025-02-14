import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('cat.png')
num = np.asarray(img)

print(num)

plt.imshow(num)
plt.axis('off')
plt.show()

rotate = np.rot90(num)
plt.imshow(rotate)
plt.axis('off')
plt.show()

flip = np.fliplr(num)
plt.imshow(flip)
plt.axis('off')
plt.show()

gray_img = np.dot(num[..., :3], [0.299, 0.587, 0.114])
plt.imshow(gray_img)
plt.axis('off')
plt.show()









