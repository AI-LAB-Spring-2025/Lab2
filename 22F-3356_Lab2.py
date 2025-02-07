



#question 1
import numpy as np
import matplotlib.pyplot as plot
a = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
b = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

figure, axis = plot.subplots(1,2,figsize = (10,7))
axis[0].boxplot(a)
axis[0].set_title("Group A dataset")
axis[0].set_ylabel("Group A")

axis[1].boxplot(b)
axis[1].set_title("Group B dataset")
axis[1].set_ylabel("Group B")

plot.show()

#question 2
import numpy as np
import matplotlib.pyplot as plot
genom = ""
with open('genome.txt','r') as file:
  genom = file.read().strip()

glen = len(genom)
t = np.linspace(0, 4*np.pi, glen)
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, glen)

coord = np.column_stack((x, y, z))
fig = plot.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")
colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}

for i, g in enumerate(genom):
    ax.scatter(x[i], y[i], z[i], color=colors[g])

plot.title("Genome plot")
plot.show()

#question 3

from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('pic.png')
img_arr = asarray(img)
print (img_arr)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img_arr)
rot_img = np.rot90(img_arr)
plt.subplot(2, 2, 2)
plt.title("Rotated Image")
plt.imshow(rot_img)
flipped = np.fliplr(img_arr)
plt.subplot(2, 2, 3)
plt.title("Flipped Image")
plt.imshow(flipped)
gray = np.dot(img_arr[..., :3], [0.299, 0.587, 0.114])
plt.subplot(2, 2, 4)
plt.title("Gray Image")
plt.imshow(gray, cmap='gray')
plt.show()

#question 4
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

means = np.mean(X, axis=0)
medians = np.median(X, axis=0)
stdv = np.std(X, axis=0)
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)

sepal_data = X[:, :2]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Width")

plt.subplot(1, 3, 2)
plt.hist(X[:, 0], bins=20, color='green', edgecolor='black')
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length")

plt.subplot(1, 3, 3)
plt.plot(X[:, 2], X[:, 3], 'g-o')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Width")

plt.tight_layout()
plt.show()