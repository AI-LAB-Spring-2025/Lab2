# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Z5KgSwnGYbWH85VlQt_2DRKvO3P52jlH
"""

!pip install matplotlib
!pip install numpy

import matplotlib.pyplot as plt

#Question 1
group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15,40,45,50,62]
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

data = [group_A,group_B]
plt.boxplot(data)
plt.title("Box PLot")
plt.show

#Question 2


import numpy as np
import matplotlib.pyplot as plt

file = open("genome.txt", "r")
genome_sequence = file.read()
genome = list(genome_sequence)
genome_length = len(genome)
genome.pop()
# We'll use the parametric equations for a helix:
# x = cos(t), y = sin(t), z = t (or a scaled version of t)
# We want to span a range so that the helix makes a few turns.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

t = np.linspace(0, 4 * np.pi, genome_length) # 4*pi gives about
#2 turns
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length) # z increases linearly to
#spread out the helix vertically
# Combine the coordinates into a (genome_length x 3) array
coordinates = np.column_stack((x, y, z))
colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}
# Create a scatter plot
for i, nucleotide in enumerate(genome):
 ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
 color=colors[nucleotide], marker='o')

# Adding labels and title
ax.set_xlabel('X') # Label for the x-axis
ax.set_ylabel('Y') # Label for the y-axis
ax.set_zlabel("Z")
ax.set_title('3D genome visualization on a helix') # Title of the plot

# Display the plot
plt.show()

#Question 3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

image = Image.open("image.jpeg")
numpydata = np.array(image)
plt.imshow(numpydata, interpolation='nearest')
plt.show()
numpydata = np.rot90(numpydata)
plt.imshow(numpydata, interpolation='nearest')
plt.show()
numpydata = np.fliplr(numpydata)
plt.imshow(numpydata, interpolation='nearest')
plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('image.jpeg')
gray = rgb2gray(img)
plt.imshow(gray, cmap = 'gray')
plt.show()

#Question 4

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()
# Accessing the features (data) using NumPy array
X = np.array(iris.data) # (Features (sepal length, sepal width, petal length, petal width) #Accessing the target labels (species)
Y = np.array(iris.target) # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)
mean_data = []
median_data = []
std_data = []
for i in range(4):
    print(f"Mean: {np.mean(X[:,i])}")
    mean_data.append(np.mean(X[:,i]))
    print(f"Median: {np.median(X[:,i])}")
    median_data.append(np.median(X[:,i]))
    print(f"Standard Deviation: {np.std(X[:,i])}")
    std_data.append(np.std(X[:,i]))
print(f"min mean: {min(mean_data)}")
print(f"min median: {min(median_data)}")
print(f"min standard deviation: {min(std_data)}")

#extracting sepal length
sepal_length = X[:,0]

#extracting sepal width
sepal_width = X[:,1]

#plotting
plt.figure(figsize=(8, 6))
plt.scatter(sepal_length, sepal_width, color='blue',  edgecolors='k', label="Sepal Width")
plt.scatter(sepal_length, sepal_length, color='red', edgecolors='k', label="Sepal Length")
plt.show()

#histogram
plt.figure(figsize=(8, 6))
plt.hist(sepal_length, color='blue',edgecolor='black')

plt.show()

#line plot
petal_length = X[:,2]
petal_width = X[:,3]

plt.plot(petal_length, linestyle='-', marker='o', color='green')
plt.plot(petal_width, linestyle='-', marker='o', color='red')
plt.show()

#end



