from sklearn.datasets import load_iris 
import numpy as np
import matplotlib.pyplot as plt


# Load the Iris dataset 
iris = load_iris() 

# Accessing the features (data) using NumPy array 
X = np.array(iris.data) # (Features (sepal length, sepal width, petal length, petal width) 

#Accessing the target labels (species) 
Y = np.array(iris.target) # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

mean_marks = np.mean(X)
print("Mean:",mean_marks)

median_marks = np.median(X)
print("Median:",median_marks)

min_marks = np.min(X)
print("Minimum:", min_marks)

max_marks = np.max(X)
print("Maximum:", max_marks)

sepal_length = X[:, 0]

sepal_width = X[:, 1]

plt.figure(figsize=(10, 15))

plt.subplot(3, 1, 1)
plt.scatter(sepal_length, sepal_width)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Scatter Plot: Sepal Length vs Sepal Width")

plt.subplot(3, 1, 2)
plt.hist(sepal_length, bins=15, color='green', edgecolor='black', alpha=0.7)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram: Sepal Length Distribution")

plt.subplot(3, 1, 3)
plt.plot(X[:, 2], X[:,3], 'r-o', alpha=0.7)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Line Plot: Petal Length vs Petal Width")

plt.tight_layout()
plt.show()