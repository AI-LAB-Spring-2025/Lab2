from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Accessing the features (data) using NumPy array
X = np.array(iris.data) 
Y = np.array(iris.target) 

mean_values = np.mean(X, axis=0)
median_values = np.median(X, axis=0)
std_dev_values = np.std(X, axis=0)
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

# Extract Sepal Length and Sepal Width
sepal_data = X[:, :2]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolor='k')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")

plt.subplot(1, 3, 2)
plt.hist(X[:, 0], bins=15, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sepal Length")

plt.subplot(1, 3, 3)
plt.plot(X[:, 2], X[:, 3], 'r-o')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width")

plt.tight_layout()
plt.show()

print("Mean Values:", mean_values)
print("Median Values:", median_values)
print("Standard Deviation Values:", std_dev_values)
print("Minimum Values:", min_values)
print("Maximum Values:", max_values)

