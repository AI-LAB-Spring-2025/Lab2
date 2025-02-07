from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()

X = np.array(iris.data)

Y = np.array(iris.target)

mean = np.mean(X, axis=0)
median = np.median(X, axis=0)
std_dev = np.std(X, axis=0)


min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)


sepal_length_width = X[:, :2]


print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Minimum Values:", min_values)
print("Maximum Values:", max_values)


plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.scatter(sepal_length_width[:, 0],
            sepal_length_width[:, 1], c=Y, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')


plt.subplot(1, 3, 2)
plt.hist(X[:, 0], bins=20, color='blue', edgecolor='black')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length')


plt.subplot(1, 3, 3)
plt.plot(X[:, 2], X[:, 3], 'r-')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')

plt.tight_layout()
plt.show()
