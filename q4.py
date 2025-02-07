import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


iris_data = load_iris()

features = np.array(iris_data.data) 
labels = np.array(iris_data.target) 


feature_labels = iris_data.feature_names

mean_values = np.mean(features, axis=0)
median_values = np.median(features, axis=0)
std_dev_values = np.std(features, axis=0)
min_values = np.min(features, axis=0)
max_values = np.max(features, axis=0)


for i, label in enumerate(feature_labels):
    print(f"{label.title()}:")
    print(f"  Mean: {mean_values[i]:.2f}")
    print(f"  Median: {median_values[i]:.2f}")
    print(f"  Std Dev: {std_dev_values[i]:.2f}")
    print(f"  Min: {min_values[i]:.2f}")
    print(f"  Max: {max_values[i]:.2f}\n")

plt.figure(figsize=(12, 5))


plt.subplot(1, 3, 1)
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', edgecolor='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')

plt.subplot(1, 3, 2)
plt.hist(features[:, 0], bins=12, color='teal', alpha=0.75)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Sepal Length Distribution')


plt.subplot(1, 3, 3)
plt.plot(features[:, 2], features[:, 3], marker='o', linestyle='-', color='darkorange')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')

plt.tight_layout()
plt.show()