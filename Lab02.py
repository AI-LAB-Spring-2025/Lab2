import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import urllib.request

from sklearn.datasets import load_iris
import numpy as np


def Q1():

    group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
    group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    
    ax1.boxplot(group_A, vert=True)
    ax1.set_title('Group A')
    ax1.set_ylabel('Group A Values')

    ax2.boxplot(group_B, vert=True)
    ax2.set_title('Group B')
    ax2.set_ylabel('Group B Values')   
    plt.show()


color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'purple'}
filename = "genome.txt"
def read_genome_sequence(filename):
    with open(filename, "r") as file:
        genome_sequence = file.read().strip()
    return genome_sequence

def plot_helix(filename):

    genome_sequence = read_genome_sequence(filename)

    genome_list = list(genome_sequence)
    genome_length = len(genome_list)

    t = np.linspace(0, 4 * np.pi, genome_length)  
    x = np.cos(t)
    y = np.sin(t)
    z = np.linspace(0, 5, genome_length) 

    colors = [color_map[n] for n in genome_list if n in color_map]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=colors, s=20) 

    ax.set_title("Genome Sequence")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    plt.show()





def Q3():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/2560px-Cute_dog.jpg"
    urllib.request.urlretrieve(image_url, "sample_image.jpg")

    img_array = plt.imread("sample_image.jpg")


    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    plt.suptitle("Image Transformation Demo", fontsize=14)

    axs[0,0].imshow(img_array)
    axs[0,0].set_title("Original Image")

    # Rotated Image 90° 
    rotated_img = np.rot90(img_array)
    axs[0,1].imshow(rotated_img)
    axs[0,1].set_title("Rotated 90°")

    # Flipped Image
    flipped_img = np.fliplr(img_array)
    axs[1,0].imshow(flipped_img)
    axs[1,0].set_title("Horizontally Flipped")

    # Grayscale Conversion
    gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
    axs[1,1].imshow(gray_img, cmap='gray')
    axs[1,1].set_title("Grayscale Conversion")


    plt.tight_layout()
    plt.show()


#if __name__ == "__main__":
    #Q1()
    
    #plot_helix(filename)
    #Q3()







iris = load_iris()


X = np.array(iris.data) 
Y = np.array(iris.target)

mean_values = np.mean(X, axis=0)
median_values = np.median(X, axis=0)
std_values = np.std(X, axis=0)

min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

sepal_data = X[:, :2]  

print("Mean values for each feature:", mean_values)
print("Median values for each feature:", median_values)
print("Standard deviation for each feature:", std_values)
print("Minimum values for each feature:", min_values)
print("Maximum values for each feature:", max_values)


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', marker='o')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Species')
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(X[:, 2], X[:, 3], marker='o', linestyle='-', color='green')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

