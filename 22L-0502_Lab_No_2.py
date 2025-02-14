import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####### =============================== Question No 1 =================
group_A = np.array([12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15,40,45,50,62])
group_B = np.array([12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15])

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.boxplot(group_A)
plt.title("Group A")
plt.ylabel("Measurement Values")

plt.subplot(1,2,2)
plt.boxplot(group_B)
plt.title("Group B")
plt.ylabel("Measurements Values")

plt.suptitle("Comparison of BoxPlot between Group A and Group B")
plt.show()

######===================================================================



###### =============================== Question No 2 =================
with open("genome.txt", "r") as fp:
    genome_Seq = fp.read().strip()

print(" genome Sequence is: ", genome_Seq)

genome_list = list(genome_Seq)
genome_len = len(genome_list)
print(" Lenght of Ganome List is: ", genome_len)

t = np.linspace(0, 4*np.pi,genome_len)

x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_len)

coordinates = np.column_stack((x, y, z))
# # print(" Coordinates are: ", coordinates)

###3 I am using Dictionary for mapping Color to a Character
color = {"A" : "red", "T" : "blue", "G" : "green", "C" : "yellow"}
# # print(" Color ", color)
color = [color[n] for n in genome_list]

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c = color, s = 50)

ax.set_title("3D Scatter Plot of Genome Sequence")
ax.set_xlabel("X-axes")
ax.set_ylabel("Y-axes")
ax.set_zlabel("Z-axes")
plt.show()

####### ===============================================================


####### =============================== Question No 3 =================

arr = np.array(cv2.imread("1.jpg",1))
print(" Shape of Image is: ", arr)

plt.imshow(arr)
plt.title(" GTR Picture")
plt.axis("off")
plt.show()

rotateimg = np.rot90(arr, 1)

flipimg = np.fliplr(arr)

fig, axis = plt.subplots(1, 3, figsize = (12,6))

axis[0].imshow(arr)
axis[0].set_title("Original Picture")
axis[0].axis("off")

axis[1].imshow(rotateimg)
axis[1].set_title("Rotated Picture")
axis[1].axis("off")

axis[2].imshow(flipimg)
axis[2].set_title("Flipped Picture")
axis[2].axis("off")

plt.show()

gray_img = np.dot (arr[..., :3], [0.299, 0.587, 0.114]) 

fig, axis = plt.subplots(1, 2, figsize = (12,6))


axis[0].imshow(arr)
axis[0].set_title("Original Picture")
axis[0].axis("off")

axis[1].imshow(gray_img)
axis[1].set_title("Gray Scaled Picture")
axis[1].axis("off")

plt.show()
####### ===============================================================


####### =============================== Question No 4 =================
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target


print("Mean ",df.mean())
print("Median ",df.std())
print("Standerd Daviation",df.median())
print("Minimumn",df.min())
print("Maximum ",df.min())



sepal_length = np.array(df['sepal length (cm)'])
sepal_width = np.array(df['sepal width (cm)'])
petal_length = np.array(df['petal length (cm)'])
petal_width = np.array(df['petal width (cm)'])


print(np.sum(sepal_length))
print(np.sum(sepal_width))

plt.figure(figsize = (15,7))

plt.subplot(1,3,1)
plt.scatter(sepal_length, sepal_width)
plt.title('Sepal Length vs Sepal Width')


plt.subplot(1,3,2)
plt.hist(sepal_length)
plt.title('Sepal Length')



plt.subplot(1,3,3)
plt.plot(petal_length)
plt.plot(petal_width)
plt.show()
