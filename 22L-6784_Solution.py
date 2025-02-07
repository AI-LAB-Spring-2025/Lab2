import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image



groupA = np.array([12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62])
groupB = np.array([12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15])

# print(f"Group A = {groupA}")
# print(f"Group B = {groupB}")

# # Box plot of group A 
# plt.boxplot(groupA)
# plt.title("Box plot of Group A")
# plt.ylabel("Data Values")
# plt.show()

# # Box plot of group B
# plt.boxplot(groupB)
# plt.title("Box plot of Group B")
# plt.ylabel("Data Values")
# plt.show()


# # Box Plot of Group A and B together
# plt.boxplot([groupA, groupB], labels=["Group A", "Group B"])
# plt.title("Box Plots of Group A and Group B")
# plt.ylabel("Values")
# plt.show()


# # Sub Plots of Group A
# # I am making a boxplot and a histogram of group A's data.

# # Boxplot of Group A
# plt.subplot(1, 2, 1) # rows, coloumns, index
# plt.boxplot(groupA)
# plt.title("Boxplot of group A")

# # Histogram of group A

# # EQUI-WIDTH BINNING : I recently learned this in my datamining course so I wanted to apply it!
# max_value = groupA.max()
# min_value = groupA.min()
# total_bins = 5
# num_bins = int((max_value - min_value)/total_bins)

# print(f"Max Value = {max_value}")
# print(f"Min value = {min_value}")
# print(f"Total bins I'm making = {total_bins}")
# print(f"Number of bins needed = ({max_value} - {min_value})/{total_bins} = {num_bins}")


# plt.subplot(1, 2, 2)
# plt.hist(groupA, bins=num_bins, color='skyblue')
# plt.title("Histogram")
# plt.tight_layout()
# plt.show()
#------------------------------------------------






# # QUESTION 2: 
# file = "genome.txt"

# with open(file, "r") as f:
#     genome = f.read() 

# genome = list(genome)
# genome_len = len(genome)

# # We'll use the parametric equations for a helix: 
# #   x = cos(t), y = sin(t), z = t (or a scaled version of t) 
# # We want to span a range so that the helix makes a few turns. 
# t = np.linspace(0, 4 * np.pi, genome_len)  # 4*pi gives about 2 turns 
# x = np.cos(t) 
# y = np.sin(t) 
# z = np.linspace(0, 5, genome_len)  # z increases linearly to spread out the helix vertically 
# # Combine the coordinates into a (genome_length x 3) array 
# coordinates = np.column_stack((x, y, z)) 

# color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}
# colors = [color_map.get(base, 'black') for base in genome]  # Default to black if unknown base



# fig = plt.figure(figsize=(8,8)) 
# ax = fig.add_subplot(projection='3d')


# # Scatter plot with assigned colors
# ax.scatter(x, y, z, c=colors, s=50) # 50 is the size of points in the figure
# # Labels and title
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Genome Visualization")

# plt.show()

#---------------------------------------------


# # Question 3: 
# image = Image.open("doggies.jpg") 
# image_array = np.array(image)   # converting the image into an np array

# # Print shape of the array
# print(image_array.shape)  # (height, width, channels) for RGB images

 
# plt.figure(figsize=(8,8))
# plt.subplot(2, 2, 1)
# plt.imshow(image_array)
# plt.title("Original Image")
# plt.axis("off")

# # Rotated Image
# plt.subplot(2, 2, 2)
# rotated_image = np.rot90(image_array)
# plt.imshow(rotated_image)
# plt.title("Rotated Image")
# plt.axis("off")

# # Flipped image
# plt.subplot(2, 2, 3)
# flipped_image = np.fliplr(image_array)
# plt.imshow(flipped_image)
# plt.title("Flipped Image")
# plt.axis("off")

# # Gray Scale Image
# plt.subplot(2, 2, 4)
# grayscale_image = np.dot (image_array[..., :3], [0.299, 0.587, 0.114])
# plt.imshow(grayscale_image, cmap="gray")
# plt.title("GrayScale Image")
# plt.axis("off")

# plt.show()

#--------------------------------------------------------


# Question 4: 

