# import matplotlib.pyplot as plt


# group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
# group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))


# axes[0].boxplot(group_A)
# axes[0].set_title("Group A")
# axes[0].set_ylabel("Values")

# axes[1].boxplot(group_B)
# axes[1].set_title("Group B")
# axes[1].set_ylabel("Values")

# plt.show()


# import random 
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# with open("helix.txt", "r") as file:
#     genome_sequence = file.read().strip()  

# genome_list = list(genome_sequence)
# genome_length = len(genome_list)

# # Parametric equations for a helix
# t = np.linspace(0, 4 * np.pi, genome_length)  # About 2 turns
# x = np.cos(t)
# y = np.sin(t)
# z = np.linspace(0, 5, genome_length)  # Spread helix vertically
# coordinates = np.column_stack((x, y, z))


# color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'yellow'}
# colors = [color_map.get(nucleotide) for nucleotide in genome_list] #doesnt work without this liein 

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, y, z, c=colors, marker='o')

# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("genomce visualization")

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image 
from numpy import asarray

# Load an image and convert it to a numpy array
img = Image.open("123.jpg")
img_array = asarray(img)

# Plot original image
plt.figure(figsize=(6,6))
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Rotate and flip the image
rotated_img = np.rot90(img_array)
flipped_img = np.fliplr(img_array)

# Plot rotated image
plt.figure(figsize=(6,6))
plt.imshow(rotated_img)
plt.title("Rotated Image")
plt.axis("off")
plt.show()

# Plot flipped image
plt.figure(figsize=(6,6))
plt.imshow(flipped_img)
plt.title("Flipped Image")
plt.axis("off")
plt.show()

# Convert image to grayscale
gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

# Plot grayscale image
plt.figure(figsize=(6,6))
plt.imshow(gray_img, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()














######## q4 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df=pd.read_csv('Titanic-Dataset.csv')
# df.head(5)
# dtypes = df.dtypes
# print("\nData Types:\n", dtypes)

# # Check for missing values
# missing_values = df.isnull().sum()
# print("Missing Values:\n", missing_values)

# # Check for duplicate values
# duplicates = df.duplicated().sum()
# print("\nNumber of Duplicate Rows:", duplicates)

# # Get summary statistics
# summary = df.describe()
# print("\nSummary Statistics:\n", summary)

# Binning 'Age' column into categories
# bins = [18, 25, 35, 50, 100]
# labels = ['Young', 'Adult', 'Middle-Aged', 'Senior']
# df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
# print("\nBinned Age Groups:\n", df[['Age', 'Age Group']])









#from here it is histogram of age with and without binning 

# df['Age'].fillna(df['Age'].median(), inplace=True)

# # Define bins and labels
# bins = [18, 25, 35, 50, 100]
# labels = ['Young', 'Adult', 'Middle-Aged', 'Senior']

# # Create a new column for binned age groups
# df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# # Plot histogram before binning
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)  # First subplot
# sns.histplot(df['Age'], bins=20, kde=True, color='blue')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.title('Histogram of Age (Before Binning)')

# # Plot histogram after binning
# plt.subplot(1, 2, 2)  # Second subplot
# sns.countplot(x=df['Age Group'], palette='coolwarm')
# plt.xlabel('Age Group')
# plt.ylabel('Count')
# plt.title('Histogram of Age Groups (After Binning)')

# plt.tight_layout()  # Adjust layout for better visibility
# plt.show()







# from here box plot of fares 
#df['Fare'].fillna(df['Fare'].mean(), inplace=True)

# # Create a box plot for Fare
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=df['Fare'], color='lightblue')

# # Set title and labels
# plt.title('Box Plot of Fare (Identifying Outliers)')
# plt.xlabel('Fare')

# # Show plot
# plt.show()





# from here scatter plot of fares ages and survived 

# Fill missing values in 'Age' and 'Fare' with median values
# df['Age'].fillna(df['Age'].median(), inplace=True)
# df['Fare'].fillna(df['Fare'].median(), inplace=True)

# # Set the figure size
# plt.figure(figsize=(10, 6))

# # Create a scatter plot
# sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette={0: 'red', 1: 'green'}, alpha=0.7)

# # Set title and labels
# plt.title('Scatter Plot of Age vs. Fare (Colored by Survival)', fontsize=14)
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.legend(title='Survived', labels=['Did Not Survive (0)', 'Survived (1)'])

# # Show plot
# plt.show()




#kmean work here 
# Select relevant numerical columns for clustering
df_selected = df[['Age', 'Fare']]

# Handle missing values by filling with median
df_selected.fillna(df_selected.median(), inplace=True)

# Standardize the data for better K-Means performance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# =============================
# 📌 Step 1: Elbow Method to find optimal k
# =============================
wcss = []  # List to store Within-Cluster Sum of Squares (WCSS)
K_range = range(1, 11)  # Test k values from 1 to 10

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
#     kmeans.fit(df_scaled)
#     wcss.append(kmeans.inertia_)  # Inertia = WCSS

# # Plot the Elbow Method graph
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, wcss, marker='o', linestyle='--', color='b')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# =============================
# 📌 Step 2: Apply K-Means with n_clusters=2
# =============================


kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# =============================
# 📌 Step 3: Visualizing Clusters using PCA
# =============================
pca = PCA(n_components=2)  # Reduce dimensions to 2D
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster'], palette=['red', 'blue'], alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=100, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering (2D PCA Projection)')
plt.legend(title='Cluster')
plt.show()