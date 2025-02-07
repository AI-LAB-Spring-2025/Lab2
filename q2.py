import numpy as np
import matplotlib.pyplot as plt

with open("helix_sequences.txt", "r") as file:
    genome_sequence = file.read().strip()  


genome_list = list(genome_sequence)
genome_length = len(genome_list)


t = np.linspace(0, 4 * np.pi, genome_length)  
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length)  


color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}
colors = [color_map.get(base, 'black') for base in genome_list]  
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x, y, z, c=colors, s=100, edgecolors='k')  # s controls the size of the dots


ax.set_title("3D Genome Visualization on a Helix")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


plt.show()