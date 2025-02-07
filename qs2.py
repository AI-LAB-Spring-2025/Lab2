import numpy as np
import matplotlib.pyplot as plt

def genome_sequence():
    file=open("helixs.txt", "r")
    g_seq=file.read().strip()
    return list(g_seq)

g_list=genome_sequence()
length=len(g_list)

t = np.linspace(0, 4 * np.pi, length)     # 4*pi gives about
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, length)            # z increases linearly to
coordinates = np.column_stack((x, y, z))

color = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}
colors = [color.get(base, 'black') for base in g_list]

# for i, nucleotide in enumerate(genome):
#  ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
#  color=colors[nucleotide], marker='o')



fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot( projection='3d')
ax.scatter(x, y, z, c=colors, s=20)

# Labels and title
ax.set_title("3D Helix Representation of Genome Sequence")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()