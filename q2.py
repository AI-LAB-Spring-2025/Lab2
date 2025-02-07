import numpy as np
import matplotlib.pyplot as plt
import random
f = open("genome.txt", "r")

genome_sequence = f.read()

print(genome_sequence)
l1 = list(genome_sequence)
print(l1)
print(len(l1))
genome_lenght = len(l1)

t = np.linspace(0, 4*np.pi, genome_lenght)
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_lenght)
coordenate = np.column_stack((x, y, z))
print(coordenate)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'yellow'}
genome = ""
for _ in range(genome_lenght):
    genome += random.choice(l1)

for i, l1 in enumerate(genome):
    ax.scatter(x[i], y[i], z[i], c=colors[l1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(' 3D Genome Visualization on a Helix')

plt.show()
