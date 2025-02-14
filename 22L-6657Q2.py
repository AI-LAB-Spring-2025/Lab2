import numpy as np
import matplotlib.pyplot as plt

with open('genome.txt', 'r') as file:
    content = file.read()
genome_sequence = content
print(genome_sequence)
print(len(genome_sequence))

t = np.linspace(0, 4 * np.pi, len(genome_sequence))
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, len(genome_sequence))

coordinates = np.column_stack((x, y, z))
colors = np.random.choice(['red', 'green', 'blue', 'orange'], len(genome_sequence))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, c=colors)
plt.title('3D Scatter with Random Colors')
plt.show()