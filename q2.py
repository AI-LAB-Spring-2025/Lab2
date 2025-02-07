import matplotlib.pyplot as plt
import numpy as np

with open("genome.txt", "r") as file:
    genome_sequence = file.read().strip() 

genome_list = list(genome_sequence)
genome_length = len(genome_list)

# Define helix parameters
t = np.linspace(0, 4 * np.pi, genome_length)  # 4π gives ~2 turns
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length)  # Spread helix vertically

# Color mapping for nucleotides
colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'orange'}

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

# Plot each nucleotide with its respective color
for i, nucleotide in enumerate(genome_list):
    ax.scatter(x[i], y[i], z[i], color=colors.get(nucleotide, 'black'), marker='o', label=nucleotide if i == 0 else "")

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Helix Structure of Genome Sequence")

plt.show()
