import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

#reading file into genome sequence 

myFile = open ("genome.txt","r")
data=myFile.read()

genome_sequence= data

list_genome_sequence=list(genome_sequence)

genome_length=len( list_genome_sequence)
#print("the length is ",genome_length)

# We'll use the parametric equations for a helix:
# x = cos(t), y = sin(t), z = t (or a scaled version of t)
# We want to span a range so that the helix makes a few turns.
t = np.linspace(0, 4 * np.pi, genome_length) 
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length) 
# Combine the coordinates into a (genome_length x 3) array
coordinates = np.column_stack((x, y, z))

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the helix
ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])

#color mapping
colors = {'A':'red','T':'blue','C':'green', 'G':'orange'}


plt.show()
