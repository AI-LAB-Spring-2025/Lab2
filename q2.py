import numpy as np
import matplotlib.pyplot as plt

FILENAME = "genome.txt"

def read_from_file(file_name: str):
    with open(file_name, 'r') as file:
        return file.read().strip()

def main():
    global FILENAME
    
    file_name = FILENAME
    helix_str = read_from_file(file_name)
    helix_str_len = len(helix_str)

    t = np.linspace(0, 4*np.pi, helix_str_len)

    x = np.cos(t)
    y = np.sin(t)
    z = np.linspace(0, 5, helix_str_len)

    coordinates = np.column_stack((x, y, z))

    fig =  plt.figure()
    ax = fig.add_subplot(projection="3d")

    colors = {"A": "red", "T": "blue", "C": "green", "G": "yellow"}


    for i, nucleotide in enumerate(helix_str):
        ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], color=colors[nucleotide], marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DNA Helix")
    plt.show()


if __name__ == "__main__":
    main()