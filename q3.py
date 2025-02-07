from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "image.png"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():
    global FILENAME

    image = Image.open(FILENAME)
    numpydata = np.asarray(image)
    
    print(type(numpydata))
    
    plt.imshow(numpydata)
    plt.title("Image Plot")
    plt.show()

    rotated_image = np.rot90(numpydata)

    
    plt.imshow(rotated_image)
    plt.title("Rotated Plot")
    plt.show()

    flipped_image = np.fliplr(numpydata)

    plt.imshow(flipped_image)
    plt.title("Flipped Image Plot")
    plt.show()

    gray = rgb2gray(numpydata)
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    main()