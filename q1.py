# import numpy as np

# arr = np.array([1, 20, 19, 21, 4])

# print(arr)

import matplotlib.pyplot as plt

group_a = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
group_b = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]


plt.boxplot([group_a, group_b], labels=['Group A', 'GrouB'])
plt.title('Boxplot of Group A and Group B')
plt.ylabel(' Measured Values')
plt.show()
