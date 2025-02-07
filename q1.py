import matplotlib.pyplot as plt
import numpy as np

group_A = np.array([12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62])
group_B = np.array([12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15])

plt.figure(figsize=(12, 6))

# Box plot for Group A
plt.subplot(1, 2, 1)
plt.boxplot(group_A)
plt.title("Box Plot for Group A")
plt.ylabel("Measurement Values")

# Box plot for Group B
plt.subplot(1, 2, 2)
plt.boxplot(group_B)
plt.title("Box Plot for Group B")
plt.ylabel("Measurement Values")

# Overall figure title
plt.suptitle("Comparison of Measurements for Group A and Group B")

plt.show()