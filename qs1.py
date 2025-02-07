import numpy as np
import matplotlib.pyplot as plt

group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

fig=plt.figure(figsize=(10, 5))
axes= fig.subplots(1,2)

axes[0].boxplot(group_A)
axes[0].set_title("Group A")
axes[0].set_ylabel("values")


axes[1].boxplot(group_B)
axes[1].set_title("Group B")
axes[1].set_ylabel("values")


plt.suptitle("BOX PLOTS")


plt.show()
