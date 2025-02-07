import matplotlib.pyplot as plt

group_A=[12, 15,14,13,16,18,19,15,14,20,17,14,15,40,45,52,62]
group_B=[12,17, 15,13,19,20,21,18,17,16,15,14,16,15]

plt.boxplot([group_A,group_B],labels=['Group A','Group B'])
plt.title("Comparision in Group A and Group B")
plt.ylabel ("VALUES")