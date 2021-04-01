import numpy as np
import matplotlib.pyplot as plt


traj_file = np.load('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Hopper-v2_mixture_conf.npy')
# print(traj_file)
plt.plot(traj_file)
plt.ylabel('some numbers')
plt.show()