import numpy as np
import matplotlib.pyplot as plt


traj_file = np.load('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Swimmer-v2_mixture_conf.npy')
first_traj_opt = np.load('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Swimmer-v2_mixture.npy')
print(first_traj_opt.shape)
print(first_traj_opt[0])
# print(np.load('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Hopper-v2_mixture.npy')[0])
# np.save(open("optimal_traj.npy", "wb"), first_traj_opt)
# print(traj_file)
plt.plot(traj_file)
plt.ylabel('some numbers')
plt.show()