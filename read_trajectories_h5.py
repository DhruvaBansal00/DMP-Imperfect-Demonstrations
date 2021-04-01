import numpy as np
import matplotlib.pyplot as plt
import h5py

# traj_file = np.load('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Hopper-v2_mixture_conf.npy')
# # print(traj_file)
# plt.plot(traj_file)
# plt.ylabel('some numbers')
# plt.show()

f = h5py.File('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/VILD_Demos/imitation_data/TRAJ_h5/Ant-v2/Ant-v2_TRAJ-N1000_normal0.01.h5', 'r')
# print(list(f.keys()))
expert_states = f['expert_states']
print(expert_states[0])
print(f['expert_actions'][0])
print(len(expert_states[0]) + len(f['expert_actions'][0]))