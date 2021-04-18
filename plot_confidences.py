import numpy as np
import matplotlib.pyplot as plt
import tqdm

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Ant-v2_mixture.npy'
confidence_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Walker2d-v2_mixture_conf.npy'

total_timesteps = int(np.load(confidence_location).shape[0]/100)

timestep_confidence = []
for i in tqdm.tqdm(range(total_timesteps)):
    curr_conf = np.load(confidence_location)[i*100]
    timestep_confidence.append(curr_conf[0])

print(timestep_confidence)
plt.plot([i for i in range(total_timesteps)], timestep_confidence)
plt.show()
