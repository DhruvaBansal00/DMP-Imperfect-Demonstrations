"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

# print(np.load("2.npz")["arr_0"])
total_timesteps = 50
y_des = np.load("optimal_traj.npy")[2*total_timesteps: 3*total_timesteps].T
print(y_des.shape)
# y_des -= y_des[:, 0][:, None]

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=14, n_bfs=5000, ay=np.ones(14) * 50.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=False)
y_track, dy_track, ddy_track = dmp.rollout(timesteps=total_timesteps)
y_des = y_des.T
plt.figure(1)
# # print(y_track)

plt.plot([i for i in range(0, total_timesteps)], y_track[:, 1], "b", lw=2)
plt.title("DMP system - dmp_traj")

# plt.axis("equal")
# plt.xlim([-2, 2])
# plt.ylim([-2, 2])
plt.show()
# print(y_track)
# print(y_track.shape)
# print(y_des.shape)

plt.plot([i for i in range(0, total_timesteps)], y_des[:, 1], "b", lw=2)
plt.title("DMP system - optimal_traj")

# plt.axis("equal")
# plt.xlim([-2, 2])
# plt.ylim([-2, 2])
plt.show()
