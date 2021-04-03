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

def display_tra(original, new, timesteps):
	plt.figure(1)
	# # print(y_track)

	plt.plot([i for i in range(0, timesteps)], new[:, 1], "b", lw=2)
	plt.title("DMP system - dmp_traj")

	# plt.axis("equal")
	# plt.xlim([-2, 2])
	# plt.ylim([-2, 2])
	plt.show()
	# print(y_track)
	# print(y_track.shape)
	# print(y_des.shape)

	plt.plot([i for i in range(0, timesteps)], original[:, 1], "b", lw=2)
	plt.title("DMP system - optimal_traj")

	# plt.axis("equal")
	# plt.xlim([-2, 2])
	# plt.ylim([-2, 2])
	plt.show()

def analyze_dmp_MSE():
	timestep_options = [50, 100, 200, 400, 800, 1600, 2998]
	MSE = []
	for i in timestep_options:
		total_timesteps = i
		y_des = np.load("optimal_traj.npy")[0: total_timesteps].T
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

		diff = np.mean((y_track - y_des)**2)
		print("MSE = " + str(diff))
		MSE.append(diff)
		if total_timesteps == 50:
			display_tra(y_des, y_track, total_timesteps)

	plt.figure(1)
	plt.plot(timestep_options, MSE, "b", lw=2)
	plt.title("MSE vs Timestep")
	plt.show()

def test_dmp_diff_state():
	total_timesteps = 50
	y_des = np.load("optimal_traj.npy")[2*total_timesteps: 3*total_timesteps].T
	print(y_des.shape)
	dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=14, n_bfs=5000, ay=np.ones(14) * 50.0)
	y_track = []
	dy_track = []
	ddy_track = []

	dmp.imitate_path(y_des=y_des, plot=False)
	y_track, dy_track, ddy_track = dmp.rollout(timesteps=total_timesteps, curr_state=np.load("optimal_traj.npy")[50*total_timesteps])
	y_des = y_des.T
	y_ground_truth = np.load("optimal_traj.npy")[50*total_timesteps:51*total_timesteps]

	diff = np.mean((y_track - y_ground_truth)**2)
	print("MSE = " + str(diff))
	display_tra(y_ground_truth, y_track, total_timesteps)


def analyze_dmp_MSE_diff_state():
	total_timesteps = 50
	y_train = np.load("optimal_traj.npy")[: total_timesteps].T
	dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=14, n_bfs=5000, ay=np.ones(14) * 50.0)
	y_track = []
	dy_track = []
	ddy_track = []
	dmp.imitate_path(y_des=y_train, plot=False)

	MSE = []
	complete_traj = np.load("optimal_traj.npy")
	for i in range(50):
		y_track, dy_track, ddy_track = dmp.rollout(timesteps=total_timesteps, curr_state=complete_traj[i*total_timesteps])
		y_ground_truth = complete_traj[i*total_timesteps:(i+1)*total_timesteps]
		diff = np.mean((y_track - y_ground_truth)**2)
		MSE.append(diff)
	print(MSE)

	plt.figure(1)
	plt.plot([i for i in range(50)], MSE, "b", lw=2)
	plt.title("MSE vs start state (X 50)")
	plt.show()
		# print("MSE = " + str(diff))
	# display_tra(y_ground_truth, y_track, total_timesteps)
# test_dmp_diff_state()
analyze_dmp_MSE_diff_state()