import argparse
from itertools import count

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math
import tqdm
import random 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from loss import *
import matplotlib.pyplot as plt


def select_action(state, network):
	state = torch.from_numpy(state).unsqueeze(0).float()
	action, _, _ = network(Variable(state))
	return action.data[0].numpy()

def create_video():
	env = gym.make('Ant-v2')
	num_inputs = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]


	policy_net = Policy(num_inputs, num_actions, 100)
	policy_net.load_state_dict(torch.load('log/Mixed2IWIL/Ant-v2_1111_True_mixture_0.2_4000.pt'))
	# policy_net.load_state_dict(torch.load('log/ICGAIL_ANT/Ant-v2_1111_mixture_0.2_600.pt'))
	# policy_net.load_state_dict(torch.load('log/2IWIL_ANT/Ant-v2_1111_True_mixture_0.2_600.pt'))
	policy_net.eval()

	env = gym.make("Ant-v2")
	state = env.reset()
	traj_iteration = 0
	total_reward = []
	curr_reward = 0
	while traj_iteration < 3:
		env.render()
		action = select_action(state, policy_net)
		state, reward, done, info = env.step(action)
		curr_reward += reward
		if done:
			total_reward.append(curr_reward)
			curr_reward = 0
			traj_iteration += 1
			observation = env.reset()
	env.close()
	print(f' Average Reward: {np.mean(total_reward)} Standard Deviation: {np.std(total_reward)}')

def create_curve():
	learning_curve_file = open('log/Mixed2IWIL/Ant-v2_1111_weight_mixture_0.2000_4000.csv', 'r')
	# learning_curve_file = open('log/ICGAIL_ANT/Ant-v2_1111_mixture_0.2000_600_0.7_ucGAIL.csv', 'r')
	# learning_curve_file = open('log/2IWIL_ANT/Ant-v2_1111_weight_mixture_0.2000_600.csv', 'r')
	X = []
	Y = []
	Error = []
	curr_buffer = []

	curve_file = learning_curve_file.readlines()
	for line in curve_file:
		line_split = line.split(',')
		# X.append(float(line_split[0]))
		# Y.append(float(line_split[1]))
		if len(curr_buffer) < 100:
			curr_buffer.append(float(line_split[1]))
		else:
			X.append(float(line_split[0]) - 5)
			Y.append(np.mean(curr_buffer))
			# print(curr_buffer)
			# print(np.std(curr_buffer))
			Error.append(np.std(curr_buffer))
			curr_buffer = []

	plt.errorbar(X, Y, yerr=Error, c=(random.random(), random.random(), random.random()), ecolor=(random.random(), random.random(), random.random()), capsize=2.0)
	plt.xlabel("Epochs", fontsize=24)
	plt.ylabel("Average Return (100 games)", fontsize=24)
	plt.ylim([-1000, 5000])
	plt.yticks(fontsize=20)
	plt.xlim([0, 5000])
	plt.xticks(fontsize=20)
	plt.title("Mix2IWIL Learning Curve", fontsize=24)
	plt.show()

create_curve()