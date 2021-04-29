import argparse
from itertools import count

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math
import tqdm

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
    action_mean, _, action_std = network(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action.data[0].numpy()

def create_video():
	env = gym.make('Ant-v2')
	num_inputs = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]


	policy_net = Policy(num_inputs, num_actions, 100)
	policy_net.load_state_dict(torch.load('log/ICGAIL_ANT/Ant-v2_1111_mixture_0.2_600.pt'))
	policy_net.eval()

	env = gym.make("Ant-v2")
	state = env.reset()
	traj_iteration = 0
	total_reward = 0
	while traj_iteration < 10:
	    env.render()
	    action = select_action(state, policy_net)
	    state, reward, done, info = env.step(action)
	    total_reward += reward
	    if done:
	        traj_iteration += 1
	        observation = env.reset()
	env.close()
	print(f' Average Reward: {total_reward/traj_iteration}')

def create_curve():
	learning_curve_file = open('log/ICGAIL_ANT/Ant-v2_1111_mixture_0.2000_600_0.7_ucGAIL.csv', 'r')
	X = []
	Y = []

	curve_file = learning_curve_file.readlines()
	for line in curve_file:
		line_split = line.split(',')
		X.append(float(line_split[0]))
		Y.append(float(line_split[1])/5000)

	plt.plot(X, Y)
	plt.show()

create_video()