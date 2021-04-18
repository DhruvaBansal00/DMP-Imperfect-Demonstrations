# Training command: python3 -m spinup.run ppo --hid "[64,64]" --env Swimmer-v2 --exp_name ppo_swimmer_v2_300 --epochs 300
# Agent demo: python3 -m spinup.run test_policy /home/dhruva/spinningup/data/ppo_swimmer_v2_300/ppo_swimmer_v2_300_s0

import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import tqdm
import matplotlib.pyplot as plt

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Walker2d-v2_mixture.npy'
PPO_policy_location = '/home/dhruva/spinningup/data/ppo_walker_v2_300/ppo_walker_v2_300_s0'
_, ppo_policy = load_policy_and_env(PPO_policy_location)

def get_obs(qpos, qvel):
    position = qpos
    velocity = qvel
    observations = np.concatenate((position, velocity))
    return observations

def reset_gym_env(environment, traj_file, start_timestep):
    observation = environment.reset()
    first_traj_init = np.load(traj_file)[start_timestep]
    old_state = environment.state_vector()
    qpos = np.append(old_state[0], first_traj_init[:8])
    qvel = first_traj_init[8:17]
    environment.set_state(qpos, qvel)
    return get_obs(first_traj_init[:8], qvel)

def get_reward(start_timestep):
    env = gym.make("Walker2d-v2")
    observation = reset_gym_env(env, trajectory_location, start_timestep)
    traj_iteration = 0
    total_reward = 0
    while traj_iteration < 10:
        action = ppo_policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            traj_iteration += 1
            observation = reset_gym_env(env, trajectory_location, start_timestep)
    env.close()
    return total_reward/traj_iteration

def plot_reward_vs_timestep():
    total_timesteps = int(np.load(trajectory_location).shape[0]/100)

    timestep_rewards = []
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward = get_reward(i*100)
        print(f'iteration {i} reward {curr_reward}')
        timestep_rewards.append(curr_reward)

    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()

def visualize_timestep_initialization(start_timestep):
    env = gym.make("Walker2d-v2")
    observation = reset_gym_env(env, trajectory_location, start_timestep)
    traj_iteration = 0
    total_reward = 0
    while traj_iteration < 10:
        env.render()
        action = ppo_policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            traj_iteration += 1
            observation = reset_gym_env(env, trajectory_location, start_timestep)
    env.close()
    return total_reward/traj_iteration

visualize_timestep_initialization(1*100)

