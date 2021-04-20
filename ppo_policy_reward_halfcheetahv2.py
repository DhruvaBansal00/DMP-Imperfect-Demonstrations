# Training command: python3 -m spinup.run ppo --hid "[64,64]" --env Swimmer-v2 --exp_name ppo_swimmer_v2_300 --epochs 300
# Agent demo: python3 -m spinup.run test_policy /home/dhruva/spinningup/data/ppo_swimmer_v2_300/ppo_swimmer_v2_300_s0

import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import tqdm
import matplotlib.pyplot as plt

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/HalfCheetah-v2_mixture.npy'
PPO_policy_location = '/home/dhruva/spinningup/data/ppo_halfcheetah_v2_100/ppo_halfcheetah_v2_100_s0'
_, ppo_policy = load_policy_and_env(PPO_policy_location)
env = gym.make("HalfCheetah-v2")

def get_obs(qpos, qvel):
    position = qpos
    velocity = qvel
    observations = np.concatenate((position, velocity))
    return observations

def reset_gym_env(traj_file, start_timestep):
    observation = env.reset()
    first_traj_init = np.load(traj_file)[start_timestep]
    old_state = env.state_vector()
    qpos = np.append(np.array([old_state[0]]), first_traj_init[:8])
    qvel = first_traj_init[8:17]
    env.set_state(qpos, qvel)
    return get_obs(first_traj_init[:8], first_traj_init[8:17])

def get_reward(start_timestep, max_timesteps=-1, get_augmentations=False, num_trajectories=10):
    
    observation = reset_gym_env(trajectory_location, start_timestep)
    traj_iteration = 0
    total_reward = [0]
    total_length = []
    all_states = []
    all_actions = []
    curr_states = [list(observation)]
    curr_actions = []
    while traj_iteration < num_trajectories:
        action = ppo_policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward[traj_iteration] += reward
        curr_actions.append(list(action))
        curr_states.append(list(observation))

        if done or len(curr_states) == max_timesteps:
            while len(curr_states) < max_timesteps:
                curr_states.append([1 for _ in range(len(observation))])
                curr_actions.append([1 for _ in range(len(action))])
            
            curr_states = curr_states[:-1]
            total_length.append(len(curr_states))
            traj_iteration += 1
            observation = reset_gym_env(trajectory_location, start_timestep)
            all_states.append(curr_states)
            all_actions.append(curr_actions)
            curr_states = [list(observation)]
            curr_actions = []
            total_reward.append(0)
    
    if get_augmentations:
        return np.average(total_reward), np.average(total_length), all_states, all_actions, total_reward
    return np.average(total_reward), np.average(total_length)

def plot_reward_vs_timestep():

    total_timesteps = int(np.load(trajectory_location).shape[0]/100)

    timestep_rewards = []
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward = get_reward(i*100)
        timestep_rewards.append(curr_reward)

    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()

def save_all_augmentations(num_iterations=10):
    base_directory = "Augmentations/HalfCheetah-v2/"

    total_timesteps = int(np.load(trajectory_location).shape[0])

    timestep_rewards = [0 for i in range(total_timesteps)]
    timestep_lengths = [0 for i in range(total_timesteps)]
        
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward, curr_length, all_states, all_actions, _ = get_reward(i, 51, True, 10)
        timestep_rewards[i] += curr_reward
        timestep_lengths[i] += curr_length
        np.savez_compressed(f'{base_directory}all_states_{i}.npz', all_states)
        np.savez_compressed(f'{base_directory}all_actions_{i}.npz', all_actions)
    
    print(np.average(timestep_lengths))
        
    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()
    plt.plot([i for i in range(total_timesteps)], timestep_lengths)
    plt.show()

save_all_augmentations()
env.close()