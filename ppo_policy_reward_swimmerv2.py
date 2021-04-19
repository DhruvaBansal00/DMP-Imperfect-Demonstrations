# Training command: python3 -m spinup.run ppo --hid "[64,64]" --env Swimmer-v2 --exp_name ppo_swimmer_v2_300 --epochs 300
# Agent demo: python3 -m spinup.run test_policy /home/dhruva/spinningup/data/ppo_swimmer_v2_300/ppo_swimmer_v2_300_s0

import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import tqdm
import matplotlib.pyplot as plt

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Swimmer-v2_mixture.npy'
PPO_policy_location = '/home/dhruva/spinningup/data/ppo_swimmer_v2_300/ppo_swimmer_v2_300_s0'
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
    qpos = first_traj_init[:5]
    qvel = first_traj_init[5:10]
    environment.set_state(qpos, qvel)
    return get_obs(first_traj_init[2:6], first_traj_init[6:10])

def get_reward(start_timestep, max_timesteps=-1, get_augmentations=False, num_trajectories=10):
    env = gym.make("Swimmer-v2")
    observation = reset_gym_env(env, trajectory_location, start_timestep)
    traj_iteration = 0
    total_reward = [0]
    total_length = []
    all_trajectories = []
    curr_trajectory = [list(observation)]
    while traj_iteration < num_trajectories:
        action = ppo_policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward[traj_iteration] += reward
        curr_trajectory[-1].extend(list(action))
        curr_trajectory.append(list(observation))

        if done or len(curr_trajectory) == max_timesteps:
            while len(curr_trajectory) < max_timesteps:
                curr_trajectory.append([0 for _ in range(len(action) + len(observation))])
            
            curr_trajectory = curr_trajectory[:-1]
            total_length.append(len(curr_trajectory))
            traj_iteration += 1
            observation = reset_gym_env(env, trajectory_location, start_timestep)
            all_trajectories.append(curr_trajectory)
            curr_trajectory = [list(observation)]
            total_reward.append(0)
    env.close()
    if get_augmentations:
        return np.average(total_reward), np.average(total_length), all_trajectories, total_reward
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
    base_directory = "Augmentations/Swimmer-v2/"

    total_timesteps = int(np.load(trajectory_location).shape[0])

    timestep_rewards = [0 for i in range(total_timesteps)]
    timestep_lengths = [0 for i in range(total_timesteps)]

    for augmentation_iter in range(num_iterations):
        
        curr_augmentation_all_timesteps = []
        for i in tqdm.tqdm(range(total_timesteps)):
            curr_reward, curr_length, curr_trajectory, _ = get_reward(i, 50, True)
            timestep_rewards[i] += curr_reward
            timestep_lengths[i] += curr_length
            curr_augmentation_all_timesteps.append(curr_trajectory)
        curr_augmentation_all_timesteps = np.array(curr_augmentation_all_timesteps)
        print(curr_augmentation_all_timesteps.shape)
        np.savez_compressed(f'{base_directory}augmentation_{augmentation_iter}.pkl', curr_augmentation_all_timesteps)
    
    timestep_rewards /= num_iterations
    timestep_lengths /= num_iterations
    print(np.average(timestep_lengths))
        
    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()
    plt.plot([i for i in range(total_timesteps)], timestep_lengths)
    plt.show()

save_all_augmentations()