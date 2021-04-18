# Training command: python3 -m spinup.run ppo --hid "[64,64]" --env Swimmer-v2 --exp_name ppo_swimmer_v2_300 --epochs 300



import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import tqdm
import matplotlib.pyplot as plt
import pickle

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Ant-v2_mixture.npy'
PPO_policy_location = '/home/dhruva/spinningup/data/ppo_ant_v2/ppo_ant_v2_s0'
_, ppo_policy = load_policy_and_env(PPO_policy_location)

def get_obs(qpos, qvel, cf):
    position = qpos
    velocity = qvel
    contact_force = cf
    observations = np.concatenate((position, velocity, contact_force))
    return observations

def reset_gym_env(environment, traj_file, start_timestep):
    observation = environment.reset()
    first_traj_init = np.load(traj_file)[start_timestep]
    old_state = environment.state_vector()
    qpos = np.append(old_state[:2], first_traj_init[:13])
    qvel = first_traj_init[13:27]
    environment.set_state(qpos, qvel)
    return get_obs(first_traj_init[:13], qvel, observation[27:])

def get_reward(start_timestep, max_timesteps=-1, get_augmentations=False):
    env = gym.make("Ant-v2")
    observation = reset_gym_env(env, trajectory_location, start_timestep)
    traj_iteration = 0
    total_reward = [0]
    total_length = []
    curr_length = 0
    all_trajectories = []
    curr_trajectory = [list(observation)]
    while traj_iteration < 10:
        action = ppo_policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward[traj_iteration] += reward
        curr_length += 1
        curr_trajectory.append(list(observation))

        if done or curr_length == max_timesteps:
            total_length.append(curr_length)
            curr_length = 0
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
    timestep_lengths = []
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward, curr_length = get_reward(i*100)
        timestep_rewards.append(curr_reward)
        timestep_lengths.append(curr_length)
    
    print(np.average(timestep_lengths))
    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()
    plt.plot([i for i in range(total_timesteps)], timestep_lengths)
    plt.show()

def plot_all_reward_vs_timestep():
    total_timesteps = int(np.load(trajectory_location).shape[0])

    timestep_rewards = []
    timestep_lengths = []
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward, curr_length = get_reward(i, 50)
        timestep_rewards.append(curr_reward)
        timestep_lengths.append(curr_length)
    
    print(np.average(timestep_lengths))
        
    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()
    plt.plot([i for i in range(total_timesteps)], timestep_lengths)
    plt.show()

    with open('antv2_all_rewards.npy', 'wb') as f:
        np.save(f, timestep_rewards)
    with open('antv2_all_lenghts.npy', 'wb') as f:
        np.save(f, timestep_lengths)

def save_all_augmentations():
    base_directory = "Augmentations/Ant-v2/"

    total_timesteps = int(np.load(trajectory_location).shape[0])

    timestep_rewards = []
    timestep_lengths = []
    for i in tqdm.tqdm(range(total_timesteps)):
        curr_reward, curr_length, all_trajectories, all_rewards = get_reward(i, 50, True)
        timestep_rewards.append(curr_reward)
        timestep_lengths.append(curr_length)
        pickle.dump(all_trajectories, open(f'{base_directory}iteration_{i}_augmentations.pkl', 'wb'))
        pickle.dump(all_rewards, open(f'{base_directory}iteration_{i}_rewards.pkl', 'wb'))
    
    print(np.average(timestep_lengths))
        
    plt.plot([i for i in range(total_timesteps)], timestep_rewards)
    plt.show()
    plt.plot([i for i in range(total_timesteps)], timestep_lengths)
    plt.show()

    with open('antv2_all_rewards.npy', 'wb') as f:
        np.save(f, timestep_rewards)
    with open('antv2_all_lenghts.npy', 'wb') as f:
        np.save(f, timestep_lengths)

save_all_augmentations()