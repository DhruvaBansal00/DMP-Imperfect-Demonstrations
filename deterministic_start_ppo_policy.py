import gym
import numpy as np
from spinup.utils.test_policy import load_policy_and_env

trajectory_location = '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Ant-v2_mixture.npy'
PPO_policy_location = '/home/dhruva/spinningup/data/ppo_ant_v2/ppo_ant_v2_s0'

def get_obs(qpos, qvel, cf):
    position = qpos
    velocity = qvel
    contact_force = cf
    observations = np.concatenate((position, velocity, contact_force))
    return observations

def reset_gym_env(environment, traj_file):
    observation = environment.reset()
    first_traj_opt = np.load(traj_file)[:2998]
    first_traj_init = first_traj_opt[0]
    old_state = env.state_vector()
    qpos = np.append(old_state[:2], first_traj_init[:13])
    qvel = first_traj_init[13:27]
    env.set_state(qpos, qvel)
    return get_obs(first_traj_init[:13], qvel, observation[27:])


env = gym.make("Ant-v2")
observation = reset_gym_env(env, trajectory_location)
_, ppo_policy = load_policy_and_env(PPO_policy_location)
for _ in range(10000):
  env.render()
  action = ppo_policy(observation)
  observation, reward, done, info = env.step(action)

  if done:
    observation = reset_gym_env(env, trajectory_location)
    print("NEW EPISODE")
env.close()