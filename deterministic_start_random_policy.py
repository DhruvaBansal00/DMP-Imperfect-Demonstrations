import gym
import numpy as np

def reset_gym_env(environment, traj_file):
    observation = environment.reset()
    first_traj_opt = np.load(traj_file)[:2998]
    first_traj_init = first_traj_opt[0]

    old_state = env.state_vector()
    qpos = np.append(old_state[:2], first_traj_init[:13])
    qvel = first_traj_init[14:28]
    env.set_state(qpos, qvel)


env = gym.make("Ant-v2")
reset_gym_env(env, '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Ant-v2_mixture.npy')
for _ in range(1000):
  env.render()
  # print(env.action_space)
  action = env.action_space.sample() # your agent here (this takes random actions)
  # deterministic_action = np.array([-0.8508554, -0.40371445, 0.17282555, -0.23918863, 0.8985878, 0.26646522, 0.679272, 0.8176589])
  observation, reward, done, info = env.step(action)

  if done:
    reset_gym_env(env, '/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/2IWIL_Repo/demonstrations/Ant-v2_mixture.npy')
    print("NEW")
env.close()