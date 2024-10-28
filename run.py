import gymnasium as gym
import dsrl

# Create the environment
env = gym.make('OfflineCarGoal2Gymnasium-v0')

# Each task is associated with a dataset
# dataset contains observations, next_observatiosn, actions, rewards, costs, terminals, timeouts
dataset = env.get_dataset()
print(dataset.keys())
print(dataset['terminals'])
print(dataset['timeouts'])
print(dataset['observations'].shape) # An N x obs_dim Numpy array of observations

# dsrl abides by the OpenAI gym interface
obs, info = env.reset()
infos = env.step(env.action_space.sample())
# cost = infos["cost"]
print(infos)
print(cost)
print(obs)
# Apply dataset filters [optional]
# dataset = env.pre_process_data(dataset, filter_cfgs)