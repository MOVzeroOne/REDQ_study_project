import gymnasium as gym
 


env_name ="SpaceInvaders-ramDeterministic-v4"

env = gym.make(env_name)

print(env.unwrapped._frameskip)

env_name ="SpaceInvaders-ramNoFrameskip-v4"

env = gym.make(env_name)

print(env.unwrapped._frameskip)

env_name ="ALE/SpaceInvaders-v5"

env = gym.make(env_name)

print(env.unwrapped._frameskip)