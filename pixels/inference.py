from SAC_REDQ_pixels.discrete_pixel_REDQ import policy_network
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D
import gymnasium as gym
from wrappers import one_life,AtariPreprocessing
from gymnasium.wrappers import FrameStack,ResizeObservation


def inference(policy_net:policy_network,env:gym.Env) -> int:
    state = env.reset()
    cummulative_reward = 0 

    with torch.no_grad():
            
        while(True):
            #_,dist = policy_net(torch.tensor(state,dtype=torch.float).view(1,-1))
            #action = dist.sample().item()
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            cummulative_reward += reward

            if(done):
                break 
    
    return cummulative_reward


def env_space_invaders(render_mode="human"):
    env = gym.make("ALE/SpaceInvaders-v5",obs_type="grayscale",repeat_action_probability=0,full_action_space=False,render_mode=render_mode) #already skips 4 frames
    env = ResizeObservation(env,(84,84))
    return one_life(FrameStack(env, 4))

if __name__ == "__main__":
    #hyperparam
    obs_size = 128
    action_size = 6
    struct_body = [100,100,100]
    file_name = "./checkpoints/num_estimates_10step_2800_mean111.5_median115.0_std21.86448097229004_max140.0_min70.0.pt"
    env_name ="ALE/SpaceInvaders-v5"
    load_param = False

    #init
    #load parameters
    if(load_param):
        parameters = torch.load(file_name)
    policy_net = policy_network(obs_size,action_size,struct_body)

    if(load_param):   
        policy_net.load_state_dict(parameters)

    #make env
    env = env_space_invaders()
    #run
    for _ in range(1000):
        print(inference(policy_net,env))
