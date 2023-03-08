from discrete_pixel_noskip_scaled_rewards_REDQ import policy_network
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D
import gymnasium as gym
from wrappers import one_life
from gymnasium.wrappers import FrameStack,ResizeObservation
import numpy as np 

def inference(policy_net:policy_network,env:gym.Env,device="cuda:0") -> int:
    state,_ = env.reset()
    cummulative_reward = 0 

    with torch.no_grad():
            
        while(True):
            _,dist = policy_net(torch.tensor(np.array(state),dtype=torch.float,device=device).unsqueeze(dim=0))
            action = dist.sample().item()
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            cummulative_reward += reward

            if(done):
                break 
    
    return cummulative_reward


def env_space_invaders(render_mode="human"):
    env = gym.make("SpaceInvadersDeterministic-v4",obs_type="grayscale",repeat_action_probability=0,full_action_space=False,render_mode=render_mode)
    env = ResizeObservation(env,(84,84))
    return one_life(FrameStack(env, 12))

if __name__ == "__main__":
    #hyperparam
    obs_size = 128
    action_size = 6
    struct_body = [100,100,100]
    file_name = "./checkpoints/num_estimates_10step_1400_mean50.0_median55.0_std20.0_max80.0_min15.0.pt"
    load_param = True

    #init
    #load parameters
    if(load_param):
        parameters = torch.load(file_name)
    policy_net = policy_network(obs_size,action_size,struct_body).cuda()

    if(load_param):   
        policy_net.load_state_dict(parameters)

    #make env
    env = env_space_invaders()
    #run
    for _ in range(1000):
        print(inference(policy_net,env))
