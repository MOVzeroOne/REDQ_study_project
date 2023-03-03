from discrete_ram_noskip_truncated_rewards_REDQ import policy_network
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D
import gymnasium as gym
from wrappers import one_life

def inference(policy_net:policy_network,env:gym.Env) -> int:
    state,_ = env.reset()
    cummulative_reward = 0 

    with torch.no_grad():
            
        while(True):
            _,dist = policy_net(torch.tensor(state,dtype=torch.float).view(1,-1))
            action = dist.sample().item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            cummulative_reward += reward

            if(done):
                break 
    
    return cummulative_reward


if __name__ == "__main__":
    #hyperparam
    obs_size = 128
    action_size = 6
    struct_body = [100,100,100,100]
    file_name = "./checkpoints/num_estimates_25step_8600_mean54.0_median60.0_std15.0_max90.0_min30.0.pt"
    env_name ="SpaceInvaders-ramNoFrameskip-v4"
    load_param = True
    only_one_Life = True 

    #init
    #load parameters
    if(load_param):
        parameters = torch.load(file_name)
    policy_net = policy_network(obs_size,action_size,struct_body)

    if(load_param):   
        policy_net.load_state_dict(parameters)

    #make env
    if(only_one_Life):
        env = one_life(gym.make(env_name,obs_type="ram",repeat_action_probability=0,full_action_space=False,render_mode="human"))
    else:
        env = gym.make(env_name,obs_type="ram",repeat_action_probability=0,full_action_space=False,render_mode="human")
    #run
    for _ in range(1000):
        print(inference(policy_net,env))
