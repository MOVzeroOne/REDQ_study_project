import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D
import gymnasium as gym
from collections import deque 
import numpy as np 
from copy import deepcopy
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import os
from wrappers import one_life,scaled_rewards
from gymnasium.wrappers import FrameStack,ResizeObservation
import psutil 

class experience_buffer():
    """
    Stores (state, next_state, action, reward, done) tuples in a cyclic memory buffer
    and allows for easy sampling 
    """
    def __init__(self,maxlen:int,device:str = "cpu") -> None:
        self.buffer = deque(maxlen=maxlen)
        self.device = device 
    
    def append(self,experience:tuple) -> None:
        """
        experience: (state, next_state, action, reward, done)
        """
        self.buffer.append(experience)
    
    def sample(self,batch_size:int) -> tuple:
        """
        input:
        returns: (states, next_states, actions, rewards, dones)
        Where the size of each element of the tuple is equal to the batchsize
        """
        indexes = np.random.choice(np.arange(0,len(self.buffer)),batch_size)
        states, next_states, actions, rewards, dones = list(zip(*[self.buffer[index] for index in indexes]))
        return (self.cat(states).to(self.device,dtype=torch.float), self.cat(next_states).to(self.device,dtype=torch.float), self.cat(actions).to(self.device,dtype=torch.long), self.cat(rewards).to(self.device,dtype=torch.float), self.cat(dones).to(self.device,dtype=torch.bool))
    
    def __len__(self) -> int:
        """
        returns the length of the experience buffer
        """
        return len(self.buffer)
    
    def cat(self,x:list) -> torch.Tensor:
        """
        concatentates everything in a list whether from numpy or torch.
        As long as the type is the same over the entire list
        """
        if(type(x[0]) is np.ndarray):
            return torch.cat([torch.from_numpy(elem).unsqueeze(dim=0) for elem in x],dim=0)
        
        elif(type(x[0]) is torch.Tensor):
            return torch.cat([elem.unsqueeze(dim=0) for elem in x],dim=0)
        
        elif(type(x[0]) is int or type(x[0]) is float or type(x[0]) is bool):
            return torch.cat([torch.tensor(elem).unsqueeze(dim=0).unsqueeze(dim=1) for elem in x],dim=0)
        
        elif(type(x[0]) is gym.wrappers.frame_stack.LazyFrames):
            return torch.tensor(np.array(x),dtype=torch.float,device=self.device)
        else:
            print(type(x[0]))
            exit("TYPE UNKNOWN")
            

class mlp(nn.Module):
    def __init__(self,struct:list,act_f:nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.struct = struct 
        self.act_f = act_f
        self.layers = self._build()
    
    def _build(self) -> nn.Sequential:
        layers = []
        for index,(i,j) in enumerate(zip(self.struct[:-1],self.struct[1:])):
            layers.append(nn.Linear(i,j))
            if(not (index == len(self.struct)-2)):#if not last layer
                layers.append(self.act_f())
        return nn.Sequential(*layers)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers(x) 

class critic_network(nn.Module):
    """
    produce Q value for each action given a state.
    This will be usefull during policy optimization,
    as we can directly maximize the innerproduct of the policy distribution with Q+Entropy.
    (maximize so the policy gives the highest Q+entropy)
    arXiv:1910.07207 (discrete soft actor critic)

    """
    def __init__(self,obs_size:int,action_size:int,struct_body:list) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ) 
   
        self.body = mlp([obs_size] + struct_body + [action_size])

    def forward(self,x) -> torch.Tensor:
        return self.body(self.conv(x))

class policy_network(nn.Module):
    def __init__(self,obs_size:int,action_size:int,struct_body:list,softmax_dim:int=1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ) 

        self.body = mlp([obs_size] + struct_body + [action_size])
        self.head = nn.Softmax(softmax_dim)
    
    def forward(self,x:torch.Tensor) -> tuple:
        probs = self.head(self.body(self.conv(x)))
        dist = D.Categorical(probs)
        return probs,dist
         
class discrete_REDQ(nn.Module):
    """
    1 policy network
    N critics in an ensemble 
    M in-target minimization parameter (The size of the subset of the ensemble that gives the minimum Q(a,s))
    G UTD (Update-To-Data) ratio
    ρ target smoothing coefficient for updating the target networks φtarget = ρ*(φtarget) + (1 - ρ)φcritic 
    γ discount 
    
    default values from paper:
        N = 10
        M = 2 
        G = 20 
        ρ = 0.005
        γ = 0.99
        
    1. take action using policy network  a ~ πθ(.|s) 
    2. store (s,a,s',r,done) in buffer #remember that when it is a terminal state Q = r instead of Q = r + gamma*(Q(a',s')+log(πθ(a'|s')))  a' ~ πθ(.|s')
    3. for G updates sample from buffer
    4.      sample M indices to select in-target model's from the target ensemble with.
    5.      sample a' ~ πθ(.|s') 
    6.      grab of these M models the one that has the lowest Q(a',s') 
    7.      now y = r + gamma*(Q(s',a')+H) where H is the entropy H = α*log(πθ(a'|s'))  #note that if it's a terminal state y = r
    8.      #note (we just compute a single y!!!! per element in the batch from the buffer)
    9.      for all N models in the critic ensemble:
    10.         update critics by going for each over the batch of Q(a,s) = y  so MSE(Q(a,s),y)/batch_size note that each a,s from the batch has it's own y. (because a,s -> s' is different)
    11.     update all target networks φtarget = ρ*(φtarget) + (1 - ρ)φcritic 
    12.     update policy network (while not clear in the paper, it is probably during the G loop)
            we can directly maximize the innerproduct of the policy distribution with Q+Entropy. but we do this for each (a,s) of the batch over the entire ensemble. (all summed together in a single loss)
            Then devide by the batch_size and the ensemble size.
            update α 
            
    arXiv:2101.05982 randomized ensembled double Q-learning: learning fast without a model
    arXiv:1910.07207 discrete soft actor critic
    """

    def __init__(self,obs_size:int,action_size:int,policy_body_struct:list,critic_body_struct:list,env:gym.Env,writer:SummaryWriter,optim_kwargs:dict,optimizer:optim=optim.Adam,N:int=20,M:int=2,G:int=5,γ:float=0.99,ρ:float=0.4,α:float=0.2,buffer_size:int=int(1e6),device:str="cuda:0",measure_memory_cuda:bool=True,measure_memory_ram:bool=True) -> None:
        super().__init__()
        self.device = device 
        self.measure_memory_cuda =  measure_memory_cuda
        self.measure_memory_ram = measure_memory_ram
        
        self.buffer_size = buffer_size
        
        self.env = env 
        self.buffer = experience_buffer(maxlen=self.buffer_size,device=self.device)
        self.is_done = True #need to reset to start 
        self.current_state = None 
        
        self.N = N #ensemble size
        self.M = M #in-target minimization parameter
        self.G = G #UTD (Update-To-Data) ratio
        self.γ = γ #discount factor 
        self.ρ = ρ #target smoothing coefficient (how much of the target critic is kept for each update, 0.005 is 0.5% of target critic is kept)
        self.α = α #entropy scaling factor (since entropy is not calculated with log(p) but we have the actual entropy range [0,1], tuning by hand should be fine instead of setting a target entropy and scaling alpha to get that target to be the expected entropy when visiting states)

    

        self.writer = writer
        self.cummulative_reward = 0
        self.Q_mean = 0
        self.chosen_Q_mean = 0
        self.entropy_mean = 0
        self.env_steps = 1
        self.life_num = 0

        self.obs_size = obs_size
        self.action_size = action_size
        self.policy_body_struct = policy_body_struct
        self.critic_body_struct = critic_body_struct

        self.policy_net = policy_network(self.obs_size,self.action_size,self.policy_body_struct).to(self.device)
        self.policy_optimizer = optimizer(self.policy_net.parameters(),**optim_kwargs)
        self.critic_ensemble = None 
        self.target_critic_ensemble = None 
        self.optimizer_list = None 

        self.__build_ensemble() 
        self.__build_optimizers(optimizer,optim_kwargs)

    def __build_optimizers(self,optimizer:optim,optim_kwargs:dict) -> None:
        """
        assigns an optimizer to every critic with learning rate defined in 
        """
        self.optimizer_list = [optimizer(critic.parameters(),**optim_kwargs) for critic in self.critic_ensemble]     

    def __build_ensemble(self) -> None:
        """
        makes an self.critic_ensemble and a self.target_critic_ensemble
        """
        self.critic_ensemble = nn.ModuleList([critic_network(self.obs_size,self.action_size,self.critic_body_struct).to(device=self.device) for _ in range(self.N)])
        self.target_critic_ensemble = [deepcopy(critic) for critic in self.critic_ensemble]
        self.__disable_grads_target_ensemble()

    def __disable_grads_target_ensemble(self) -> None:
        """
        sets requires_grads = False for all target networks
        """

        for critic in self.target_critic_ensemble:
            critic.requires_grad_(False)
    
    def set_requires_grad_critic_ensemble(self,requires_grad:bool):
        for critic in self.critic_ensemble:
            critic.requires_grad_(requires_grad)

    def zero_grad_ensemble(self) -> None:
        """
        zero grad for all critic optimizers in the ensemble
        """
        for optimizer_critic in self.optimizer_list :
            optimizer_critic.zero_grad(set_to_none=True)
        
    def optimizer_step_ensemble(self) -> None:
        """
        optimizer.step() for all critic optimizers in the ensemble
        """
        for optimizer_critic in self.optimizer_list :
            optimizer_critic.step()

        
    def sync_target_ensemble(self) -> None:
        """
        for all target critics in the target critic ensemble update like
        φtarget = ρ*(φtarget) + (1 - ρ)φcritic
        """

        #for all critics in ensemble
        for target_critic,critic in zip(self.target_critic_ensemble,self.critic_ensemble):
            target_state_dict = target_critic.state_dict()
            state_dict = critic.state_dict()
            #update state_dict using interpolation
            for (name_target,param_target),(name,param) in zip(target_state_dict.items(),state_dict.items()):
                param_target.copy_(self.ρ*param_target+(1-self.ρ)*param)
            
            #update the parameters 
            target_critic.load_state_dict(target_state_dict)

    def min_Q_M(self,M_indices:np.ndarray,next_actions:torch.Tensor,next_states:torch.Tensor) -> torch.Tensor:
        """
        minimum Q of target ensemble models belonging to indices M given next state and next action
        """

        Q = torch.cat([torch.gather(self.target_critic_ensemble[index](next_states),dim=1,index=next_actions.view(-1,1)) for index in M_indices],dim=1)
        #Q size=(batch,num_models) so [[Q_model_1,...,Q_model_n],[Q_model_1,...,Q_model_n]]  where each Q is the action done in that state
        
        min_Q = (torch.min(Q,dim=1).values).view(-1,1)
        #min_Q size= (batch)
        
        return min_Q 
    
    def average_Q(self,state:torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.tensor([critic(state).mean() for critic in self.critic_ensemble]))
        
    
    def chosen_Q(self,state:torch.Tensor,action:torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.tensor([torch.gather(critic(state),dim=1,index=action) for critic in self.critic_ensemble]))

    def policy_update(self,states:torch.Tensor) -> None:
        """
        maximize the expected Q+entropy.
        we do this by multiplying the policy distribution with the Q value distribution 
        (implicitly the Q value distribution contains future discounted entropy as well because that is what the critics are train on)
        and ontop of that we also add entropy of the current distribution as a value to maximize.
        (so if it doesn't hurt the performance maximize the entropy)

        now if we maximize the sum of this multiplication +self.α*entropy 
        we will maximize the expected future perfomance and entropy and trade that off with the current entropy where α is the 
        trade off parameter

        we calculate the average loss over all ensembles critics and the batch. 
        (take the negative of the loss to turn gradient decent into gradient ascent)
        """
        batch_size = states.size(0)

        self.set_requires_grad_critic_ensemble(False)
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss = 0
        for critic in self.critic_ensemble:
            probs, dist = self.policy_net(states)
            Q_values = critic(states) 
            loss -= (torch.sum(torch.sum(probs*Q_values,dim=1,keepdim=True) + self.α*dist.entropy().view(-1,1))) #-= as we want to maximize
        loss /= (batch_size*self.N)
        loss.backward()
        self.policy_optimizer.step()
        self.set_requires_grad_critic_ensemble(True)


    def update(self,batch_size:int) -> None:
        """
        update ensemble and policy network
        """

        for _ in range(self.G):
            self.zero_grad_ensemble()
            
            states, next_states, actions, rewards, dones = self.buffer.sample(batch_size) #batch from experience buffer
            
            M_indices = np.random.choice(np.arange(self.N),size=self.M,replace=False)
            
            _,dist = self.policy_net(next_states)
            next_actions = dist.sample()

            min_Q = self.min_Q_M(M_indices,next_actions,next_states)

            entropy = dist.entropy().detach().view(-1,1)

            y = rewards + self.γ*(min_Q+self.α*entropy) # (Q' + entropy) as this is not log probability but the real entropy of the policy distribution. 
            #And we want to maximize entropy, so we should get into states that increase the entropy of the actions. (so adding is appropriate)

            y[dones] = rewards[dones] #since terminal states should be Q(s,a) = reward instead of Q(s,a) = reward + gamma*(Q(a',s')+alpha*H)

            #update critics
            for critic in self.critic_ensemble:
                Q = torch.gather(critic(states),dim=1,index=actions)
                loss = nn.MSELoss()(Q,y)
                loss.backward()
            self.optimizer_step_ensemble()

            #sync target networks
            self.sync_target_ensemble()

            #update policy network
            self.policy_update(states)
            
    def step(self,batch_size:int,step_num:int) -> None:
        """
        One step of the algorithm so
        one enviroment step and G update steps
        """
        
        #1 enviroment step
        if(self.is_done): 
            #reset enviroment when start
            self.is_done = False 

            state,_ = self.env.reset()
            
            self.current_state = state

            self.writer.add_scalar("entropy_mean",self.entropy_mean/self.env_steps,step_num)

            self.life_num += 1
            self.cummulative_reward = 0
            self.Q_mean = 0
            self.chosen_Q_mean = 0
            self.entropy_mean = 0
            self.env_steps = 1

        with torch.no_grad():
            _,dist = self.policy_net(torch.tensor(np.array(self.current_state),dtype=torch.float,device=self.device).unsqueeze(dim=0))
            
            
        action = dist.sample().item()
        next_state, reward, done, _, _ = self.env.step(action)
        self.buffer.append((self.current_state,next_state,action,reward,done))
        
        self.cummulative_reward += reward
        self.entropy_mean += dist.entropy()

        self.Q_mean = self.average_Q(torch.tensor(np.array(self.current_state),dtype=torch.float,device=self.device).unsqueeze(dim=0))
        self.chosen_Q_mean = self.chosen_Q(torch.tensor(np.array(self.current_state),dtype=torch.float,device=self.device).unsqueeze(dim=0),torch.tensor([[action]],dtype=torch.long,device=self.device))

        
        self.writer.add_scalar("cummulative_rewards/"+str(self.life_num),self.cummulative_reward,step_num)
        self.writer.add_scalar("entropy/"+str(self.life_num),dist.entropy(),step_num)
        self.writer.add_scalar("chosen_Q/"+str(self.life_num),self.Q_mean,step_num)
        self.writer.add_scalar("Q_mean/"+str(self.life_num),self.chosen_Q_mean,step_num)    
        if(done):
            self.is_done = True 

        #G updates
        self.update(batch_size) 
        self.env_steps += 1
        #update state 
        self.current_state = next_state

        if(self.measure_memory_cuda):
            self.cuda_memory_consumption(step_num)
        
        if(self.measure_memory_ram):
            self.memory_consumption(step_num)

        
        self.buffer_consumption(step_num)

    def buffer_consumption(self,step_num:int) -> None:
        used_buffer = len(self.buffer)
        free = self.buffer_size-used_buffer
        self.writer.add_scalar("experience_buffer/used",used_buffer,step_num)
        self.writer.add_scalar("experience_buffer/total",self.buffer_size,step_num)
        self.writer.add_scalar("experience_buffer/free",free,step_num)


    def memory_consumption(self,step_num:int) -> None:
        """
        CONVERSION 
        bytes to GB 
        Bytes / 2^30 
        so bytes / (1<<30)
        """

        total,_,percentage,used,free,_,_,_,_,_,_ = psutil.virtual_memory()
        self.writer.add_scalar("RAM/total",total/(1<<30),step_num)
        self.writer.add_scalar("RAM/percentage_used",percentage,step_num)
        self.writer.add_scalar("RAM/used",used/(1<<30),step_num)
        self.writer.add_scalar("RAM/free",free/(1<<30),step_num)
        

    def cuda_memory_consumption(self,step_num:int) -> None:
        """
        https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
        
        CONVERSION 
        bytes to GB 
        Bytes / 2^30 
        so bytes / (1<<30)
        """
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f_r = r-a  # free inside reserved
        f_t  = t - r  #free total memory
        
        self.writer.add_scalar("cuda_memory/reserved_memory",r/(1<<30),step_num) 
        self.writer.add_scalar("cuda_memory/allocated_memory",a/(1<<30),step_num)
        self.writer.add_scalar("cuda_memory/total_memory",t/(1<<30),step_num)
        self.writer.add_scalar("cuda_memory/free_reserved",f_r/(1<<30),step_num)
        self.writer.add_scalar("cuda_memory/free_total",f_t/(1<<30),step_num)
    
    def inference(self,env:gym.Env,save_to_buffer:bool) -> int:
        """
        when save_to_buffer is true all (s,s',a,r,done) will be saved to the experience buffer 
        """
        state,_ = env.reset()
        cummulative_reward = 0 

        with torch.no_grad():
            
            while(True):
                _,dist = self.policy_net(torch.tensor(np.array(state),dtype=torch.float,device=self.device).unsqueeze(dim=0))
                action = dist.sample().item()
                next_state, reward, done, _, _ = env.step(action)
                if(save_to_buffer):
                    self.buffer.append((state,next_state,action,reward,done))

                state = next_state
                cummulative_reward += reward

                if(done):
                    break 
        return cummulative_reward
    
    def noisy_inference(self,env:gym.Env,epsilon:float=0.3) -> None:

        state,_ = env.reset()
        cummulative_reward = 0 

        with torch.no_grad():
            
            while(True):
                if(np.random.rand() > epsilon):
                    _,dist = self.policy_net(torch.tensor(np.array(state),dtype=torch.float,device=self.device).unsqueeze(dim=0))
                    
                    action = dist.sample().item()
                else:
                    action = int(env.action_space.sample())
                next_state, reward, done, _, _ = env.step(action)
                self.buffer.append((state,next_state,action,reward,done))

                state = next_state
                cummulative_reward += reward

                if(done):
                    break 
            
            return cummulative_reward


def make_checkpoints_folder() -> None:
    if(not os.path.exists("./checkpoints")):
        os.mkdir("./checkpoints")
    else:
        if(not os.path.isdir("./checkpoints")):
            exit("checkpoints is not a folder")
            
"""
https://www.gymlibrary.dev/environments/atari/#id2 #modes and stuff for artari env
"""

def env_space_invaders(render_mode:str=None) -> None:
    env = gym.make("SpaceInvadersDeterministic-v4",obs_type="grayscale",repeat_action_probability=0,full_action_space=False,render_mode=render_mode) 
    env = ResizeObservation(env,(84,84))
    return scaled_rewards(one_life(FrameStack(env, 4)))

if __name__ == "__main__":
    make_checkpoints_folder()

    #hyperparam 
    lr = 0.0001
    policy_body_struct = [256,256,256]
    critic_body_struct = [256,256,256]
    obs_size = 2592
    action_size = 6 #reduced action space
    batch_size = 128
    device = "cuda:0"
    
    iterations = 1e10       

    moving_average_size = 100
    steps_per_append_Moving_Average = 10

    rollout_ever_n_steps = 100
    epsilon = 0.0

    moving_average = deque(maxlen=moving_average_size)
    writer = SummaryWriter()

    

    #init
    torch.backends.cudnn.benchmark = True
    env = env_space_invaders()
    print("action_space: ",env.action_space)
    print("observation_space: ",env.observation_space)
    REDQ = discrete_REDQ(writer=writer,obs_size=obs_size, action_size=action_size, policy_body_struct=policy_body_struct , critic_body_struct= critic_body_struct, env=env, optim_kwargs={"lr":lr},device=device)

    #pre-fill moving average 
    for _ in tqdm(range(moving_average_size),ascii=True,desc="prefill moving average"):
        moving_average.append(REDQ.inference(env_space_invaders(),False))
    
    #train loop
    for step_num in tqdm(range(int(iterations)),ascii=True,desc="train"):
        #train
        REDQ.step(batch_size,step_num)
        
        #update moving avarege
        if(step_num % steps_per_append_Moving_Average == 0):
            moving_average.append(REDQ.inference(env_space_invaders(),False))
            perf = torch.tensor(moving_average,dtype=torch.float)
            writer.add_scalar("moving_average",perf.mean(),step_num)

        #saving file
        if(step_num % 100 == 0):    

            file_name = "checkpoints/" + str(step_num) + ".pt"
            torch.save(REDQ.policy_net.state_dict(),file_name)

        #saving extra trace to buffer 
        if(step_num % rollout_ever_n_steps == 0):
            REDQ.noisy_inference(env_space_invaders(),epsilon)
