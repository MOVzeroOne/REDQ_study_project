import gymnasium as gym

"""
code adapted from gymnasium.wrappers.AtariPreprocessing

"""


class one_life(gym.Wrapper):
    def __init__(self,env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.game_over = False
    
    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale
    
    def reset(self,**kwargs):
        obs, reset_info = self.env.reset(**kwargs)
        self.lives = self.ale.lives()
        self.game_over = False 
        return obs, reset_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) 
        
        new_lives = self.ale.lives()
        terminated = terminated or new_lives < self.lives
        self.game_over = terminated

        return obs, reward, terminated, truncated, info
