import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, RecordVideo, RecordEpisodeStatistics
import pickle

class NegativeRewardWrapper(gym.RewardWrapper):
    """
    Shifts reward to be negative log-distance.
    Prevents 'hovering' by penalizing existence.
    """
    def reward(self, reward):
        return np.log(max(reward, 1e-5))

class WallHitPenaltyWrapper(gym.Wrapper):
    """
    Penalizes the agent for pushing against walls (High Action + Low Velocity).
    """
    def __init__(self, env, penalty=0.5, vel_thresh=0.01, act_thresh=0.5):
        super().__init__(env)
        self.penalty = penalty
        self.vel_thresh = vel_thresh
        self.act_thresh = act_thresh

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Handle Dictionary vs Flattened observation
        if isinstance(obs, dict):
            velocities = obs['observation'][2:4]
        else:
            velocities = obs[2:4] # Assumes Flattened PointMaze structure

        speed = np.linalg.norm(velocities)
        effort = np.linalg.norm(action)

        if effort > self.act_thresh and speed < self.vel_thresh:
            reward -= self.penalty
            info['wall_hit'] = True
        
        return obs, reward, term, trunc, info

class SuccessBonusWrapper(gym.Wrapper):
    """
    Checks for the specific 'success' condition and overrides reward.
    """
    def __init__(self, env, dist_threshold=-0.5, bonus=1.0):
        super().__init__(env)
        self.dist_threshold = dist_threshold
        self.bonus = bonus

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # 'reward' here is already log-distance from NegativeRewardWrapper
        if reward > self.dist_threshold:
            reward = self.bonus
            term = True # Force termination
            info['is_success'] = True
        
        return obs, reward, term, trunc, info
    

class OfflineStatsWrapper(gym.Wrapper):
    """
    Normalizes observations and scales rewards using fixed statistics 
    computed during offline training.
    """
    def __init__(self, env, stats_path):
        super().__init__(env)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            
        self.mean = stats["mean"]
        self.std = stats["std"]
        self.reward_scale = stats.get("reward_scale", 1.0)
        
        print(f"--> Loaded Offline Stats from {stats_path}")
        print(f"    Reward Scale: {self.reward_scale}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Normalize Observation
        # (obs - mean) / std
        obs = (obs - self.mean) / self.std
        
        # 2. Scale Reward
        reward = reward * self.reward_scale
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = (obs - self.mean) / self.std
        return obs, info