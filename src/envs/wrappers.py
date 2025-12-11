import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, RecordVideo, RecordEpisodeStatistics

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