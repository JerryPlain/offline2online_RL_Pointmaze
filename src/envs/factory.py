from .wrappers import NegativeRewardWrapper, WallHitPenaltyWrapper, SuccessBonusWrapper
from gymnasium.wrappers import FlattenObservation, RecordVideo
import gymnasium as gym
import gymnasium_robotics 

def make_env(env_id, seed, is_eval=False, video_folder=None, render_mode="rgb_array"):
    env = gym.make(env_id, render_mode=render_mode, reward_type='dense')
    
    if is_eval and video_folder and render_mode != "human":
        env = RecordVideo(env, video_folder=video_folder, 
                          episode_trigger=lambda x: True, disable_logger=True)
    
    env = NegativeRewardWrapper(env)
    env = WallHitPenaltyWrapper(env, penalty=0.5, act_thresh=0.5)
    env = SuccessBonusWrapper(env, dist_threshold=-0.15, bonus=1.0)
    env = FlattenObservation(env)
    
    env.reset(seed=seed)
    return env