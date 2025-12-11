import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

# Import our modular classes
from envs.wrappers import NegativeRewardWrapper, WallHitPenaltyWrapper, SuccessBonusWrapper
from agents.td3_agent import TD3Agent
from utils.replaybuffer import ReplayBuffer
from gymnasium.wrappers import FlattenObservation, RecordVideo
from envs.factory import make_env

# --- Configuration ---
ENV_ID = "PointMaze_Medium-v3"
SEED = 0
MAX_STEPS = 300_000
START_STEPS = 10_000
BATCH_SIZE = 256
EVAL_FREQ = 5_000

# --- 2. Evaluation Logic ---
def evaluate(agent, env, episodes=5):
    avg_reward = 0.
    successes = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(state, noise=0.0) # No noise for eval
            state, reward, done, truncated, info = env.step(action)
            avg_reward += reward
            
            # Check success (BonusWrapper sets this key)
            if info.get('is_success', False):
                successes += 1
                
    return avg_reward / episodes, successes / episodes

# --- 3. Main Loop ---
if __name__ == "__main__":
    if not os.path.exists("./models"): os.makedirs("./models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("./runs/TD3_Clean_Build")

    # Initialize Envs
    train_env = make_env(ENV_ID, SEED, is_eval=False)
    eval_env = make_env(ENV_ID, SEED + 100, is_eval=True, video_folder="./videos")

    # Initialize Agent
    s_dim = train_env.observation_space.shape[0]
    a_dim = train_env.action_space.shape[0]
    max_action = float(train_env.action_space.high[0])

    agent = TD3Agent(s_dim, a_dim, max_action, device)
    buffer = ReplayBuffer(s_dim, a_dim, device=device)

    print(f"Initialized Clean TD3 on {ENV_ID}")

    state, _ = train_env.reset()
    best_success = -1.0

    for t in range(MAX_STEPS):
        # Action Selection
        if t < START_STEPS:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(state, noise=0.2) # Exploration noise

        # Step
        next_state, reward, terminated, truncated, info = train_env.step(action)
        
        # Buffer Storage Logic
        done_bool = terminated
        buffer.add(state, action, reward, next_state, done_bool)
        
        state = next_state
        if terminated or truncated:
            state, _ = train_env.reset()

        # Training
        if t >= START_STEPS:
            c_loss, a_loss = agent.train(buffer, BATCH_SIZE)
            
            # Logging
            if t % 100 == 0:
                writer.add_scalar("Train/Critic_Loss", c_loss, t)
                if a_loss: writer.add_scalar("Train/Actor_Loss", a_loss, t)

        # Evaluation
        if (t + 1) % EVAL_FREQ == 0:
            avg_rew, success_rate = evaluate(agent, eval_env)
            writer.add_scalar("Eval/Reward", avg_rew, t)
            writer.add_scalar("Eval/Success_Rate", success_rate, t)
            print(f"Step {t+1}: Success Rate: {success_rate:.2f} | Avg Reward: {avg_rew:.2f}")

            if success_rate >= best_success:
                best_success = success_rate
                agent.save(f"./models/best_model")
                print("--> New Best Model Saved")

    train_env.close()
    eval_env.close()
    writer.close()