import time
import torch
import numpy as np
import os, sys
import argparse
FILE_DIR = os.path.dirname(os.path.abspath(__file__))     # src/scripts
SRC_DIR  = os.path.dirname(FILE_DIR)                      # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                       # project root

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import from your modular project structure
from envs.factory import make_env
from agents.td3_agent import TD3Agent
from agents.td3_bc_agent import TD3BCAgent
from stable_baselines3 import PPO as SB3PPO

def visualize(model_path, env_id="PointMaze_Medium-v3", seed=5, episodes=5):
    # 1. Setup Environment (Human Render Mode for Window)
    print(f"Launching {env_id} visualization...")
    env = make_env(env_id, seed=seed, is_eval=True, render_mode="human")

    # 2. Initialize Agent
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cpu") # CPU is fine for inference/watching

    if "bc" in model_path.lower():
        print("Using TD3BCAgent for offline training.")
        agent = TD3BCAgent(s_dim, a_dim, max_action, device)
    else:
        print("Using TD3Agent for online training.")
        agent = TD3Agent(s_dim, a_dim, max_action, device=device)

    # 3. Load Weights
    if os.path.exists(model_path + "_actor.pth"):
        print(f"Loading weights from: {model_path}")
        agent.load(model_path)
    else:
        print(f"ERROR: Model not found at {model_path}_actor.pth")
        return

    # 4. Live Loop
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"\n--- Episode {ep + 1} Start ---")
        
        while not (done or truncated):
            # Select action (Deterministic / No Noise)
            action = agent.select_action(state, noise=0.0)
            
            # Step
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # --- Live Metrics Calculation ---
            # Logic: We know Reward = log(distance). So Distance = exp(reward).
            # Exception: Success Bonus is 10.0
            if reward == 10.0:
                dist_str = "SUCCESS (0.00)"
                status_icon = "🟢"
            else:
                # Reverse the log wrapper to get actual distance
                real_dist = np.exp(reward)
                dist_str = f"{real_dist:.4f}"
                status_icon = "running"

            # Print inplace (overwrite line) for clean terminal
            print(f"\rStep: {step:03d} | Reward: {reward: .4f} | Dist: {dist_str} | {status_icon}", end="")
            
            # Slow down slightly so you can watch comfortably (20-30 FPS)
            time.sleep(0.03)

        # Final Status
        end_status = "GOAL REACHED!" if info.get('is_success') else "FAILED (Time Limit)"
        print(f"\nResult: {end_status} | Total Reward: {total_reward:.2f}")
        time.sleep(1.0) # Pause between episodes

    env.close()

def visualize_ppo(model_path, env_id="PointMaze_Medium-v3", seed=5, episodes=5, render=True):
    """Visualize an SB3 PPO policy saved as a .zip file."""
    print(f"Launching {env_id} PPO visualization...")
    env = make_env(env_id, seed=seed, is_eval=True, render_mode="human")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found at: {model_path}")

    print(f"Loading PPO model from: {model_path}")
    model = SB3PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0.0

        print(f"\n--- PPO Episode {ep + 1} ---")
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()

            print(f"\rStep: {step:03d} | Reward: {reward: .4f}", end="")
            time.sleep(0.03)

        end_status = "GOAL REACHED!" if info.get("is_success") else "FAILED (Time Limit)"
        print(f"\nResult: {end_status} | Total Reward: {total_reward:.2f}")
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["td3", "td3bc", "ppo"],
                        help="Which algorithm to visualize")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model. "
                             "TD3/TD3BC: base path without _actor.pth; "
                             "PPO: full .zip path")
    parser.add_argument("--env_id", type=str, default="PointMaze_Medium-v3")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no_render", action="store_true",
                        help="Disable env.render() for PPO")

    args = parser.parse_args()

    if args.algo == "ppo":
        # PPO
        visualize_ppo(
            model_path=args.model_path,
            env_id=args.env_id,
            seed=args.seed,
            episodes=args.episodes,
            render=not args.no_render,
        )
    else:
        # TD3 / TD3BC
        visualize(
            model_path=args.model_path,
            env_id=args.env_id,
            seed=args.seed,
            episodes=args.episodes,
        )
