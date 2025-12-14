import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/scripts
SRC_DIR = os.path.dirname(FILE_DIR)                        # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import pickle
import argparse
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.td3_bc_agent import TD3BCAgent
from utils.replaybuffer import ReplayBuffer
from envs.factory import make_env
import gymnasium as gym  # noqa: F401  (ensure gymnasium is imported)
import gymnasium_robotics  # noqa: F401  (ensure robotics envs are registered)


def load_config(path: str) -> dict:
    """
    Load a YAML config file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A dictionary with configuration parameters.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(device_cfg: str) -> torch.device:
    """
    Decide which device to use based on config.

    Args:
        device_cfg: "auto", "cuda", or "cpu"

    Returns:
        A torch.device object.
    """
    device_cfg = device_cfg.lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_cfg == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_dataset_to_buffer(path, state_dim, action_dim, device):
    """
    Loads expert dataset, reconstructs flattened states, and performs
    Standard Normalization (Z-Score) on states and Scaling on rewards.
    """
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    buffer = ReplayBuffer(state_dim, action_dim, device=device)
    
    # 1. Collect raw data into lists
    print(f"[Data] Loading raw data from {path}...")
    all_states, all_next_states, all_actions, all_rewards, all_dones = [], [], [], [], []

    for d in dataset:
        obs      = np.array(d["obs"], dtype=np.float32)
        goal     = np.array(d["goal"], dtype=np.float32)
        next_obs = np.array(d["next_obs"], dtype=np.float32)
        
        # Reconstruct Flattened State: [Achieved(2), Goal(2), Obs(4)]
        state      = np.concatenate([obs[:2], goal, obs], axis=-1)
        next_state = np.concatenate([next_obs[:2], goal, next_obs], axis=-1)
        
        all_states.append(state)
        all_next_states.append(next_state)
        all_actions.append(np.array(d["action"], dtype=np.float32))
        all_rewards.append(float(d["reward"]))
        all_dones.append(float(d["done"]))

    # Convert to Numpy for vector math
    np_states = np.array(all_states)
    np_next_states = np.array(all_next_states)
    np_rewards = np.array(all_rewards)
    
    # --- 2. STATE NORMALIZATION (Standard Scaler) ---
    # Compute stats on the full dataset
    state_mean = np_states.mean(axis=0)
    state_std  = np_states.std(axis=0) + 1e-3  # +1e-3 prevents division by zero
    
    # Apply (x - mu) / sigma
    np_states = (np_states - state_mean) / state_std
    np_next_states = (np_next_states - state_mean) / state_std

    print(f"[Data] Normalized States. Mean ~0, Std ~1.")

    # --- 3. REWARD SCALING ---
    # PointMaze dense rewards are often large negative distances (e.g., -5.0 to 0.0)
    # Neural networks prefer rewards roughly in [-1, 1] or [0, 1].
    
    # Heuristic: If rewards are large (absolute value > 10), scale them down.
    # If using sparse rewards (0/1), this block does nothing.
    r_min, r_max = np_rewards.min(), np_rewards.max()
    scale_factor = 1.0
    
    print(f"r_min: {r_min}, r_max: {r_max}")

    # Only scale if the range is weird (e.g. 0 to -100)
    if np.abs(r_min) > 10.0 or np.abs(r_max) > 10.0:
        scale_factor = 1.0 / max(np.abs(r_min), np.abs(r_max))
        np_rewards = np_rewards * scale_factor
        print(f"[Data] Scaled Rewards by {scale_factor:.4f}. New Range: [{np_rewards.min():.2f}, {np_rewards.max():.2f}]")

    # --- 4. Fill Buffer ---
    for i in range(len(np_states)):
        buffer.add(
            np_states[i], 
            all_actions[i], 
            np_rewards[i], 
            np_next_states[i], 
            all_dones[i]
        )

    print(f"[TD3-BC Offline] Buffer filled with {buffer.size} transitions.")
    
    # --- CRITICAL: Save Stats for Online Usage ---
    # You MUST save these so your online agent knows how to view the world!
    stats = {
        "mean": state_mean,
        "std": state_std,
        "reward_scale": scale_factor
    }
    with open("normalization_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    print("[Data] Saved normalization_stats.pkl (KEEP THIS FILE)")

    return buffer



def main():
    # -----------------------
    # 1. Parse command line
    # -----------------------
    parser = argparse.ArgumentParser(
        description="Offline TD3-BC training on an expert dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for offline TD3-BC."
    )
    args = parser.parse_args()

    # -----------------------
    # 2. Load configuration
    # -----------------------
    cfg = load_config(args.config)

    # Basic config with defaults
    dataset_path = cfg.get("dataset", "expert_data_hires.pkl")
    max_updates = int(cfg.get("max_updates", 300_000))
    batch_size = int(cfg.get("batch_size", 256))
    alpha = float(cfg.get("alpha", 2.5))  # TD3-BC regularization weight

    env_id = cfg.get("env_id", "PointMaze_Medium-v3")
    seed = int(cfg.get("seed", 0))

    models_dir = cfg.get("models_dir", "./models")
    log_dir = cfg.get("log_dir", "./runs/TD3_BC_Offline")
    save_freq = int(cfg.get("save_freq", 10000))

    device_cfg = cfg.get("device", "auto")  # "auto", "cuda", or "cpu"
    device = get_device(device_cfg)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("======================================")
    print("[TD3-BC Offline] Configuration summary")
    print(f" Dataset path : {dataset_path}")
    print(f" Env ID       : {env_id}")
    print(f" Seed         : {seed}")
    print(f" Max updates  : {max_updates}")
    print(f" Batch size   : {batch_size}")
    print(f" Alpha        : {alpha}")
    print(f" Models dir   : {models_dir}")
    print(f" Log dir      : {log_dir}")
    print(f" Save freq    : {save_freq}")
    print(f" Device       : {device}")
    print("======================================")

    # -----------------------
    # 3. Set random seeds
    # -----------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------------------
    # 4. Use env only to get dimensions
    # ---------------------------------
    tmp_env = make_env(env_id, seed=seed, is_eval=False)
    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]
    max_action = float(tmp_env.action_space.high[0])
    tmp_env.close()

    print(f"[TD3-BC Offline] State dim  : {state_dim}")
    print(f"[TD3-BC Offline] Action dim : {action_dim}")
    print(f"[TD3-BC Offline] Max action : {max_action}")

    # -----------------------
    # 5. Build buffer & agent
    # -----------------------
    buffer = load_dataset_to_buffer(dataset_path, state_dim, action_dim, device)

    agent = TD3BCAgent(
            state_dim, 
            action_dim, 
            max_action,
            device=device,
            bc_coef=alpha,
            lr=3e-4 
        )

    writer = SummaryWriter(log_dir)

    # -----------------------
    # 6. Training loop
    # -----------------------
    print("[TD3-BC Offline] Start training...")

    # Track the best model (by smallest critic loss)
    best_critic_loss = float("inf")
    best_model_path = os.path.join(models_dir, "td3_bc_offline_best")

    for t in range(max_updates):
        critic_loss, actor_loss = agent.train(buffer, batch_size)

        # FIX: Use (t + 1) to align with total_it counters (100, 200, etc.)
        if (t + 1) % 100 == 0:
            writer.add_scalar("Offline/Critic_Loss", critic_loss, t)
            
            # Now this will successfully log, because at t=99, total_it=100 (Even)
            if actor_loss is not None:
                writer.add_scalar("Offline/Actor_Loss", actor_loss, t)

        if critic_loss < best_critic_loss:
            best_critic_loss = critic_loss
            agent.save(best_model_path)
            print(f"[TD3-BC Offline] New BEST model at step {t+1}, "
                f"critic {critic_loss:.3f}, saved to {best_model_path}")
            
        # Periodic model saving
        if (t + 1) % save_freq == 0:
            save_path = os.path.join(models_dir, f"td3_bc_offline_step_{t+1}")
            agent.save(save_path)
            print(f"[TD3-BC Offline] step {t+1}, critic {critic_loss:.3f}, "
                  f"actor {actor_loss}, saved to {save_path}")

    print("[TD3-BC Offline] Training finished.")
    writer.close()


if __name__ == "__main__":
    main()
