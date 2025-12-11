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
    Load expert dataset and rebuild the flattened state to match the
    online env observation (FlattenObservation over dict obs).

    Online env state structure (after FlattenObservation) is:
        [achieved_goal(2), desired_goal(2), observation(4)] -> 8 dims

    Our expert dataset stores:
        "obs"      : observation (4,)  = [x, y, vx, vy]
        "goal"     : desired_goal (2,)
        "next_obs" : next observation (4,)

    We reconstruct:
        achieved_goal      = obs[:2]
        next_achieved_goal = next_obs[:2]
        state      = concat([achieved_goal, goal, obs])
        next_state = concat([next_achieved_goal, goal, next_obs])
    """
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    buffer = ReplayBuffer(state_dim, action_dim, device=device)

    for d in dataset:
        obs      = np.array(d["obs"],      dtype=np.float32)      # (4,)
        goal     = np.array(d["goal"],     dtype=np.float32)      # (2,)
        next_obs = np.array(d["next_obs"], dtype=np.float32)      # (4,)
        action   = np.array(d["action"],   dtype=np.float32)
        reward   = float(d["reward"])
        done     = float(d["done"])

        # Reconstruct achieved goals (current & next)
        achieved      = obs[:2]           # (2,)
        next_achieved = next_obs[:2]      # (2,)

        # Flattened states to match online env
        state      = np.concatenate([achieved, goal, obs], axis=-1)      # (8,)
        next_state = np.concatenate([next_achieved, goal, next_obs], axis=-1)  # (8,)

        # Optional sanity check
        assert state.shape[0] == state_dim, \
            f"State dim mismatch: got {state.shape[0]}, expected {state_dim}"

        buffer.add(state, action, reward, next_state, done)

    print("[TD3-BC Offline] Loaded transitions:", buffer.size)
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

    # Some TD3-BC implementations accept alpha in the constructor;
    # if not, we fall back to the old signature.
    try:
        agent = TD3BCAgent(
            state_dim, action_dim, max_action,
            device=device,
            alpha=alpha
        )
        print("[TD3-BC Offline] Created TD3BCAgent with explicit alpha.")
    except TypeError:
        agent = TD3BCAgent(
            state_dim, action_dim, max_action,
            device=device
        )
        print("[TD3-BC Offline] Created TD3BCAgent WITHOUT alpha argument "
              "(using default inside the agent).")

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

        # TensorBoard logging
        if t % 100 == 0:
            writer.add_scalar("Offline/Critic_Loss", critic_loss, t)
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
