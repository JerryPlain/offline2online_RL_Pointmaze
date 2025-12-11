import os,sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/scripts
SRC_DIR = os.path.dirname(FILE_DIR)                        # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

from envs.factory import make_env   

ENV_ID = "PointMaze_Medium-v3"
SEED = 0


# ================================
# PPO Evaluation
# ================================
def eval_policy_ppo(model, env, eval_episodes=5, render=False, deterministic=True):
    """
    Simple PPO evaluation.
    Returns average reward and success rate (if `info["is_success"]` exists).
    """
    start_time = time.time()
    episode_rewards = []
    successes = 0

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        ep_rew = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward
            if render:
                env.render()

        episode_rewards.append(ep_rew)
        if info.get("is_success", False):
            successes += 1

    avg_reward = float(np.mean(episode_rewards))
    success_rate = successes / eval_episodes
    duration = time.time() - start_time

    print(f"[PPO Eval] AvgReward={avg_reward:.2f}, SuccessRate={success_rate:.2f}, Duration={duration:.1f}s")
    return avg_reward, success_rate, duration


# ================================
# Main PPO Training Loop
# ================================
def run(ppo_train_dict, seed, env, log_dir="./runs/PPO_PointMaze"):
    file_name = f"PPO_{seed}_train"
    print("---------------------------------------")
    print(f"Training PPO on {ENV_ID}, Seed: {seed}")
    print("---------------------------------------")

    os.makedirs("./models", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    current_datetime = datetime.now()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------- PPO Hyperparameters ----------
    ppo_hyperparams = {
        "policy": "MlpPolicy",
        "n_steps": ppo_train_dict.get("n_steps", 1024),
        "batch_size": ppo_train_dict.get("batch_size", 256),
        "n_epochs": ppo_train_dict.get("n_epochs", 10),
        "gamma": ppo_train_dict.get("gamma", 0.99),
        "gae_lambda": ppo_train_dict.get("gae_lambda", 0.95),
        "clip_range": ppo_train_dict.get("clip_range", 0.2),
        "normalize_advantage": ppo_train_dict.get("normalize_advantage", True),
        "ent_coef": ppo_train_dict.get("ent_coef", 0.0),
        "vf_coef": ppo_train_dict.get("vf_coef", 0.5),
        "max_grad_norm": ppo_train_dict.get("max_grad_norm", 0.5),
        "learning_rate": ppo_train_dict.get("learning_rate", 3e-4),
        "seed": seed,
        "verbose": 1,
    }

    # Activation function and network structure
    activation_fn = ppo_train_dict.get("activation_fn", "tanh")
    if isinstance(activation_fn, str):
        activation_fn = nn.Tanh if activation_fn.lower() == "tanh" else nn.ReLU

    policy_kwargs = {
        "net_arch": ppo_train_dict.get("net_arch", [dict(pi=[64, 64], vf=[64, 64])]),
        "activation_fn": activation_fn,
    }
    ppo_hyperparams["policy_kwargs"] = policy_kwargs

    # ---------- Load pretrained weights (optional) ----------
    load_name = ppo_train_dict.get("load_model", "").strip()

    if load_name == "":
        print("[PPO] No checkpoint specified → training from scratch.")
        model = PPO(env=env, tensorboard_log="./tensorboard_logs", **ppo_hyperparams)
    else:
        model_path = os.path.join("./models", load_name)
        print(f"[PPO] Loading checkpoint: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        model = PPO.load(model_path, env=env)

    # ---------- Training settings ----------
    total_timesteps = ppo_train_dict.get("total_timesteps", 300_000)
    eval_freq = ppo_train_dict.get("eval_freq", 30_000)
    eval_episodes = ppo_train_dict.get("eval_episodes", 5)
    save_freq = ppo_train_dict.get("save_freq", 100_000)
    render = ppo_train_dict.get("render", False)

    best_eval_reward = -np.inf
    timesteps_trained = 0

    print(f"[PPO] Training for {total_timesteps} timesteps | Eval every {eval_freq} steps")

    # ================================
    # Training Loop
    # ================================
    while timesteps_trained < total_timesteps:

        timesteps_to_train = min(eval_freq, total_timesteps - timesteps_trained)
        print(f"\n[PPO] Training {timesteps_trained} → {timesteps_trained + timesteps_to_train}")

        model.learn(
            total_timesteps=timesteps_to_train,
            reset_num_timesteps=False,  # Do not reset; allows incremental training
            progress_bar=True,
        )
        timesteps_trained += timesteps_to_train

        # ---------- Evaluation ----------
        avg_rew, success_rate, eval_dur = eval_policy_ppo(
            model,
            env,
            eval_episodes=eval_episodes,
            render=render,
            deterministic=ppo_train_dict.get("deterministic_eval", True),
        )

        writer.add_scalar("Eval/AvgReward", avg_rew, timesteps_trained)
        writer.add_scalar("Eval/SuccessRate", success_rate, timesteps_trained)

        # ---------- Save latest model ----------
        model.save(f"./models/{file_name}")

        # ---------- Save best model ----------
        if avg_rew > best_eval_reward:
            best_eval_reward = avg_rew
            best_model_name = f"ppo_best_{current_datetime.strftime('%Y-%m-%d-%H-%M-%S')}_{file_name}"
            print(f"[PPO] New BEST model saved → {best_model_name} (avg_reward={avg_rew:.2f})")
            model.save(f"./models/{best_model_name}")

        # ---------- Periodic checkpoint ----------
        if timesteps_trained % save_freq == 0:
            checkpoint_name = f"{file_name}_step_{timesteps_trained}"
            print(f"[PPO] Saving checkpoint: {checkpoint_name}")
            model.save(f"./models/{checkpoint_name}")

    # ---------- Final save ----------
    final_name = f"final_{file_name}"
    print(f"\n[PPO] Training finished. Saving final model → {final_name}")
    model.save(f"./models/{final_name}")

    writer.close()
    return model


# ================================
# Default PPO Config
# ================================
def get_default_ppo_config():
    return dict(
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        total_timesteps=300_000,
        eval_freq=10_000,
        eval_episodes=5,
        save_freq=50_000,
        deterministic_eval=True,
        load_model="",  # "" → train from scratch; otherwise load model name
        render=False,
    )


if __name__ == "__main__":
    ppo_cfg = get_default_ppo_config()
    env = make_env(ENV_ID, SEED, is_eval=False)
    run(ppo_cfg, SEED, env)
    env.close()
