# rl-point-maze
Reinforcement Learning in the Point Maze env using Gymnasium
```bash
【Stage 1】Expert Collection
   A* + PD
   ↓
   (s,a,r,s')

【Stage 2】Offline TD3-BC
   ReplayBuffer (fixed)
   ↓
   TD3 + BC
   ↓
   π_offline

【Stage 3】Online TD3
   π_offline or random
   ↓
   rollout + exploration
   ↓
   ReplayBuffer (growing)
   ↓
   TD3 fine-tune
```

## Build Project
Inside some conda env:
```bash
pip install -e .
```

## Pipeline for evaluating TD3 Random Initialization vs Warm Start Initialization
```bash
python src/scripts/generate_expert.py # set SHOW_VISUALIZATION = False
python src/scripts/train_offline_td3_bc.py --config config/td3_bc.yaml

# online TD3 without warm start
python src/scripts/train_online.py --mode scratch --seed 0
# online TD3 with warm start
python src/scripts/train_online.py \
  --mode warm_start \
  --seed 0 \
  --pretrained_path ./models/td3_bc_offline_step_300000 \
  --dataset_path expert_data_hires.pkl \
  --stats_path models/normalization_stats.pkl
```

## Run Scripts
### Generate expert dataset
```bash
python src/scripts/generate_expert.py
```

### Train offline TD3-BC
We first train a TD3-BC agent purely from the expert dataset:

```bash
python src/scripts/train_offline_td3_bc.py --config config/td3_bc.yaml
```

### Train online TD3 (from scratch)
```bash
python src/scripts/train_online.py
```

### Train online TD3 (warm-start from offline TD3-BC)
fine tune from the offline checkpoint (actor + critic)
```bash
python src/scripts/train_online_warm_start.py
```

### PPO
install SB3
```bash
pip install "stable-baselines3[extra]"
```
then run the script
```bash
python src/scripts/train_ppo.py
```

### Visualize best agent
1. td3 offline and online:
    td3 models are saves as:
    ```bash
    <name>_actor.pth
    <name>_critic.pth
    ```
    run visualization:
    ```bash
    python src/scripts/visualize_agent.py --algo td3 --model_path ./models/<path>
    ```
    e.g.
    ```bash
    python src/scripts/visualize_agent.py --algo td3 --model_path ./models/best_model
    ```
    or 
    ```bash
    python src/scripts/visualize_agent.py --algo td3 --model_path ./models/td3_bc_offline_step_300000
    ```
2. PPO:
   ppo models are saves as:
    ```bash
    <name>.zip
    ```
   run
   ```bash
   python src/scripts/visualize_agent.py --algo ppo --model_path ./models/<name>.zip
   ```
   e.g.
   ```bash
   python src/scripts/visualize_agent.py --algo ppo --model_path ./models/ppo_best_2025-12-09-17-53-33_PPO_0_train.zip
   ``` 

## Overview:
Set the scene: (AGV Navigation)
- task: navigate AGV from A point to B point
- env: walls/obstacles/...
- state: laser/distance/position/orientation
- action: linear and angular velocity

### For the offline TD3-BC run:
The "best model" saving logic was added afterwards, so during my training it did not produce a separate `..._best` file.
For now, the visualization uses the `td3_bc_offline_step_300000` checkpoint.
If you prefer, you can rerun the offline training to generate a true best-model file

### Expert trajectory collection
Pipeline:
A* global & PD local control -> Crashing safety validation -> rollout -> only save the successful trajectory
Data format (input as the offline RL)
```bash
{
  obs,          # 当前连续观测（pos + vel）
  goal,         # desired_goal
  action,       # 连续控制动作 (Fx, Fy)
  reward,       # dense reward + penalty + bonus
  next_obs,     # 下一时刻观测
  done          # 是否成功终止
}
```

#### How to get expert trajectory
layer 1: discrete layer - A_star algorithm is responsible for where to go (global best)
```bash
path_nodes = astar_grid(safe_map, start_node, goal_node)
```
layer 2: continuous layer - PDPathFollower is responsible for how to go
```bash
action = controller.get_action(current_pos, current_vel)
```
layer3: Safety layer - let critic know which is good trajectory
```bash
env = NegativeRewardWrapper
env = WallHitPenaltyWrapper
env = SuccessBonusWrapper
```

#### Why world <-> grid projection
- World is the "physical position"; Grid is the "map cell."
- Rollout means "actually running a full trial."
- Reward isn't for imitation (BC); it's for calculating "how good/bad" an action was later (RL).

```bash
The Misconception: "The environment already knows where the walls are; why do I need my own grid?"
Environment ≠ Planner: A simulator (MuJoCo/Gym) only executes actions and returns observations. It does not calculate paths.
- A requires Discretization:* A* doesn't understand x=0.37,y=−0.12. It understands (Row 12, Col 34). You must map the continuous world to a discrete grid to ask: "Is this coordinate inside a wall?"
- The "Contradiction" Risk: If your projection is inaccurate, the Planner might say a point is safe, while the Controller (in the physical world) sees a collision.
- Key Takeaway: Projection is about Unifying Coordinate Languages so the high-level plan and low-level execution don't conflict.
```

#### how to rollout in an episode
A Rollout is simply "running the process from start to finish."
- Reset the environment.
- Input an Action (from your Expert: A* + PD).
- The Environment moves.
- Repeat until the Goal is reached or time runs out.

Why do this? You need to record what happens under real dynamics (physics/friction/inertia), not just the "perfect" mathematical path.

#### Why need reward in expert collection
later, we need to do offline RL, and use online RL to fine tune the policy
if we don't save reward now:
- critic cannot learn
- Q(s,a) is all blind

Reward specification:
- closer to the goal -> high reward
- crash the wall -> law reward
- arrive -> sparse high reward


### Offline RL: TD3-BC

```bash
expert trajectories (s,a,r,s')
        ↓
【重构 state + 归一化】 
        ↓
ReplayBuffer（静态，永不增长）
        ↓
TD3-BC
  ├── Critic：学 Q(s,a)
  └── Actor ：在“像 expert”的前提下最大化 Q
        ↓
存 best / checkpoint
```

Not only knows how the expert does, but also knows WHICH ACTION IS SAFER & FASTER TO get to-- the destination.
- Critic: in this situation, if AGV drives like this, will the result be good?
In AGV navigation, the critic will learn:

Fast wall-hugging → Low Q (Prone to crashes)

Slower bypassing of obstacles → Higher Q

Straight through narrow passages → Max Q

In conclusion, Critic is not the regulation that human wrote
but the agent learned from expert data and reward

- Actor: 
In a specific state s (e.g., obstacle 0.7m ahead, 0.3m to the left, 1.2m to the right): The Expert data might contain 3 different actions:
Action	Explanation
a1​	Slight deceleration + right turn
a2​	Significant deceleration + small right turn
a3​	Nearly stopping before turning

What does TD3-BC do?

It performs two main steps:
Step 1: BC Regularization (The "Anchor")
"Don't go rogue; stay within the vicinity of a1​,a2​, or a3​."

Step 2: Critic Evaluation
- "a1​: Fast but high risk."
- "a2​: Stable and clears the path."
- "a3​: Safe but too slow."
What the Actor learns: Within the "expert-permitted range," it gravitates toward the action with the highest Q-value (a2​).


#### Comparison with imitation learning (Behavior Cloning)
BC only learns the projection from Input to output of expert
HOWEVER, in reality, AGV can be not accurate, and maybe the laser scan becomes the number that AGV never saw before
In this situation, BC will output an average action, which results into more serious deviation.

#### If the AGV slightly deviates from the expert trajectory?

This is the decisive difference between TD3-BC and BC.

When a deviation occurs (which is very common):
- Scenario: The AGV is 10cm further to the left than the expert was.
- Feedback: The LiDAR readings change.

❌ Behavior of BC (Behavior Cloning)
  Logic: "I haven't seen this specific frame before; let me take a guess."
  Result: 👉 Jittering / Freezing / Crashing

✅ Behavior of TD3-BC

  The Critic says: "Being on the left is more dangerous now. If we steer slightly to the right, the Q-value will be higher."
  The Actor says: "In that case, I will stay close to the expert's action but adjust toward a safer direction."
  Result: 👉 Ability to "Self-Correct"

#### Why is TD3-BC considered Offline RL rather than mere imitation?
```bash
Because:
Question	                                        BC	TD3-BC
Knows if it will hit a wall?	                    ❌    ✅
Distinguishes between 'fast/slow' or 'good/bad'?	❌	   ✅
Can optimize within the expert's range?	          ❌	   ✅
Simply follows the expert blindly?	              ✅	   ❌
```

In a nutshell: BC only knows "what was done," while TD3-BC knows "is doing this worth it?"

#### Why is TD3-BC ideal for AGV Warm-Start?
Because it provides a policy that is:
- Safe (BC Constraint): Doesn't charge ahead recklessly.
- Aware (Critic): Doesn't hug obstacles.
- Stable (TD3 Stability): Doesn't jitter.

During the Online Phase, you can then allow it to:
- Adapt to new obstacles.
- Learn faster paths.
- Fix imperfections in the global planner.

### Online RL: TD3
```bash
        ┌──────────────┐
        │  Environment │  (PointMaze + wrappers)
        └──────┬───────┘
               │  s_t
               ▼
        ┌──────────────┐
        │    TD3       │
        │  Actor π(s)  │───► a_t (+ noise)
        │  Critic Q    │
        └──────┬───────┘
               │
        (s_t, a_t, r_t, s_t+1, done)
               │
               ▼
        ┌──────────────┐
        │ ReplayBuffer │  (不断增长)
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ TD3.train()  │
        └──────────────┘

```
Difference to offline RL and expert collection:
- env layer: train_env, eval_env
Raw PointMaze
  + NegativeRewardWrapper
  + WallHitPenaltyWrapper
  + SuccessBonusWrapper

Reward is calculated instantly, no planner and expert

- Agent layer: (TD3Agent)
```bash
agent = TD3Agent(s_dim, a_dim, max_action)
```
Actor:  π(s) -> a
Critic: Q1(s,a), Q2(s,a)
- no BC Loss
- No expert constraint
- fully believe the data collected by agent itself

- Rollout layer
rollout = policy tested in true env
- actor gives an action
- env runs
- env gives reward

- ReplayBuffer layer 
```bash
buffer.add(state, action, reward, next_state, done)
```
in offline RL: Buffer = recording
here: buffer = memory

- Training layer:
```bash
c_loss, a_loss = agent.train(buffer)
```
Critic:
  Q(s,a) ← r + γ min(Q'(s', π'(s')))

Actor:
  max Q(s, π(s))

- evaluation layer:
```bash
policy (no noise)
   ↓
eval_env rollout
   ↓
统计 reward / success
```
eval only sees, no new data, don't update the network

#### expert_data_hires.pkl
```bash
[
  {
    'obs': np.ndarray,        # 当前观测
    'goal': np.ndarray,       # 目标
    'action': np.ndarray,     # expert 动作
    'reward': float,          # 即时 reward
    'next_obs': np.ndarray,   # 下一步观测
    'done': bool              # 是否成功结束
  },
  {
    ...
  },
  ...
]
```

```bash
with open(path, "rb") as f:
    dataset = pickle.load(f)
```
重构 state
normalize
塞进 ReplayBuffer
👉 这一步之后，.pkl 的使命就结束了
TD3-BC 以后只认 ReplayBuffer

#### normalization_stats.pkl
```bash
{
  "mean": np.ndarray,
  "std": np.ndarray,
  "reward_scale": float
}
告诉后面的 agent：
“你看到的世界，必须和 offline 时一样”

它什么时候用？👉 online warm-start 时
```


```bash
python train_online.py \
  --mode warm_start \
  --pretrained_path td3_bc_offline_best \
  --stats_path normalization_stats.pkl
```
如果你不用它，会发生什么？

offline policy 看 state 是“标准化后的世界”
online 环境给的是“原始世界”
👉 warm-start 直接失效

#### best model
td3_bc_offline_best
best_model
```bash
if critic_loss < best_critic_loss:
    agent.save(best_model_path)

if success_rate >= best_success:
    agent.save("./models/best_model")
```
在训练过程中：

offline：critic 最稳定 / loss 最低
online：success rate 最高

这是“推荐使用”的模型
👉 90% 的情况下，只用这个