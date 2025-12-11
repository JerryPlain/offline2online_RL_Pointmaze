# rl-point-maze
Reinforcement Learning in the Point Maze env using Gymnasium

## Build Project
Inside some conda env:
```bash
pip install -e .
```

## Run Scripts
### Generate expert dataset
```bash
python src/scripts/generate_expert.py
```

### Train online TD3
```bash
python src/scripts/train_online.py
```

### Train offline TD3-BC

We first train a TD3-BC agent purely from the expert dataset:

```bash
python src/scripts/train_offline_td3_bc.py --config config/td3_bc.yaml
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


### PPO
install SB3
```bash
pip install "stable-baselines3[extra]"
```
then run the script
```bash
python src/scripts/train_ppo.py
```


## Note for our team
1. For the offline TD3-BC run:
The "best model" saving logic was added afterwards, so during my training it did not produce a separate `..._best` file.
For now, the visualization uses the `td3_bc_offline_step_300000` checkpoint.
If you prefer, you can rerun the offline training to generate a true best-model file 😊
