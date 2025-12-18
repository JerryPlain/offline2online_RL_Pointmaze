
# OnlineRL combines with HPRS (autoshaping)
```bash
┌────────────────────────────────────────┐
│            RL Algorithm                │
│        (PPO / TD3 / SAC)               │
│   ─────────────────────────────────   │
│   sees: obs, shaped_reward, done       │
└────────────────▲───────────────────────┘
                 │
┌────────────────┴───────────────────────┐
│          VecEnvWrapper Layer            │
│                                        │
│  HPRSVecWrapper   ←【你做的核心创新】  │
│   - ignores base reward                │
│   - computes potential-based shaping   │
│                                        │
│  VecMonitor                            │
│   - logging / success stats            │
└────────────────▲───────────────────────┘
                 │
┌────────────────┴───────────────────────┐
│             VecEnv                     │
│     (SimulatorVecEnv / DummyVecEnv)    │
│   - step_async / step_wait             │
└────────────────▲───────────────────────┘
                 │
┌────────────────┴───────────────────────┐
│         Gymnasium Env (Unity)           │
│        WarehouseUnityEnv / AGV          │
│   - physics, collisions, sensors       │
│   - NO reward logic (or trivial)       │
└────────────────────────────────────────┘
```
NOW:
```bash
PPO
 ↑
HPRSVecWrapper
 ↑
SimulatorVecEnv
 ↑
WarehouseUnityEnv
```

Later:
```bash
TD3 (SB3 or custom)
 ↑
HPRSVecWrapper   ← 完全不动
 ↑
SimulatorVecEnv
 ↑
WarehouseUnityEnv
```

```bash
            Expert / Planner
                    ↓
          Offline TD3-BC (no HPRS)
                    ↓
             π_offline (safe init)
                    ↓
        ┌────────────────────────┐
        │   Online RL Algorithm  │
        │     (TD3 / PPO)        │
        └────────────▲───────────┘
                     │
        ┌────────────┴───────────┐
        │     HPRSVecWrapper      │
        │  (task logic layer)     │
        └────────────▲───────────┘
                     │
              SimulatorVecEnv
                     │
            WarehouseUnityEnv
```
```bash
| 方法                  | Offline init | HPRS | Online algo |
| ------------------- | ------------ | ---- | ----------- |
| TD3-scratch         | ❌            | ❌    | TD3         |
| TD3-warm            | ✅            | ❌    | TD3         |
| TD3 + HPRS          | ❌            | ✅    | TD3         |
| **TD3-warm + HPRS** | ✅            | ✅    | TD3         |
```