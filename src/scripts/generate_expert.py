import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
import heapq
import matplotlib.pyplot as plt

from utils.pd_path_follower import PDPathFollower
from envs.wrappers import NegativeRewardWrapper, WallHitPenaltyWrapper, SuccessBonusWrapper

# --- CONFIGURATION ---
ENV_ID = "PointMaze_Medium-v3"
TOTAL_EPISODES = 1000        # Set this to how many episodes you want to record
OUTPUT_FILE = "expert_data_hires.pkl"
SHOW_VISUALIZATION = False # Set True to watch the agent live
RESOLUTION_SCALE = 10     # High resolution for smooth paths

#MANUAL_OFFSET_X = -0.5 # Try -1.2 first (shifts grid Center Left)
#MANUAL_OFFSET_Y = -0.4  # Slight adjustment Up might be needed too
MANUAL_OFFSET_X = 0
MANUAL_OFFSET_Y = 0

# --- 1. COORDINATE MAPPING ---
def world_to_grid(x, y, shape):
    rows, cols = shape
    # Use float division / 2.0 to handle both even (4x4) and odd (5x5) maps correctly
    # If 5x5: 5/2 = 2.5 (Center of middle block)
    # If 4x4: 4/2 = 2.0 (Line between blocks)
    off_x = (cols / 2.0) + MANUAL_OFFSET_X
    off_y = (rows / 2.0) + MANUAL_OFFSET_Y
    
    # Scale coordinates
    # Logic: 
    #   World X+ -> Grid Col+
    #   World Y+ -> Grid Row- (Inverted for image coords)
    col_float = (off_x + x) * RESOLUTION_SCALE
    row_float = (off_y - y) * RESOLUTION_SCALE
    
    # Use floor to strictly bin coordinates
    col = int(np.floor(col_float))
    row = int(np.floor(row_float))
    
    # Clamp to bounds
    max_r = (rows * RESOLUTION_SCALE) - 1
    max_c = (cols * RESOLUTION_SCALE) - 1
    return max(0, min(row, max_r)), max(0, min(col, max_c))

# --- ADD THIS HELPER ---
def world_to_grid_float(x, y, shape):
    """
    Same as world_to_grid, but returns EXACT float coordinates (row, col)
    instead of rounding to integers. This reveals sub-pixel misalignment.
    """
    rows, cols = shape 
    off_x = (cols // 2) + 0.5+ MANUAL_OFFSET_X
    off_y = (rows // 2) + 0.5+ MANUAL_OFFSET_Y
    
    scaled_x = x * RESOLUTION_SCALE
    scaled_y = y * RESOLUTION_SCALE
    
    # Precise float calculation
    row = (off_y * RESOLUTION_SCALE) - scaled_y - 0.5
    col = (off_x * RESOLUTION_SCALE) + scaled_x - 0.5
    
    return row, col

# --- UPDATED DEBUG VISUALIZER ---
def visualize_debug(env, hires_map, safe_map, start_node, goal_node, path_nodes, start_xy):
    real_img = env.render()
    
    # Calculate the raw floating-point grid position
    raw_r, raw_c = world_to_grid_float(start_xy[0], start_xy[1], (5, 5)) # 5x5 is original shape for Medium? 
    # NOTE: Pass 'original_shape' variable here in main loop, likely (rows, cols) of raw_map
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Reality
    axes[0].imshow(real_img)
    axes[0].set_title("Reality (MuJoCo)")
    axes[0].axis('off')
    
    # Right: The Planner's Brain
    axes[1].imshow(safe_map, cmap='Greys', origin='upper')
    
    # Plot Path
    if path_nodes:
        rs = [p[0] for p in path_nodes]
        cs = [p[1] for p in path_nodes]
        axes[1].plot(cs, rs, c='blue', linewidth=2, alpha=0.5, label="A* Path")
        
    # 1. Plot the Snapped Integer Node (What A* uses)
    axes[1].scatter(start_node[1], start_node[0], c='green', s=150, label="Snapped Node (Integer)")
    
    # 2. Plot the TRUE Continuous Position (Where physics puts it)
    axes[1].scatter(raw_c, raw_r, c='gold', marker='x', s=150, linewidth=3, label="True Raw Pos (Float)")
    
    axes[1].scatter(goal_node[1], goal_node[0], c='red', s=150, marker='*', label="Goal")
    
    axes[1].set_title(f"Alignment Check\nYellow X inside Wall = BAD OFFSET")
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


def grid_to_world(r, c, shape):
    rows, cols = shape
    # 1. FIX: Added MANUAL_OFFSET here to match world_to_grid
    # 2. FIX: Used / 2.0 logic for consistency
    off_x = (cols / 2.0) + MANUAL_OFFSET_X
    off_y = (rows / 2.0) + MANUAL_OFFSET_Y
    
    # We add 0.5 to return the CENTER of the grid cell
    # Inverse algebra of world_to_grid
    x = (c + 0.5) / RESOLUTION_SCALE - off_x
    y = off_y - (r + 0.5) / RESOLUTION_SCALE
    
    return x, y

def get_closest_valid_node(node, maze_map):
    rows, cols = maze_map.shape
    r, c = node
    if maze_map[r, c] == 0: return node
    
    # Spiral search for nearest empty cell
    for radius in range(1, 5): 
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if maze_map[nr, nc] == 0:
                        return (nr, nc)
    return node

def inflate_maze(original_map, scale):
    return np.kron(original_map, np.ones((scale, scale)))

def inflate_walls(maze_map, radius=1):
    """
    Adds a safety layer around walls.
    radius: How many extra layers of pixels to mark as occupied.
    """
    rows, cols = maze_map.shape
    new_map = maze_map.copy()
    
    # Find all current wall pixels
    wall_indices = np.argwhere(maze_map == 1)
    
    for r, c in wall_indices:
        # Mark neighbors as occupied
        r_min = max(0, r - radius)
        r_max = min(rows, r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(cols, c + radius + 1)
        
        new_map[r_min:r_max, c_min:c_max] = 1
        
    return new_map

# --- 2. A* ALGORITHM ---
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar_grid(maze_map, start, goal):
    rows, cols = maze_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze_map[neighbor] == 1: continue
                dist = np.sqrt(dr**2 + dc**2)
                tentative_g_score = g_score[current] + dist
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))
    return None

def pd_controller(current_pos, target_pos, current_vel, is_last_point=False):
    # Aggressive Gains
    kp = 15.0  # Increased from 8.0 to push harder
    kd = 4.0   # Damping to prevent oscillating
    
    error = target_pos - current_pos
    
    # Standard PD
    action = (error * kp) - (current_vel * kd)
    
    # SPEED HACK: If we are NOT at the final goal, clamp the braking force.
    # If the velocity is pointing towards the target, don't let KD (damping) slow us down too much.
    if not is_last_point:
        # Dot product checks alignment. If velocity aligns with error, we are good.
        alignment = np.dot(error, current_vel)
        if alignment > 0:
            # We are moving in the right direction, reduce damping effect to keep speed up
            action += current_vel * 2.0 

    return np.clip(action, -1.0, 1.0)


def check_collision(robot_pos, maze_map, original_shape):
    """
    Returns True if robot_pos (world coords) hits a wall.
    """
    r, c = world_to_grid(robot_pos[0], robot_pos[1], original_shape)
    # Check bounds
    if r < 0 or r >= maze_map.shape[0] or c < 0 or c >= maze_map.shape[1]:
        return True
    # Check occupancy (1 = wall)
    return maze_map[r, c] == 1

def simulate_trajectory(current_pos, current_vel, action, dt=0.1, steps=5):
    """
    Predicts where the robot will go if we apply 'action' for 'steps'
    Simple physics integration (PointMass)
    """
    path = []
    pos = current_pos.copy()
    vel = current_vel.copy()
    
    # PointMaze physics approximation (friction included usually, simplified here)
    friction = 0.95 
    
    for _ in range(steps):
        vel = (vel * friction) + action  # Apply force
        pos = pos + (vel * dt)           # Update position
        path.append(pos.copy())
        
    return path


def dwa_controller(current_pos, current_vel, target_pos, maze_map, shape):
    # 1. Configuration
    MAX_FORCE = 1.0
    N_SAMPLES = 15      # How many actions to test per axis (15x15 = 225 total)
    DT = 0.05           # Simulation step size
    PREDICT_STEPS = 10  # Look ahead 10 steps (~0.5 seconds)
    
    best_score = -float('inf')
    best_action = np.array([0.0, 0.0])
    
    # 2. Search Space: Sample possible actions (Forces in X and Y)
    # We scan from -1.0 to 1.0
    sample_space = np.linspace(-MAX_FORCE, MAX_FORCE, N_SAMPLES)
    
    for fx in sample_space:
        for fy in sample_space:
            action = np.array([fx, fy])
            
            # 3. Predict Trajectory
            traj = simulate_trajectory(current_pos, current_vel, action, dt=DT, steps=PREDICT_STEPS)
            
            # 4. Evaluate (Score)
            
            # A. Collision Check
            collision = False
            for pt in traj:
                if check_collision(pt, maze_map, shape):
                    collision = True
                    break
            
            if collision:
                score = -1000.0 # Penalty for hitting wall
            else:
                # B. Distance to Goal (Closer is better)
                # We look at the END of the predicted path
                final_pt = traj[-1]
                dist_score = -np.linalg.norm(final_pt - target_pos) 
                
                # C. Alignment (Optional: Are we moving toward target?)
                # Dot product of velocity and direction to target
                to_target = target_pos - current_pos
                to_target /= (np.linalg.norm(to_target) + 1e-6)
                vel_norm = current_vel / (np.linalg.norm(current_vel) + 1e-6)
                align_score = np.dot(vel_norm, to_target)
                
                # WEIGHTS: Tune these to change behavior
                # Safety is hard constraint (collision check above)
                # We prioritize getting to the target
                score = (2.0 * dist_score) + (0.5 * align_score)
            
            if score > best_score:
                best_score = score
                best_action = action
                
    return best_action

# --- 3. LIVE VISUALIZATION FUNCTION ---
def update_plot(env, hires_map, start_node, goal_node, path_nodes, current_pos_node, ax_real, ax_grid, target_node=None):
    """
    Updates the existing axes with new data.
    """
    # 1. Real MuJoCo Image
    real_img = env.render()
    ax_real.clear()
    ax_real.imshow(real_img)
    ax_real.set_title("MuJoCo View (Live)")
    ax_real.axis('off')

    # 2. Grid Map View
    ax_grid.clear()
    ax_grid.imshow(hires_map, cmap='Greys', origin='upper')
    
    # --- VISUALIZE PATH NODES ---
    if path_nodes:
        rs = [p[0] for p in path_nodes]
        cs = [p[1] for p in path_nodes]
        # Draw the line
        ax_grid.plot(cs, rs, c='blue', linewidth=1, alpha=0.5, label="Path Line")
        # Draw the individual NODES (What you asked for)
        ax_grid.scatter(cs, rs, c='cyan', s=10, marker='o', alpha=0.8, zorder=2, label="Nodes")
        
    ax_grid.scatter(start_node[1], start_node[0], c='green', s=100, label="Start")
    ax_grid.scatter(goal_node[1], goal_node[0], c='red', s=150, marker='*', label="Goal")
    
    # Plot Dynamic Element (Current Agent Position)
    ax_grid.scatter(current_pos_node[1], current_pos_node[0], c='orange', s=120, edgecolors='black', label="Agent")
    
    # Plot the IMMEDIATE TARGET (The node we are aiming at) 
    if target_node is not None:
        ax_grid.scatter(target_node[1], target_node[0], c='magenta', s=150, marker='x', linewidth=3, zorder=5, label="Target")
    
    ax_grid.set_title(f"Grid Tracking (Scale {RESOLUTION_SCALE}x)")
    ax_grid.legend(loc='upper right', fontsize='small')

    # Draw and Pause
    plt.draw()
    plt.pause(0.001)

def main_test():
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=1000, reward_type='dense')

    raw_map = np.array(env.unwrapped.maze.maze_map)
    original_shape = raw_map.shape
    hires_map = inflate_maze(raw_map, RESOLUTION_SCALE)
    safe_map = inflate_walls(hires_map, radius=2)
    
    # Initialize Plot
    if SHOW_VISUALIZATION:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    obs, _ = env.reset()
        
    start_xy = obs['observation'][:2]
    goal_xy = obs['desired_goal']
    
    # --- LOGIC: USE INTEGERS ---
    # We must use the integer version for A* and map checking
    start_node_int = world_to_grid(start_xy[0], start_xy[1], original_shape)
    goal_node_int = world_to_grid(goal_xy[0], goal_xy[1], original_shape)
    
    # Snap to safe integers
    start_node_int = get_closest_valid_node(start_node_int, safe_map)
    goal_node_int = get_closest_valid_node(goal_node_int, safe_map)
    
    # Plan Path using integers
    path_nodes = astar_grid(safe_map, start_node_int, goal_node_int)

    current_pos = obs['observation'][:2]
    
    # --- VISUALIZATION: USE FLOATS (OPTIONAL) ---
    if SHOW_VISUALIZATION:
        # 1. Calc float position purely for the "Yellow X" alignment check
        float_r, float_c = world_to_grid_float(current_pos[0], current_pos[1], original_shape)
        
        # 2. Update plot using the INTEGER nodes for the green/red dots
        # (You can pass float_r/float_c to your update_plot function if you modified it to show the X)
        update_plot(env, safe_map, start_node_int, goal_node_int, path_nodes, start_node_int, ax1, ax2)
        
        # Optional: Plot the Yellow X manually here if update_plot doesn't handle it
        ax2.scatter(float_c, float_r, c='gold', marker='x', s=150, linewidth=3, label="Exact Pos")
        ax2.legend()
        
        plt.ioff()
        plt.show() # This will block now
    
    print(f"Starting Data Generation...")

# --- 4. MAIN LOOP ---
def main():
    # Use rgb_array to capture frames for matplotlib
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=1000, reward_type='dense')
    env = NegativeRewardWrapper(env)
    env = WallHitPenaltyWrapper(env, penalty=0.5, act_thresh=0.5)
    env = SuccessBonusWrapper(env, dist_threshold=-0.15, bonus=10.0)
    
    raw_map = np.array(env.unwrapped.maze.maze_map)

    original_shape = raw_map.shape
    hires_map = inflate_maze(raw_map, RESOLUTION_SCALE)
    safe_map = inflate_walls(hires_map, radius=3)

    controller = PDPathFollower(original_shape, RESOLUTION_SCALE, kp=15.0, kd=4.0, lookahead=5)
    
    dataset = []
    success_count = 0
    
    # Initialize Plot only if visualization is on
    if SHOW_VISUALIZATION:
        plt.ion() # Interactive mode on
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    print(f"Starting Data Generation...")
    
    while success_count < TOTAL_EPISODES:
        obs, _ = env.reset()
        
        # Setup Start/Goal
        start_xy = obs['observation'][:2]
        goal_xy = obs['desired_goal']
        
        start_node = world_to_grid(start_xy[0], start_xy[1], original_shape)
        goal_node = world_to_grid(goal_xy[0], goal_xy[1], original_shape)
        
        start_node = get_closest_valid_node(start_node, safe_map)
        goal_node = get_closest_valid_node(goal_node, safe_map)
        
        # Plan Path
        path_nodes = astar_grid(safe_map, start_node, goal_node)
        
        if not path_nodes:
            print("Path planning failed. Retrying...")
            continue

        controller.set_path(path_nodes, obs['desired_goal'])
            
        done = False
        current_path_idx = 3
        episode_data = []
        step_count = 0
        
        print(f"\n--- Episode {success_count + 1} Started ---")
        
        while not done:
            step_count += 1
            
            current_pos = obs['observation'][:2]
            current_vel = obs['observation'][2:]
                
            action = controller.get_action(current_pos, current_vel)

            if SHOW_VISUALIZATION:
                update_plot(env, safe_map, start_node, goal_node, path_nodes, 
                           world_to_grid(*current_pos, original_shape), ax1, ax2, 
                           controller.current_target_node)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            if reward >= 0.85:
                terminated = True
        
            
            episode_data.append({
                'obs': obs['observation'],
                'goal': obs['desired_goal'],
                'action': action,
                'reward': reward,
                'next_obs': next_obs['observation'],
                'done': terminated
            })
            
            obs = next_obs
            done = terminated or truncated
        
        if terminated:
            dataset.extend(episode_data)
            success_count += 1
            print(f"--- Episode {success_count} Success! ---")
            print(f"Step: {step_count:03d} | Action: [{action[0]:.2f}, {action[1]:.2f}] | Reward: {reward:.4f}")
        else:
            print("--- Episode Failed (Time limit) ---")
            
    env.close()

    print(f"Success count: {success_count}")
    
    #Save Data
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} transitions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()