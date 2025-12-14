import numpy as np

class PDPathFollower:
    def __init__(self, original_shape, resolution_scale, kp=15.0, kd=4.0, lookahead=5):
        self.original_shape = original_shape
        self.resolution_scale = resolution_scale
        
        # Tuning Parameters
        self.kp = kp
        self.kd = kd
        self.lookahead = lookahead
        
        # State
        self.path_nodes = []
        self.goal_xy = None
        self.current_path_idx = 0
        self.current_target_node = None # For visualization

    def set_path(self, path_nodes, goal_xy):
        """
        Call this at the start of an episode or when replanning.
        """
        self.path_nodes = path_nodes
        self.goal_xy = goal_xy
        self.current_path_idx = 0
        self.current_target_node = None

    def _update_index(self, current_pos):
        """
        Internal: specific logic to keep the index moving forward.
        Uses a search window to find the closest node on the path.
        """
        if not self.path_nodes:
            return

        # Optimization: Only search the next 50 nodes to prevents jumping backwards
        search_window = 50
        search_start = self.current_path_idx
        search_end = min(self.current_path_idx + search_window, len(self.path_nodes))
        
        closest_dist = float('inf')
        best_idx = self.current_path_idx

        for i in range(search_start, search_end):
            # We need grid_to_world here. 
            # Assuming grid_to_world is available in scope or passed in. 
            # Ideally, reimplement the math here to be self-contained:
            node_r, node_c = self.path_nodes[i]
            
            # --- Grid to World Math (Inline) ---
            rows, cols = self.original_shape
            # Assuming MANUAL_OFFSET_X/Y are 0 as per your config
            off_x = (cols / 2.0)
            off_y = (rows / 2.0)
            
            x = (node_c + 0.5) / self.resolution_scale - off_x
            y = off_y - (node_r + 0.5) / self.resolution_scale
            node_pos = np.array([x, y])
            # -----------------------------------

            dist = np.linalg.norm(current_pos - node_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_idx = i
        
        self.current_path_idx = best_idx

    def get_action(self, current_pos, current_vel):
        """
        Calculates the PD action based on the lookahead logic.
        """
        if not self.path_nodes:
            return np.zeros(2) # No path, do nothing

        # 1. Update where we are on the path
        self._update_index(current_pos)

        # 2. Calculate Target (The logic you provided)
        if self.current_path_idx < len(self.path_nodes):
            # Aim 'lookahead' steps ahead
            target_idx = min(self.current_path_idx + self.lookahead, len(self.path_nodes) - 1)
            
            self.current_target_node = self.path_nodes[target_idx]
            
            # Convert target node to world space
            # (Reusing the inline math for safety/speed)
            node_r, node_c = self.current_target_node
            rows, cols = self.original_shape
            off_x = cols / 2.0
            off_y = rows / 2.0
            tx = (node_c + 0.5) / self.resolution_scale - off_x
            ty = off_y - (node_r + 0.5) / self.resolution_scale
            target_world = np.array([tx, ty])

            # Special Case: If nearing end, aim at exact goal
            if self.current_path_idx >= len(self.path_nodes) - 3:
                target_world = self.goal_xy
        else:
            target_world = self.goal_xy

        # 3. PD Controller Logic
        error = target_world - current_pos
        action = (error * self.kp) - (current_vel * self.kd)

        # Speed Hack: Reduce damping if we are aligned with target
        # (Allows faster movement on straight lines)
        if self.current_path_idx < len(self.path_nodes) - 3:
            alignment = np.dot(error, current_vel)
            if alignment > 0:
                action += current_vel * 2.0 

        return np.clip(action, -1.0, 1.0)