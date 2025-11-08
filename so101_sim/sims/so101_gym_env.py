"""
Gymnasium environment for SO-101 robot pick-and-place task
State-based RL with low-dimensional observations
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from pathlib import Path


class SO101PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for SO-101 pick-and-place task
    
    State Space (31D):
        - Joint positions [6D]
        - Joint velocities [6D]
        - Cube position [3D]
        - Cube orientation [4D] (quaternion)
        - Cube linear velocity [3D]
        - Cube angular velocity [3D]
        - Gripper position [3D]
        - Gripper-to-cube distance [1D]
        - Cube-to-bin distance [1D]
        - Cube-in-bin flag [1D]
        
    Action Space (6D continuous):
        - Joint position deltas [-1, 1] normalized
        - Applied as small position changes each step
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        xml_path='scenes/so_arm.xml',
        max_episode_steps=500,
        render_mode=None,
        cube_init_pos=None,
        cube_init_noise=0.02,
        action_scale=0.05,
        reward_weights=None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.cube_init_noise = cube_init_noise
        self.action_scale = action_scale  # Max joint delta per step
        
        # Default cube starting position (on table near robot)
        self.cube_init_pos = cube_init_pos or np.array([0.4, 0.15, 0.4375])
        
        # Target bin AABB (from config)
        self.bin_min = np.array([0.32, -0.28, 0.42])
        self.bin_max = np.array([0.48, -0.12, 0.50])
        self.bin_center = (self.bin_min + self.bin_max) / 2
        
        # Reward weights (can be tuned for curriculum learning)
        self.reward_weights = reward_weights or {
            'approach': 1.0,
            'grasp': 10.0,
            'lift': 20.0,
            'place': 100.0,
            'collision': -0.01,
            'action': -0.001,
            'time': -0.1
        }
        
        # Load MuJoCo model
        print(f"Loading MuJoCo model from {xml_path}...")
        self.model, self.data = self._load_model(xml_path)
        
        # Get important body/joint IDs
        self._get_ids()
        
        # Define Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0.0
        self._success = False
        
        # For rendering
        self._viewer = None
        
    def _load_model(self, xml_path):
        """Load and merge MuJoCo XML files"""
        from so101_sim import merge_xml_files  # Reuse existing function
        import os
        import tempfile
        
        base_path = Path(xml_path).parent.parent
        scene_files = [
            os.path.abspath(str(base_path / 'scenes' / 'table.xml')),
            os.path.abspath(str(base_path / 'scenes' / 'cube.xml')),
            os.path.abspath(str(base_path / 'scenes' / 'bin.xml'))
        ]
        
        merged_xml = merge_xml_files(os.path.abspath(xml_path), scene_files)
        
        # Write to unique temp file to avoid race conditions with parallel envs
        # Use tempfile to ensure uniqueness across processes
        temp_dir = Path(xml_path).parent
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=temp_dir, delete=False) as f:
            f.write(merged_xml)
            temp_xml_path = f.name
        
        try:
            model = mujoco.MjModel.from_xml_path(temp_xml_path)
            data = mujoco.MjData(model)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_xml_path)
            except:
                pass  # Ignore cleanup errors
        
        return model, data
    
    def _get_ids(self):
        """Get MuJoCo IDs for important bodies/joints"""
        # Joint IDs (actuated joints)
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_flex_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            'gripper_joint'
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.joint_names]
        
        # Body IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        
        # Site ID for gripper (use link6 as proxy for end-effector)
        try:
            self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'Link6')
        except:
            self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        
        # Joint ranges (for clipping actions)
        self.joint_ranges = np.array([
            [-1.92, 1.92],   # shoulder_pan
            [-1.75, 1.75],   # shoulder_lift
            [-1.69, 1.69],   # elbow_flex
            [-1.66, 1.66],   # wrist_flex
            [-2.74, 2.84],   # wrist_roll
            [-0.17, 1.75]    # gripper
        ])
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set robot to home position (all zeros)
        home_qpos = np.zeros(6)
        home_qpos[-1] = -0.15  # Gripper open
        self.data.qpos[:6] = home_qpos
        self.data.ctrl[:6] = home_qpos
        
        # Randomize cube position with noise
        cube_pos = self.cube_init_pos.copy()
        if self.np_random is not None:
            noise = self.np_random.uniform(-self.cube_init_noise, self.cube_init_noise, size=3)
            cube_pos += noise
            cube_pos[2] = max(cube_pos[2], 0.4375)  # Keep above table
        
        # Set cube position (qpos indices 6-12: 3 pos + 4 quat)
        self.data.qpos[6:9] = cube_pos  # Position
        self.data.qpos[9:13] = np.array([1, 0, 0, 0])  # Quaternion (identity)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset episode tracking
        self._step_count = 0
        self._episode_reward = 0.0
        self._success = False
        
        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        # Convert action to joint position deltas
        action = np.clip(action, -1.0, 1.0)
        delta_qpos = action * self.action_scale
        
        # Apply action to joints (position control)
        target_qpos = self.data.qpos[:6] + delta_qpos
        target_qpos = np.clip(target_qpos, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self.data.ctrl[:6] = target_qpos
        
        # Step physics (multiple substeps for stability)
        for _ in range(10):  # 10 substeps = 0.02s at 0.002s timestep
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = (self._step_count >= self.max_episode_steps)
        
        # Update tracking
        self._step_count += 1
        self._episode_reward += reward
        
        # Get info
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Extract 31D state vector from MuJoCo data"""
        # Joint positions [6D]
        joint_pos = self.data.qpos[:6].copy()
        
        # Joint velocities [6D]
        joint_vel = self.data.qvel[:6].copy()
        
        # Cube state [13D]
        cube_pos = self.data.qpos[6:9].copy()  # Position [3D]
        cube_quat = self.data.qpos[9:13].copy()  # Quaternion [4D]
        cube_linvel = self.data.qvel[6:9].copy()  # Linear velocity [3D]
        cube_angvel = self.data.qvel[9:12].copy()  # Angular velocity [3D]
        
        # Gripper position [3D]
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        
        # Derived features [3D]
        gripper_to_cube_dist = np.linalg.norm(gripper_pos - cube_pos)
        cube_to_bin_dist = np.linalg.norm(cube_pos - self.bin_center)
        cube_in_bin = self._is_cube_in_bin(cube_pos)
        
        # Concatenate into 31D vector
        obs = np.concatenate([
            joint_pos,              # [6D]
            joint_vel,              # [6D]
            cube_pos,               # [3D]
            cube_quat,              # [4D]
            cube_linvel,            # [3D]
            cube_angvel,            # [3D]
            gripper_pos,            # [3D]
            [gripper_to_cube_dist], # [1D]
            [cube_to_bin_dist],     # [1D]
            [float(cube_in_bin)]    # [1D]
        ]).astype(np.float32)      # Total: 31D
        
        return obs
    
    def _compute_reward(self, action):
        """Compute dense reward with multiple components"""
        reward = 0.0
        
        # Get state
        cube_pos = self.data.qpos[6:9]
        gripper_pos = self.data.xpos[self.gripper_body_id]
        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_to_bin = np.linalg.norm(cube_pos - self.bin_center)
        cube_height = cube_pos[2]
        table_height = 0.42
        
        # 1. Approach reward (dense guidance to cube)
        approach_reward = -self.reward_weights['approach'] * gripper_to_cube
        reward += approach_reward
        
        # 2. Grasp detection (contact-based)
        is_grasping = self._is_grasping()
        if is_grasping:
            reward += self.reward_weights['grasp']
            
            # 3. Lift reward (if grasping and cube is lifted)
            if cube_height > table_height + 0.03:  # 3cm above table
                lift_bonus = self.reward_weights['lift'] * (cube_height - table_height)
                reward += lift_bonus
                
                # 4. Transport reward (bring cube to bin)
                transport_reward = -0.5 * cube_to_bin
                reward += transport_reward
                
                # 5. Place reward (terminal success)
                if self._is_cube_in_bin(cube_pos):
                    reward += self.reward_weights['place']
                    self._success = True
        
        # Penalties
        
        # Action smoothness (L2 norm)
        action_penalty = self.reward_weights['action'] * np.sum(action**2)
        reward += action_penalty
        
        # Collision penalty
        collision_impulse = self._get_collision_impulse()
        collision_penalty = self.reward_weights['collision'] * collision_impulse
        reward += collision_penalty
        
        # Dropped cube penalty
        if cube_pos[2] < 0.1:  # Cube fell on floor
            reward -= 50.0
        
        # Time penalty (encourage efficiency)
        reward += self.reward_weights['time']
        
        return reward
    
    def _is_grasping(self):
        """Check if gripper is in contact with cube"""
        # Check for contacts between gripper and cube
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Get body IDs for contact geoms
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            # Check if one is gripper and other is cube
            if (body1 == self.gripper_body_id and body2 == self.cube_body_id) or \
               (body1 == self.cube_body_id and body2 == self.gripper_body_id):
                # Check if gripper is closed enough
                gripper_pos = self.data.qpos[5]  # Gripper joint
                if gripper_pos > 0.5:  # Gripper closed threshold
                    return True
        
        return False
    
    def _is_cube_in_bin(self, cube_pos):
        """Check if cube center is within bin AABB"""
        return np.all(cube_pos >= self.bin_min) and np.all(cube_pos <= self.bin_max)
    
    def _get_collision_impulse(self):
        """Get total collision impulse magnitude"""
        total_impulse = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Approximate impulse from contact force
            if i < len(self.data.efc_force):
                total_impulse += abs(self.data.efc_force[i])
        return total_impulse
    
    def _check_termination(self):
        """Check if episode should terminate (success or failure)"""
        cube_pos = self.data.qpos[6:9]
        
        # Success: cube in bin
        if self._is_cube_in_bin(cube_pos):
            self._success = True
            return True
        
        # Failure: cube fell on floor
        if cube_pos[2] < 0.1:
            return True
        
        return False
    
    def _get_info(self):
        """Additional information for logging"""
        cube_pos = self.data.qpos[6:9]
        gripper_pos = self.data.xpos[self.gripper_body_id]
        
        return {
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'success': self._success,
            'cube_pos': cube_pos.copy(),
            'gripper_pos': gripper_pos.copy(),
            'gripper_to_cube': np.linalg.norm(gripper_pos - cube_pos),
            'cube_to_bin': np.linalg.norm(cube_pos - self.bin_center),
            'cube_in_bin': self._is_cube_in_bin(cube_pos),
            'is_grasping': self._is_grasping(),
        }
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == 'human':
            if self._viewer is None:
                import mujoco_viewer
                self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self._viewer.render()
        elif self.render_mode == 'rgb_array':
            # TODO: Implement offscreen rendering
            pass
    
    def close(self):
        """Clean up resources"""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    # Quick test
    print("Testing SO101PickPlaceEnv...")
    env = SO101PickPlaceEnv(render_mode=None)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, gripper_to_cube={info['gripper_to_cube']:.3f}m")
        
        if terminated or truncated:
            break
    
    print("✅ Environment test passed!")
    env.close()

