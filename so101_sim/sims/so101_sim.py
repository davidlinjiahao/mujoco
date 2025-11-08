#!/usr/bin/env python3
"""
SO-101 Sim2Real Pick-and-Place Simulator
MuJoCo-based simulation for SO-101/ARM100 robots with AEM-style logging.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import yaml

try:
    import mujoco_viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


@dataclass
class Config:
    """CLI configuration."""
    xml: str = "scenes/so_arm.xml"
    headless: bool = False
    steps: int = 4000
    seed: int = 42
    log: str = "runs/episodes.jsonl"
    task_config: str = "configs/task_pick_place.yaml"


def load_yaml(path: str) -> Dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_xml_files(base_xml: str, scene_files: List[str]) -> str:
    """Merge base robot XML with scene MJCF snippets.
    
    Note: For production, use proper XML parsing. This is a simple concatenation
    approach that works for basic includes.
    """
    # Read base XML
    with open(base_xml, 'r') as f:
        base_content = f.read()
    
    # Position robot on table by modifying the base body position
    # Robot should be mounted on table at height 0.42, positioned to reach objects
    import re
    # Find the base body definition and add position
    base_content = re.sub(
        r'<body name="base" pos="0 0 0"',
        '<body name="base" pos="0.15 0 0.42"',  # On table, can reach cube and bin
        base_content
    )
    
    # Insert include directives before </mujoco>
    includes = ""
    for scene_file in scene_files:
        includes += f'  <include file="{scene_file}"/>\n'
    
    # Insert before closing tag
    merged = base_content.replace('</mujoco>', f'{includes}</mujoco>')
    
    return merged


def load_model(base_xml_path: str, scene_dir: str = "scenes") -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MuJoCo model by merging base XML with scene elements."""
    base_path = Path(base_xml_path).parent
    
    # Scene files to include (absolute paths)
    scene_files = [
        os.path.abspath(os.path.join(scene_dir, "table.xml")),
        os.path.abspath(os.path.join(scene_dir, "cube.xml")),
        os.path.abspath(os.path.join(scene_dir, "bin.xml")),
    ]
    
    # Merge XMLs
    merged_xml = merge_xml_files(base_xml_path, scene_files)
    
    # Write to temporary file to preserve relative paths for assets
    import tempfile
    temp_dir = Path(base_xml_path).parent
    temp_xml = temp_dir / "temp_scene.xml"
    
    with open(temp_xml, 'w') as f:
        f.write(merged_xml)
    
    try:
        # Load from file to maintain relative asset paths
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        data = mujoco.MjData(model)
    finally:
        # Clean up temp file
        if temp_xml.exists():
            temp_xml.unlink()
    
    return model, data


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def initialize_scene(model: mujoco.MjModel, data: mujoco.MjData, task_config: Dict):
    """Initialize scene with randomized cube position."""
    # Set cube initial position with noise
    cube_pos = np.array(task_config['cube_init_xyz'])
    noise = task_config.get('cube_init_noise', 0.01)
    cube_pos[:2] += np.random.uniform(-noise, noise, size=2)
    
    # Find cube body and set position (cube has freejoint: 7 DOFs)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_body_id >= 0:
        # Find qpos indices for cube (freejoint: pos + quat)
        # Typically the robot joints come first, then the cube's 7 DOFs
        # We need to find the right offset
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        if jnt_id < 0:
            # Freejoint is automatically named after body
            jnt_id = model.body_jntadr[cube_body_id]
        
        jnt_qposadr = model.jnt_qposadr[jnt_id]
        
        # Set position (first 3 DOFs)
        data.qpos[jnt_qposadr:jnt_qposadr+3] = cube_pos
        # Set orientation (quaternion: w, x, y, z)
        data.qpos[jnt_qposadr+3:jnt_qposadr+7] = [1, 0, 0, 0]
    
    # Reset velocities
    data.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)


def interpolate_waypoints(current: np.ndarray, target: np.ndarray, steps: int) -> np.ndarray:
    """Linear interpolation between joint configurations."""
    return np.linspace(current, target, steps)


def run_episode(model: mujoco.MjModel, data: mujoco.MjData, 
                task_config: Dict, headless: bool, max_steps: int) -> Dict:
    """Execute pick-and-place episode with waypoint control.
    
    Returns metrics dictionary.
    """
    # Extract waypoints and config
    waypoints = task_config['waypoints']
    interp_steps = task_config['interp_steps_per_phase']
    
    # Detect number of actuators (SO-101 has 6: 5 arm joints + 1 gripper)
    n_ctrl = model.nu
    
    # Build trajectory from all waypoint phases
    # For SO-101: home -> approach -> grasp -> close -> hold -> lift -> place -> release
    phase_order = ['home', 'approach', 'grasp', 'close', 'hold', 'lift', 'place', 'release']
    
    # Filter to only phases that exist in config
    phases = [p for p in phase_order if p in waypoints]
    
    trajectory = []
    
    # Start from home position if it exists, otherwise current position
    if 'home' in waypoints:
        current_q = np.array(waypoints['home'])
    else:
        current_q = data.qpos[:n_ctrl].copy()
    
    for phase in phases:
        target_q = np.array(waypoints[phase])
        
        # Ensure target has correct number of DOFs
        if len(target_q) != n_ctrl:
            print(f"Warning: Phase '{phase}' has {len(target_q)} DOFs, expected {n_ctrl}")
            continue
        
        # Interpolate from current to target
        traj_segment = interpolate_waypoints(current_q, target_q, interp_steps)
        
        # Add all waypoints in this segment
        for q in traj_segment:
            trajectory.append(q)
        
        current_q = target_q
    
    # Print trajectory info
    print(f"Generated trajectory with {len(trajectory)} waypoints across {len(phases)} phases")
    print(f"Phases: {' -> '.join(phases)}")
    print(f"Robot has {n_ctrl} actuators")
    
    # Metrics tracking
    collision_count = 0
    contact_impulses = []
    cube_positions = []
    
    # Viewer setup
    viewer = None
    if not headless and VIEWER_AVAILABLE:
        viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Control loop
    step = 0
    traj_idx = 0
    dt = model.opt.timestep
    
    # Track gripper position for debugging
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    
    min_distance = float('inf')
    closest_step = 0
    
    try:
        while step < max_steps and (viewer is None or viewer.is_alive):
            # Set control from trajectory
            if traj_idx < len(trajectory):
                data.ctrl[:] = trajectory[traj_idx]
                traj_idx += 1
            
            # Step simulation
            mujoco.mj_step(model, data)
            step += 1
            
            # Track cube position
            if cube_body_id >= 0:
                cube_pos = data.xpos[cube_body_id].copy()
                cube_positions.append(cube_pos)
                
                # Track gripper-to-cube distance
                if gripper_body_id >= 0:
                    gripper_pos = data.xpos[gripper_body_id].copy()
                    distance = np.linalg.norm(gripper_pos - cube_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_step = step
                        closest_gripper_pos = gripper_pos.copy()
                        closest_cube_pos = cube_pos.copy()
            
            # Track contacts/collisions
            for i in range(data.ncon):
                contact = data.contact[i]
                # Get contact force from efc_force (constraint forces)
                # For contact i, forces are stored in data.efc_force
                if i < len(data.efc_force):
                    # Approximate impulse from constraint force
                    force_mag = abs(data.efc_force[i])
                    impulse = force_mag * dt
                    if impulse > task_config.get('contact_impulse_threshold', 0.01):
                        collision_count += 1
                        contact_impulses.append(impulse)
            
            # Render
            if viewer is not None:
                viewer.render()
            
    finally:
        if viewer is not None:
            viewer.close()
    
    # Compute metrics
    success = check_success(cube_positions, task_config['success_aabb'])
    
    # Print debugging info
    print(f"\n🎯 GRIPPER-CUBE DEBUG INFO:")
    print(f"Minimum distance: {min_distance*1000:.1f}mm at step {closest_step}")
    if min_distance < float('inf'):
        print(f"Gripper position: [{closest_gripper_pos[0]:.3f}, {closest_gripper_pos[1]:.3f}, {closest_gripper_pos[2]:.3f}]")
        print(f"Cube position:    [{closest_cube_pos[0]:.3f}, {closest_cube_pos[1]:.3f}, {closest_cube_pos[2]:.3f}]")
        print(f"Delta:            [{closest_gripper_pos[0]-closest_cube_pos[0]:.3f}, {closest_gripper_pos[1]-closest_cube_pos[1]:.3f}, {closest_gripper_pos[2]-closest_cube_pos[2]:.3f}]")
    
    metrics = {
        'success': bool(success),  # Convert numpy bool to Python bool
        'collisions': int(collision_count),
        'contact_impulse_mean': float(np.mean(contact_impulses)) if contact_impulses else 0.0,
        'time_s': float(step * dt),
        'min_gripper_cube_distance_mm': float(min_distance * 1000) if min_distance < float('inf') else None
    }
    
    return metrics


def check_success(cube_positions: List[np.ndarray], aabb: Dict) -> bool:
    """Check if cube final position is within target AABB."""
    if not cube_positions:
        return False
    
    final_pos = cube_positions[-1]
    min_bounds = np.array(aabb['min'])
    max_bounds = np.array(aabb['max'])
    
    return np.all(final_pos >= min_bounds) and np.all(final_pos <= max_bounds)


def log_episode(log_path: str, episode_data: Dict):
    """Append episode data to JSONL file (AEM-style schema)."""
    # Ensure runs directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(episode_data) + '\n')


def main():
    """Main entry point."""
    # Parse arguments using simple argparse (tyro alternative for simplicity)
    import argparse
    
    parser = argparse.ArgumentParser(description='SO-101 Sim2Real Pick-and-Place')
    parser.add_argument('--xml', default='scenes/so_arm.xml', help='Path to robot MJCF')
    parser.add_argument('--headless', action='store_true', help='Run without viewer')
    parser.add_argument('--steps', type=int, default=4000, help='Max simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log', default='runs/episodes.jsonl', help='Output log file')
    parser.add_argument('--task-config', default='configs/task_pick_place.yaml', 
                       help='Task configuration YAML')
    
    args = parser.parse_args()
    
    # Check if viewer is available when not headless
    if not args.headless and not VIEWER_AVAILABLE:
        print("Warning: mujoco-viewer not available, running headless")
        args.headless = True
    
    # Set seed
    set_seed(args.seed)
    
    # Load task configuration
    print(f"Loading task config: {args.task_config}")
    task_config = load_yaml(args.task_config)
    
    # Load model and merge scene
    print(f"Loading model: {args.xml}")
    model, data = load_model(args.xml)
    
    # Initialize scene
    print("Initializing scene...")
    initialize_scene(model, data, task_config)
    
    # Run episode
    print(f"Running episode ({'headless' if args.headless else 'with viewer'})...")
    metrics = run_episode(model, data, task_config, args.headless, args.steps)
    
    # Print results
    print("\n" + "="*50)
    print("EPISODE RESULTS")
    print("="*50)
    print(f"Success: {metrics['success']}")
    print(f"Collisions: {metrics['collisions']}")
    print(f"Contact Impulse (mean): {metrics['contact_impulse_mean']:.4f}")
    print(f"Time: {metrics['time_s']:.2f}s")
    print("="*50)
    
    # Log to JSONL (AEM schema)
    episode_id = str(uuid.uuid4())
    episode_data = {
        'episode_id': episode_id,
        'env_type': 'base_sim',
        'task_id': 'pick_place_binA',
        'model_xml': args.xml,
        'seed': args.seed,
        'metrics': metrics
    }
    
    print(f"\nLogging to: {args.log}")
    log_episode(args.log, episode_data)
    
    print("\n✓ Episode complete!")
    
    # TODO: Future enhancements
    # - Load grounded assets (OBJ/GLB meshes) for realistic objects
    # - Replay real teleop trajectories from recorded data
    # - Add domain randomization for sim2real transfer
    # - Implement force/torque sensing for contact-rich manipulation


if __name__ == '__main__':
    main()

