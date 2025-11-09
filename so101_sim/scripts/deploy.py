#!/usr/bin/env python3
"""
Deploy trained LeRobot policy in MuJoCo simulation

This script loads a trained ACT/Diffusion policy from LeRobot and
deploys it in your SO-101 MuJoCo simulation for evaluation.

Usage:
    python3 scripts/deploy.py \
        --policy-path outputs/train/act_sim_v1/checkpoints/last/pretrained_model \
        --episodes 20 \
        --render \
        --record
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

# Import utility functions
from utilities.mujoco_utils import reset_cube_position, randomize_cube_position

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not installed - video recording disabled")

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    import torch
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("❌ LeRobot not installed")
    print("   Install with: pip install lerobot")
    sys.exit(1)


def merge_xml_files(xml_files, output_path="scenes/_merged_deploy_lerobot.xml"):
    """Merge multiple MJCF XML files"""
    output_dir = Path(output_path).parent
    
    includes = []
    for xml_file in xml_files:
        xml_path = Path(xml_file)
        includes.append(f'  <include file="{xml_path.name}"/>')
    
    includes_str = "\n".join(includes)
    merged_xml = f"""<mujoco>
{includes_str}
</mujoco>"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(merged_xml)
    
    return str(output_path)


def is_cube_in_bin(cube_pos, env_type="grounded_sim"):
    """
    Check if cube is in target bin using environment-specific boundaries
    
    Args:
        cube_pos: [x, y, z] position of cube
        env_type: "base_sim" or "grounded_sim"
    
    Returns:
        bool: True if cube is in bin
    """
    if env_type == "grounded_sim":
        # Mesh bin boundaries (from scanned bin mesh)
        # Center: [0.495330, -0.193083, 0.420000]
        bin_aabb_min = [0.401, -0.295, 0.420]
        bin_aabb_max = [0.590, -0.091, 0.520]
    else:  # base_sim
        # Simple box bin boundaries (from bin.xml)
        # Center: [0.4, -0.18, 0.42], size: [0.08, 0.08, 0.04]
        bin_aabb_min = [0.320, -0.260, 0.420]
        bin_aabb_max = [0.480, -0.100, 0.480]
    
    return all(bin_aabb_min[i] <= cube_pos[i] <= bin_aabb_max[i] for i in range(3))


def is_cube_out_of_reach(cube_pos, center=[0.4, 0.0, 0.42], max_radius=0.35):
    """
    Check if cube has fallen too far from the reachable workspace
    
    Args:
        cube_pos: Current cube position [x, y, z]
        center: Center of reachable workspace
        max_radius: Maximum reachable distance (meters)
    
    Returns:
        bool: True if cube is out of reach
    """
    # Calculate horizontal distance from workspace center
    dx = cube_pos[0] - center[0]
    dy = cube_pos[1] - center[1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    
    # Also check if it fell off the table (z < 0.3)
    if cube_pos[2] < 0.3:
        return True
    
    return horizontal_dist > max_radius


def capture_camera_frame(model, data, camera_name="c920", width=640, height=480):
    """Capture RGB frame from MuJoCo camera"""
    if not hasattr(capture_camera_frame, 'renderer'):
        capture_camera_frame.renderer = mujoco.Renderer(model, height=height, width=width)
        capture_camera_frame.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found")
    
    capture_camera_frame.renderer.update_scene(data, camera=camera_id)
    frame_rgb = capture_camera_frame.renderer.render()
    
    return frame_rgb


class LeRobotPolicyWrapper:
    """Wrapper for LeRobot policies (ACT, Diffusion, etc.)"""
    
    def __init__(self, policy_path, device="mps"):
        self.policy_path = Path(policy_path)
        self.device = device
        
        print(f"\n🧠 Loading LeRobot policy...")
        print(f"   Path: {policy_path}")
        print(f"   Device: {device}")
        
        # Load policy (ACT or Diffusion)
        try:
            # Try ACT first
            self.policy = ACTPolicy.from_pretrained(str(policy_path))
            self.policy_type = "ACT"
        except:
            try:
                # Try Diffusion
                self.policy = DiffusionPolicy.from_pretrained(str(policy_path))
                self.policy_type = "Diffusion"
            except Exception as e:
                raise ValueError(f"Failed to load policy: {e}")
        
        self.policy.to(device)
        self.policy.eval()
        
        print(f"✅ Loaded {self.policy_type} policy")
        
        # Get observation and action specs
        self.obs_dim = None  # Will be inferred from first observation
        self.action_dim = 6  # SO-101 has 6 joints
    
    def predict(self, observation):
        """
        Predict action from observation
        
        Args:
            observation: dict with keys:
                - 'observation.state': (state_dim,) tensor
                - 'observation.images.front': (H, W, 3) tensor or (C, H, W)
        
        Returns:
            action: (6,) numpy array of joint positions
        """
        with torch.no_grad():
            # Convert numpy arrays to tensors and move to device
            obs = {}
            for k, v in observation.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.to(self.device)
                else:
                    # Convert numpy to tensor (float32 for MPS compatibility)
                    if v.dtype == np.float64:
                        v = v.astype(np.float32)
                    elif v.dtype == np.uint8:
                        # Images are uint8, convert to float32 and normalize to [0, 1]
                        v = v.astype(np.float32) / 255.0
                    obs[k] = torch.from_numpy(v).to(self.device)
            
            # Add batch dimension if needed
            for key in obs:
                if len(obs[key].shape) == 1:  # State vector
                    obs[key] = obs[key].unsqueeze(0)
                elif len(obs[key].shape) == 3:  # Image (H, W, C)
                    obs[key] = obs[key].permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
            
            # Get action from policy
            output = self.policy.select_action(obs)
            
            # Extract action (handle different output formats)
            if isinstance(output, dict):
                action = output['action']
            else:
                action = output
            
            # Remove batch dimension and convert to numpy
            action = action.squeeze(0).cpu().numpy()
            
            return action


def main():
    parser = argparse.ArgumentParser(description="Deploy LeRobot policy in MuJoCo simulation")
    parser.add_argument("--policy-path", required=True, help="Path to trained policy checkpoint")
    parser.add_argument("--env", type=str, default="base_sim", choices=["base_sim", "grounded_sim"],
                       help="Environment type: base_sim (merged XMLs) or grounded_sim (complete scene)")
    parser.add_argument("--arm-xml", default="scenes/so_arm.xml", help="Robot model XML (used only for base_sim)")
    parser.add_argument("--scene-xmls", nargs="+",
                       default=["scenes/table.xml", "scenes/cube.xml", "scenes/bin.xml", "scenes/camera_c920.xml"],
                       help="Scene XML files (used only for base_sim)")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--record", action="store_true", help="Record videos")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--hz", type=float, default=10.0, help="Control frequency (Hz)")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"],
                       help="Device for policy inference")
    parser.add_argument("--randomize", action="store_true", help="Randomize cube position")
    parser.add_argument("--camera-name", default="c920", help="Camera name for rendering")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LEROBOT POLICY DEPLOYMENT IN MUJOCO SIMULATION")
    print("="*80)
    print(f"Policy: {args.policy_path}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Control frequency: {args.hz} Hz")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Load policy
    policy_wrapper = LeRobotPolicyWrapper(args.policy_path, device=args.device)
    
    # Load MuJoCo scene
    print("🔧 Loading MuJoCo scene...")
    if args.env == "grounded_sim":
        # Use complete grounded scene file
        scene_path = "scenes/grounded_scene.xml"
        print(f"   Using grounded scene: {scene_path}")
        model = mujoco.MjModel.from_xml_path(scene_path)
    else:
        # Use base_sim with merged XMLs
        merged_xml = merge_xml_files([args.arm_xml] + args.scene_xmls)
        model = mujoco.MjModel.from_xml_path(merged_xml)
    data = mujoco.MjData(model)
    print(f"✅ MuJoCo loaded\n")
    
    # Open viewer if requested
    viewer_handle = None
    if args.render:
        viewer_handle = mujoco.viewer.launch_passive(model, data)
        viewer_handle.cam.distance = 1.2
        viewer_handle.cam.azimuth = 135
        viewer_handle.cam.elevation = -25
        viewer_handle.cam.lookat[:] = [0.4, 0.0, 0.45]
        print("🖥️  Viewer opened\n")
    
    # Episode results
    results = []
    dt = 1.0 / args.hz
    
    # Run episodes
    for episode_idx in range(args.episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_idx + 1}/{args.episodes}")
        print(f"{'='*80}")
        
        # Reset environment
        mujoco.mj_resetData(model, data)
        if args.randomize:
            randomize_cube_position(model, data)
        else:
            reset_cube_position(model, data)
        
        # Get initial cube position
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_start_pos = data.xpos[cube_body_id].copy()
        print(f"Cube start: [{cube_start_pos[0]:.3f}, {cube_start_pos[1]:.3f}, {cube_start_pos[2]:.3f}]")
        
        # Video recording
        video_frames = []
        
        # Run episode
        success = False
        episode_start = time.time()
        
        for step in range(args.max_steps):
            step_start = time.time()
            
            # Get observation
            joint_positions = data.qpos[:6].copy()
            cube_pos = data.xpos[cube_body_id].copy()
            BIN_CENTER = np.array([0.4, -0.18, 0.42])
            cube_to_bin = BIN_CENTER - cube_pos
            
            # Build state observation (12D: joints + cube_pos + cube_to_bin)
            state = np.concatenate([joint_positions, cube_pos, cube_to_bin]).astype(np.float32)
            
            # Capture camera frame
            rgb_frame = capture_camera_frame(model, data, args.camera_name)
            
            # Build observation dict for policy
            observation = {
                'observation.state': torch.from_numpy(state),
                'observation.images.front': torch.from_numpy(rgb_frame).float() / 255.0
            }
            
            # Get action from policy
            action = policy_wrapper.predict(observation)
            
            # Clip action to joint limits (safety)
            action = np.clip(action, -3.0, 3.0)
            
            # Apply action
            data.ctrl[:6] = action
            
            # Step simulation
            n_steps = int(dt / model.opt.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            
            # Update viewer
            if viewer_handle:
                viewer_handle.sync()
            
            # Record frame
            if args.record:
                video_frames.append(rgb_frame)
            
            # Check success IMMEDIATELY
            cube_pos = data.xpos[cube_body_id].copy()
            if is_cube_in_bin(cube_pos, env_type=args.env):
                success = True
                print(f"\n✅ SUCCESS at step {step}!")
                print(f"   Cube final: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
                break
            
            # Check if cube is out of reach - auto-reset to give policy another chance
            if is_cube_out_of_reach(cube_pos):
                print(f"\n🔄 Cube out of reach at step {step} - resetting")
                reset_cube_position(model, data)
                # Give robot a moment to adjust
                for _ in range(10):
                    mujoco.mj_step(model, data)
                continue
            
            # Print status more frequently
            if step % 20 == 0:
                print(f"Step {step:4d}/{args.max_steps} | Cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]", flush=True)
            
            # Maintain control rate
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        episode_time = time.time() - episode_start
        
        # Episode summary
        cube_final_pos = data.xpos[cube_body_id].copy()
        
        result = {
            'episode': episode_idx + 1,
            'success': success,
            'steps': step + 1,
            'time_s': episode_time,
            'cube_start': cube_start_pos.tolist(),
            'cube_final': cube_final_pos.tolist(),
        }
        results.append(result)
        
        if not success:
            print(f"\n❌ Episode {episode_idx + 1} failed (timeout)")
            print(f"   Cube final: [{cube_final_pos[0]:.3f}, {cube_final_pos[1]:.3f}, {cube_final_pos[2]:.3f}]")
        
        # Save video
        if args.record and video_frames:
            output_dir = Path("runs/videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / f"deploy_ep{episode_idx+1:03d}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = video_frames[0].shape[:2]
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.hz, (width, height))
            
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            print(f"📹 Video saved: {video_path}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    successes = sum(1 for r in results if r['success'])
    success_rate = 100.0 * successes / len(results)
    mean_time = np.mean([r['time_s'] for r in results])
    mean_steps = np.mean([r['steps'] for r in results])
    
    print(f"Total episodes: {len(results)}")
    print(f"Successes: {successes}/{len(results)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Mean time: {mean_time:.2f}s")
    print(f"Mean steps: {mean_steps:.1f}")
    
    # Save results
    results_dir = Path("runs")
    results_dir.mkdir(exist_ok=True)
    
    # Extract model name from policy path for filename
    policy_name = Path(args.policy_path).parent.name if Path(args.policy_path).parent.name != "pretrained_model" else Path(args.policy_path).parent.parent.name
    results_path = results_dir / f"eval_{args.env}_{policy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_path, 'w') as f:
        json.dump({
            'policy_path': str(args.policy_path),
            'environment': args.env,
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'mean_time': mean_time,
            'mean_steps': mean_steps,
            'episodes': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    print("="*80 + "\n")
    
    # Cleanup
    if viewer_handle:
        viewer_handle.close()


if __name__ == "__main__":
    main()

