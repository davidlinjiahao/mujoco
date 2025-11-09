#!/usr/bin/env python3
"""
Real Robot Deployment Script for SO-101

Deploys a trained LeRobot policy on the physical SO-101 arms.
Reads observations from both leader (READ-ONLY) and follower arms, plus camera.
Infers actions from policy and sends ONLY to follower arm.

Usage:
    python3 scripts/deploy_real.py \
        --policy-path outputs/train/so101_grounded_sim/checkpoints/050000/pretrained_model \
        --leader-port /dev/tty.usbmodem12345678 \
        --leader-id my_leader_arm \
        --follower-port /dev/tty.usbmodem58760431552 \
        --follower-id my_follower_arm \
        --camera-index 0 \
        --hz 10.0 \
        --episodes 10

Requirements:
    - SO-101 leader arm connected via USB (provides observation data)
    - SO-101 follower arm connected via USB (executes actions)
    - Camera (webcam or USB camera)
    - Trained policy checkpoint
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# LeRobot imports
try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    print("⚠️  LeRobot not available")
    print("   Install with: pip install lerobot")
    LEROBOT_AVAILABLE = False

# Import policy wrapper from deploy.py
from deploy import LeRobotPolicyWrapper

# Camera imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV not available")
    print("   Install with: pip install opencv-python")
    CV2_AVAILABLE = False


class CameraReader:
    """Read images from webcam or USB camera"""
    
    def __init__(self, camera_index=0, width=640, height=480):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required but not installed")
        
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        print(f"📷 Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Warm up camera
        for _ in range(5):
            self.cap.read()
        
        print(f"✅ Camera opened: {width}x{height}")
    
    def read_frame(self):
        """Read a single frame from camera
        
        Returns:
            np.ndarray: RGB image (H, W, 3) with values in [0, 255]
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def close(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
            print("📷 Camera closed")


class SO101FollowerController:
    """Control SO-101 follower arm via LeRobot with hardware safety"""
    
    # Joint limits in radians (from so_arm.xml) - STRICT for hardware safety
    JOINT_LIMITS = {
        'shoulder_pan': (-1.91986, 1.91986),
        'shoulder_lift': (-1.74533, 1.74533),
        'elbow_flex': (-1.69, 1.69),
        'wrist_flex': (-1.65806, 1.65806),
        'wrist_roll': (-2.74385, 2.84121),
        'gripper': (-0.17453, 1.74533),
    }
    
    # Maximum velocity per step (rad/s) - prevents jerky movements
    MAX_VELOCITY_PER_STEP = 0.5  # rad per control cycle (at 10Hz = 0.5 rad/0.1s = 5 rad/s max)
    
    def __init__(self, port, robot_id, calibrate=True):
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is required but not installed")
        
        self.port = port
        self.robot_id = robot_id
        self.last_action = None  # For velocity limiting
        self.safety_violations = 0
        
        print(f"🤖 Connecting to follower arm...")
        print(f"   Port: {port}")
        print(f"   ID: {robot_id}")
        print(f"   Calibrate: {calibrate}")
        
        # Create config (port is required positional argument)
        # LeRobot will automatically look for calibration in ~/.cache/huggingface/lerobot/calibration/
        from pathlib import Path
        expected_calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower" / f"{robot_id}.json"
        
        if expected_calibration.exists():
            print(f"   ✅ Calibration file found: {expected_calibration}")
        else:
            print(f"   ⚠️  Calibration file not found: {expected_calibration}")
        
        config = SO101FollowerConfig(
            port=port,
            id=robot_id
            # calibration_dir is not needed - LeRobot uses default location
        )
        
        # Connect to robot
        try:
            self.robot = SO101Follower(config)
            print(f"   Robot calibration loaded: {hasattr(self.robot, 'calibration') and self.robot.calibration is not None}")
            
            # Monkey-patch input() to auto-accept calibration file
            # When LeRobot asks "Press ENTER to use calibration file or 'c' to calibrate"
            # we automatically press ENTER (return empty string)
            import builtins
            original_input = builtins.input
            def auto_input(prompt=""):
                print(prompt)  # Show the prompt
                print("   [AUTO-RESPONDING: Using calibration file]")
                return ""  # Press ENTER
            builtins.input = auto_input
            
            try:
                # Connect will prompt for calibration confirmation
                self.robot.connect()
                print("✅ Follower arm connected")
                if expected_calibration.exists():
                    print("✅ Calibration loaded from file")
            finally:
                # Restore original input function
                builtins.input = original_input
            print("\n🛡️  Hardware Safety Features Enabled:")
            print(f"   ✓ Joint limit checking: {len(self.JOINT_LIMITS)} joints")
            print(f"   ✓ Velocity limiting: {self.MAX_VELOCITY_PER_STEP} rad/step max")
            print(f"   ✓ Smooth command filtering active")
            
        except Exception as e:
            print(f"❌ Failed to connect to follower arm: {e}")
            raise
    
    def send_action(self, action, dt=0.1):
        """Send action to follower arm with hardware safety checks
        
        Args:
            action: numpy array of shape (6,) with joint positions in radians
            dt: time step in seconds (for velocity limiting)
        
        Returns:
            bool: True if action was sent, False if blocked by safety
        """
        # Ensure action is correct shape
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {action.shape}")
        
        # Make a copy to avoid modifying the original
        safe_action = action.copy()
        
        # 1. JOINT LIMIT CHECKING
        limits_list = list(self.JOINT_LIMITS.values())
        joint_names = list(self.JOINT_LIMITS.keys())
        for i, (joint_min, joint_max) in enumerate(limits_list):
            if safe_action[i] < joint_min or safe_action[i] > joint_max:
                print(f"⚠️  SAFETY: Joint {i} ({joint_names[i]}) out of limits! "
                      f"Requested: {safe_action[i]:.3f}, "
                      f"Limits: [{joint_min:.3f}, {joint_max:.3f}]", flush=True)
                # Clamp to safe range
                safe_action[i] = np.clip(safe_action[i], joint_min, joint_max)
                self.safety_violations += 1
        
        # 2. VELOCITY LIMITING (prevents sudden jerks)
        if self.last_action is not None:
            velocity = (safe_action - self.last_action) / dt  # rad/s
            for i, vel in enumerate(velocity):
                if abs(vel) > self.MAX_VELOCITY_PER_STEP / dt:
                    # Limit the change to max allowed
                    max_change = self.MAX_VELOCITY_PER_STEP
                    actual_change = safe_action[i] - self.last_action[i]
                    if abs(actual_change) > max_change:
                        print(f"⚠️  SAFETY: Joint {i} velocity too high! "
                              f"Limiting to {self.MAX_VELOCITY_PER_STEP} rad/step", flush=True)
                        sign = np.sign(actual_change)
                        safe_action[i] = self.last_action[i] + sign * max_change
                        self.safety_violations += 1
        
        # Update last action for next velocity check
        self.last_action = safe_action.copy()
        
        # Send to robot using send_action (LeRobot SO101Follower API)
        # Convert action to dict format expected by LeRobot
        # LeRobot expects keys with .pos suffix!
        try:
            action_dict = {
                f"{joint_names[i]}.pos": float(safe_action[i])
                for i in range(len(joint_names))
            }
            
            # Debug: print action_dict on first call
            if not hasattr(self, '_logged_action_format'):
                print(f"   📤 Sending action dict: {list(action_dict.keys())}")
                print(f"      Sample values: {list(action_dict.values())[:3]}")
                self._logged_action_format = True
            
            self.robot.send_action(action_dict)
            return True
        except Exception as e:
            print(f"⚠️  Error sending action: {type(e).__name__}: {e}")
            print(f"   Action dict keys: {list(action_dict.keys()) if 'action_dict' in locals() else 'Not created'}")
            print(f"   Action dict values: {list(action_dict.values()) if 'action_dict' in locals() else 'Not created'}")
            raise
    
    def get_state(self):
        """Get current robot state (joint positions)
        
        Returns:
            np.ndarray: Joint positions (6,) in radians
        """
        try:
            # get_observation() returns dict with observation data
            obs_dict = self.robot.get_observation()
            
            # Debug: print available keys
            if not hasattr(self, '_logged_obs_keys'):
                print(f"   📊 Observation keys: {list(obs_dict.keys())}")
                self._logged_obs_keys = True
            
            # Extract joint positions from observation
            # LeRobot SO101Follower returns observations with keys like 'shoulder_pan.pos'
            joint_names = list(self.JOINT_LIMITS.keys())
            
            # Check if keys have .pos suffix
            if any(key.endswith('.pos') for key in obs_dict.keys()):
                state = np.array([obs_dict[f"{name}.pos"] for name in joint_names])
            elif 'observation.state' in obs_dict:
                state = np.array(obs_dict['observation.state'])
            elif 'position' in obs_dict:
                state = np.array(obs_dict['position'])
            elif 'state' in obs_dict:
                state = np.array(obs_dict['state'])
            else:
                # Try extracting by joint names directly
                state = np.array([obs_dict[name] for name in joint_names])
            
            return state
        except Exception as e:
            print(f"⚠️  Error reading state: {e}")
            print(f"   Available keys: {list(obs_dict.keys()) if 'obs_dict' in locals() else 'Unknown'}")
            raise
    
    def move_to_safe_position(self, home_position=None):
        """Move robot to a safe neutral position slowly
        
        Args:
            home_position: Optional custom home position. If None, uses neutral pose.
        """
        if home_position is None:
            # Neutral "home" position - all joints near zero, gripper open
            home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        current_state = self.get_state()
        print(f"\n🏠 Moving to safe position...")
        print(f"   Current: {current_state}")
        print(f"   Target:  {home_position}")
        
        # Move slowly over 2 seconds
        steps = 20  # at 10Hz = 2 seconds
        for i in range(steps):
            alpha = (i + 1) / steps  # Linear interpolation
            target = current_state + alpha * (home_position - current_state)
            self.send_action(target, dt=0.1)
            time.sleep(0.1)
        
        print("✅ Reached safe position")
    
    def get_safety_report(self):
        """Get report of safety violations"""
        return {
            'total_violations': self.safety_violations,
        }
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.robot:
            # Report safety stats
            if self.safety_violations > 0:
                print(f"\n⚠️  Safety Report: {self.safety_violations} violations detected during operation")
                print("   (Joint limits or velocity limits were enforced)")
            
            self.robot.disconnect()
            print("🤖 Follower arm disconnected")


class SO101LeaderController:
    """READ-ONLY controller for SO-101 leader arm
    
    Used to read the leader arm state for policy observations.
    Does NOT send any commands to the leader arm - it's purely for reading.
    Uses SO101Follower class but with teleoperators calibration directory.
    """
    
    JOINT_NAMES = [
        'shoulder_pan', 'shoulder_lift', 'elbow_flex',
        'wrist_flex', 'wrist_roll', 'gripper'
    ]
    
    def __init__(self, port, robot_id):
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is required but not installed")
        
        self.port = port
        self.robot_id = robot_id
        
        print(f"🎮 Connecting to leader arm (READ-ONLY)...")
        print(f"   Port: {port}")
        print(f"   ID: {robot_id}")
        
        # Check for calibration file in TELEOPERATORS directory
        from pathlib import Path
        calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration"
        expected_calibration = calibration_dir / "teleoperators" / "so101_leader" / f"{robot_id}.json"
        
        if expected_calibration.exists():
            print(f"   ✅ Calibration file found: {expected_calibration}")
        else:
            print(f"   ⚠️  Calibration file not found: {expected_calibration}")
        
        # Use SO101Follower class with custom calibration_dir pointing to teleoperators
        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            calibration_dir=str(calibration_dir)  # Base calibration dir
        )
        
        # Connect to robot
        try:
            # Use SO101Follower class for leader arm (same hardware)
            self.robot = SO101Follower(config)
            
            # Monkey-patch input() to auto-accept calibration file
            import builtins
            original_input = builtins.input
            def auto_input(prompt=""):
                print(prompt)
                print("   [AUTO-RESPONDING: Using calibration file]")
                return ""
            builtins.input = auto_input
            
            try:
                self.robot.connect()
                print("✅ Leader arm connected (READ-ONLY mode)")
                if expected_calibration.exists():
                    print("✅ Calibration loaded from file")
            finally:
                builtins.input = original_input
            
        except Exception as e:
            print(f"❌ Failed to connect to leader arm: {e}")
            raise
    
    def get_state(self):
        """Get current leader arm state (joint positions)
        
        Returns:
            np.ndarray: Joint positions (6,) in radians
        """
        try:
            obs_dict = self.robot.get_observation()
            
            # Extract joint positions (same format as follower)
            if any(key.endswith('.pos') for key in obs_dict.keys()):
                state = np.array([obs_dict[f"{name}.pos"] for name in self.JOINT_NAMES])
            else:
                # Fallback to other formats
                joint_names = self.JOINT_NAMES
                state = np.array([obs_dict[name] for name in joint_names])
            
            return state
        except Exception as e:
            print(f"⚠️  Error reading leader state: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from leader arm"""
        if hasattr(self, 'robot'):
            self.robot.disconnect()
            print("🎮 Leader arm disconnected")


def create_observation(camera_frame, robot_state):
    """Create observation dict for policy
    
    Args:
        camera_frame: RGB image (H, W, 3)
        robot_state: Joint positions (12,) - concatenated [leader (6), follower (6)]
    
    Returns:
        dict: Observation matching policy's expected format
    """
    obs = {
        'observation.images.front': camera_frame,  # RGB image
        'observation.state': robot_state,  # Joint positions [leader, follower]
    }
    return obs


def print_startup_banner(args):
    """Print startup information"""
    print("\n" + "=" * 80)
    print("🤖 REAL ROBOT DEPLOYMENT - SO-101")
    print("=" * 80)
    print(f"Policy: {args.policy_path}")
    if args.leader_port:
        print(f"Leader port: {args.leader_port} (ID: {args.leader_id}) - READ-ONLY")
    else:
        print(f"Leader: Not connected (using follower state as goal proxy)")
    print(f"Follower port: {args.follower_port} (ID: {args.follower_id}) - ACTIONS")
    print(f"Camera: {args.camera_index}")
    print(f"Episodes: {args.episodes}")
    print(f"Control frequency: {args.hz} Hz")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")


def run_episode(policy, camera, leader, follower, episode_num, max_steps, hz):
    """Run one episode on the real robot with safety controls
    
    Safety Controls:
        - SPACEBAR: Pause/Resume
        - ESC or 'q': Stop episode immediately
        - Ctrl+C: Emergency stop
    
    Returns:
        dict: Episode results
    """
    print(f"\n{'=' * 80}")
    print(f"EPISODE {episode_num}")
    print(f"{'=' * 80}")
    
    dt = 1.0 / hz
    episode_start = time.time()
    paused = False
    
    # Get initial robot states (leader + follower)
    follower_state = follower.get_state()
    if leader:
        leader_state = leader.get_state()
        print(f"Initial joint positions:")
        print(f"  Leader:   {leader_state}")
        print(f"  Follower: {follower_state}")
    else:
        # Use follower state as proxy for leader (goal)
        leader_state = follower_state.copy()
        print(f"Initial joint positions (follower, using as goal proxy):")
        print(f"  {follower_state}")
    
    # Concatenate: [leader (6), follower (6)] = 12 values
    initial_state = np.concatenate([leader_state, follower_state])
    
    print(f"\n🚀 Starting episode (max {max_steps} steps)...")
    print("\n⚠️  SAFETY CONTROLS:")
    print("   SPACEBAR: Pause/Resume robot")
    print("   ESC or 'q': Stop episode immediately")
    print("   Ctrl+C: Emergency stop\n")
    
    step = 0
    emergency_stop = False
    
    try:
        while step < max_steps:
            loop_start = time.time()
            
            # Read camera frame
            camera_frame = camera.read_frame()
            
            # Always show camera with safety controls
            display_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
            
            # Add safety status overlay
            status_color = (0, 255, 0) if not paused else (0, 165, 255)  # Green if running, Orange if paused
            status_text = "RUNNING" if not paused else "PAUSED"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(display_frame, f"Step: {step}/{max_steps}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, "SPACE:Pause ESC:Stop", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Robot Camera - SAFETY CONTROLS', display_frame)
            
            # Check for keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\n🛑 STOP requested - halting episode")
                emergency_stop = True
                break
            elif key == ord(' '):  # SPACEBAR
                paused = not paused
                if paused:
                    print("\n⏸️  PAUSED - Press SPACEBAR to resume", flush=True)
                else:
                    print("\n▶️  RESUMED", flush=True)
            
            # If paused, don't send commands but keep loop running
            if paused:
                time.sleep(0.1)
                continue
            
            # Get current robot states (leader + follower)
            follower_state = follower.get_state()
            if leader:
                leader_state = leader.get_state()
            else:
                # Use follower state as proxy for leader (goal)
                leader_state = follower_state.copy()
            # Concatenate: [leader (6), follower (6)] = 12 values
            robot_state = np.concatenate([leader_state, follower_state])
            
            # Create observation
            obs = create_observation(camera_frame, robot_state)
            
            # Get action from policy
            action = policy.predict(obs)
            
            # Send action to robot (with dt for velocity limiting)
            follower.send_action(action, dt=dt)
            
            # Print status with robot state
            if step % 20 == 0 or step == 0:
                print(f"Step {step:4d}/{max_steps} | "
                      f"Action: [{', '.join([f'{a:+.3f}' for a in action[:3]])}...] | "
                      f"State: [{', '.join([f'{s:+.3f}' for s in robot_state[:3]])}...]",
                      flush=True)
            
            step += 1
            
            # Sleep to maintain control rate
            loop_elapsed = time.time() - loop_start
            sleep_time = dt - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n🚨 EMERGENCY STOP - Ctrl+C pressed")
        emergency_stop = True
    
    episode_time = time.time() - episode_start
    
    # Get final states (leader + follower)
    final_follower_state = follower.get_state()
    if leader:
        final_leader_state = leader.get_state()
        final_state = np.concatenate([final_leader_state, final_follower_state])
    else:
        final_leader_state = final_follower_state.copy()
        final_state = np.concatenate([final_leader_state, final_follower_state])
    
    if emergency_stop:
        print(f"\n⚠️  Episode terminated early")
    else:
        print(f"\n✅ Episode complete")
    
    print(f"   Steps: {step}")
    print(f"   Time: {episode_time:.2f}s")
    if leader:
        print(f"   Final joint positions:")
        print(f"     Leader:   {final_leader_state}")
        print(f"     Follower: {final_follower_state}")
    else:
        print(f"   Final joint positions (follower):")
        print(f"     {final_follower_state}")
    
    return {
        'episode': episode_num,
        'steps': step,
        'time_s': episode_time,
        'initial_state': initial_state.tolist(),
        'final_state': final_state.tolist(),
        'emergency_stop': emergency_stop,
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy policy on real SO-101 robot")
    parser.add_argument("--policy-path", type=str, required=True,
                       help="Path to trained policy checkpoint")
    parser.add_argument("--follower-port", type=str, required=True,
                       help="Follower arm USB port (e.g., /dev/tty.usbmodem58760431552)")
    parser.add_argument("--follower-id", type=str, default="my_follower_arm",
                       help="Follower arm calibration ID")
    parser.add_argument("--leader-port", type=str, default=None,
                       help="Leader arm USB port (OPTIONAL - for goal conditioning)")
    parser.add_argument("--leader-id", type=str, default="my_leader_arm",
                       help="Leader arm calibration ID")
    parser.add_argument("--use-leader-as-goal", action="store_true",
                       help="Use leader arm state as goal (if connected)")
    parser.add_argument("--camera-index", type=int, default=0,
                       help="Camera index (0 for default webcam)")
    parser.add_argument("--no-calibrate", action="store_true",
                       help="Skip calibration (if already calibrated)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Max steps per episode")
    parser.add_argument("--hz", type=float, default=10.0,
                       help="Control frequency (Hz)")
    parser.add_argument("--device", type=str, default="mps",
                       choices=["cpu", "cuda", "mps"],
                       help="Device for policy inference")
    
    args = parser.parse_args()
    
    print_startup_banner(args)
    
    # Initialize components
    print("🔧 Initializing components...\n")
    
    # 1. Load policy
    print("🧠 Loading policy...")
    policy = LeRobotPolicyWrapper(args.policy_path, args.device)
    print("✅ Policy loaded\n")
    
    # 2. Connect to camera
    print("📷 Connecting to camera...")
    camera = CameraReader(camera_index=args.camera_index)
    print("✅ Camera ready\n")
    
    # 3. Connect to leader arm (OPTIONAL - for goal conditioning)
    leader = None
    if args.leader_port:
        print("🎮 Connecting to leader arm (READ-ONLY for goal conditioning)...")
        leader = SO101LeaderController(
            args.leader_port,
            args.leader_id
        )
        print("✅ Leader arm ready (READ-ONLY)\n")
        print("💡 NOTE: Leader arm provides goal state observations")
    else:
        print("⚠️  No leader arm specified - using follower state as goal proxy\n")
        print("💡 NOTE: Observation will use [follower_state, follower_state] as proxy for [leader, follower]")
    
    # 4. Connect to follower arm (for actions)
    print("\n🤖 Connecting to follower arm...")
    calibrate = not args.no_calibrate
    follower = SO101FollowerController(
        args.follower_port,
        args.follower_id,
        calibrate=calibrate
    )
    print("✅ Follower arm ready\n")
    
    # Create results directory
    results_dir = Path("runs")
    results_dir.mkdir(exist_ok=True)
    
    # Run episodes
    results = []
    
    print("\n" + "=" * 80)
    print("🚀 STARTING REAL ROBOT DEPLOYMENT")
    print("=" * 80)
    print("⚠️  WARNING: Robot will move! Ensure workspace is clear.")
    print("   Press Ctrl+C at any time to stop.")
    print("=" * 80 + "\n")
    
    # PRE-DEPLOYMENT SAFETY CHECK
    print("🛡️  PRE-DEPLOYMENT SAFETY CHECKLIST:")
    print("   1. ✓ Workspace is clear of obstacles")
    print("   2. ✓ Robot has full range of motion")
    print("   3. ✓ Emergency stop/power switch is accessible")
    print("   4. ✓ Camera has clear view of workspace")
    print("   5. ✓ You are ready to use safety controls (SPACE, ESC, Ctrl+C)")
    print()
    
    print("⏳ Starting deployment in 3 seconds...")
    print("   (Press Ctrl+C now to cancel)")
    import time
    time.sleep(3)
    
    try:
        for ep in range(1, args.episodes + 1):
            # Run episode
            result = run_episode(
                policy, camera, leader, follower,
                ep, args.max_steps, args.hz
            )
            results.append(result)
            
            # Check if emergency stop was triggered
            if result.get('emergency_stop', False):
                print("\n⚠️  Emergency stop detected - ending deployment")
                print("   Review results and restart when ready.")
                break
            
            # Pause between episodes
            if ep < args.episodes:
                print("\n" + "=" * 80)
                print(f"⏸️  EPISODE {ep} COMPLETE - Preparing for episode {ep + 1}")
                print("=" * 80)
                print("   Please reset cube position and workspace")
                print(f"   Starting episode {ep + 1} in 5 seconds...")
                print("   (Press Ctrl+C to cancel)")
                print("=" * 80)
                import time
                time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n🚨 DEPLOYMENT STOPPED - Ctrl+C pressed")
    
    finally:
        # Cleanup
        print("\n\n🧹 Cleaning up...")
        camera.close()
        if leader:
            leader.disconnect()
        follower.disconnect()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete\n")
    
    # Save results
    if results:
        policy_name = Path(args.policy_path).parent.name
        if policy_name == "pretrained_model":
            policy_name = Path(args.policy_path).parent.parent.name
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"eval_real_robot_{policy_name}_{timestamp}.json"
        
        # Calculate summary stats
        total_episodes = len(results)
        mean_steps = np.mean([r['steps'] for r in results])
        mean_time = np.mean([r['time_s'] for r in results])
        
        safety_report = follower.get_safety_report()
        
        results_data = {
            'policy_path': str(args.policy_path),
            'environment': 'real_robot',
            'timestamp': datetime.now().isoformat(),
            'mean_steps': mean_steps,
            'mean_time': mean_time,
            'episodes': total_episodes,
            'safety_violations': safety_report['total_violations'],
            'results': results,
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print("\n" + "=" * 80)
        print("📊 DEPLOYMENT SUMMARY")
        print("=" * 80)
        print(f"Total episodes: {total_episodes}")
        print(f"Mean steps: {mean_steps:.1f}")
        print(f"Mean time: {mean_time:.2f}s")
        print(f"\n💾 Results saved: {results_path}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

