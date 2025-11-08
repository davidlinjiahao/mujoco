#!/usr/bin/env python3
"""
Leader Arm → MuJoCo Simulation Bridge

Reads joint positions from a physical SO-101 leader arm using LeRobot,
and sends them to MuJoCo simulation for teleoperation.

Usage:
    python3 scripts/teleoperate.py \
        --leader-port /dev/tty.usbmodem58760431551 \
        --leader-id my_leader_arm \
        --render \
        --record

This enables recording demonstrations with a physical leader arm that can be
used for behavior cloning training.
"""

import sys
import time
import argparse
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

# Import our modules from utilities
from utilities.mujoco_utils import (
    reset_cube_position,
    randomize_cube_position,
    merge_xml_files,
    check_gripper_contact_forces
)
from utilities.leader_arm import LeaderArmReader
from utilities.recording import RecordingControls, TrajectoryRecorder

# Import control panel (optional)
try:
    from utilities.control_panel import RecordingControlPanel
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Video recording
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def handle_control_buttons(controls, recorder, model, data):
    """Handle button presses from the web control panel"""
    if controls.start_recording:
        controls.start_recording = False
        recorder.start()
    
    if controls.stop_recording:
        controls.stop_recording = False
        recorder.stop()
    
    if controls.reset_cube:
        controls.reset_cube = False
        if reset_cube_position(model, data):
            print("\n🔄 Cube position reset!")
    
    if controls.randomize_cube:
        controls.randomize_cube = False
        if randomize_cube_position(model, data, max_offset_inches=3.0):
            print("\n🎲 Cube position randomized!")
    
    if controls.save_recording:
        controls.save_recording = False
        handle_save_recording(recorder)
    
    if controls.discard_recording:
        controls.discard_recording = False
        handle_discard_recording(recorder)
    
    if controls.replay_episode:
        controls.replay_episode = False
        handle_replay_request(controls, recorder, model, data)


def handle_save_recording(recorder):
    """Save the current recording - CRASHES PROGRAM IF SAVE FAILS"""
    if recorder.trajectory:
        # Save as next episode number (episode_count will be incremented inside save())
        try:
            saved_path = recorder.save(episode_num=recorder.episode_count + 1)
            if saved_path:
                print(f"\n✅ Episode {recorder.episode_count} saved: {saved_path}")
                recorder.last_saved_episode = saved_path
            recorder.trajectory = []  # Clear after save
            recorder.video_frames = []  # Clear video frames after save
        except Exception as e:
            print("\n" + "="*80)
            print("❌ CRITICAL ERROR: FAILED TO SAVE EPISODE")
            print("="*80)
            print(f"Episode {recorder.episode_count + 1} could not be saved!")
            print(f"Error: {e}")
            print("\nSTOPPING PROGRAM TO PREVENT DATA LOSS")
            print("="*80 + "\n")
            raise RuntimeError(f"Failed to save episode {recorder.episode_count + 1}: {e}") from e
    else:
        print("\n⚠️  No trajectory data to save")


def handle_discard_recording(recorder):
    """Discard the current recording"""
    if recorder.trajectory:
        print(f"\n🗑️  Episode {recorder.episode_count + 1} discarded")
        recorder.trajectory = []  # Clear trajectory
        recorder.video_frames = []  # Clear video frames too
    else:
        print("\n⚠️  No trajectory data to discard")


def handle_replay_request(controls, recorder, model, data):
    """Handle replay episode request"""
    # First, try to replay current unsaved trajectory
    if recorder.trajectory:
        # Reset cube position first
        if reset_cube_position(model, data):
            print("\n🔄 Cube reset for replay")
        
        # Convert current trajectory to replay format
        controls.replay_trajectory = []
        for frame in recorder.trajectory:
            controls.replay_trajectory.append({
                'time': frame['time'],
                'joints': frame['joints'],
                'cube_pos': frame.get('cube_pos', [0, 0, 0])
            })
        
        controls.replay_frame_idx = 0
        controls.replaying = True
        
        print(f"\n🎬 Replaying: Current unsaved episode")
        print(f"   Frames: {len(controls.replay_trajectory)}")
        print(f"   Duration: {controls.replay_trajectory[-1]['time']:.1f}s")
    
    # Otherwise, try to replay last saved episode
    elif recorder.last_saved_episode:
        # Reset cube position first
        if reset_cube_position(model, data):
            print("\n🔄 Cube reset for replay")
        
        # Load the trajectory
        import yaml
        with open(recorder.last_saved_episode, 'r') as f:
            traj_data = yaml.safe_load(f)
        controls.replay_trajectory = traj_data['trajectory']
        controls.replay_frame_idx = 0
        controls.replaying = True
        
        print(f"\n🎬 Replaying: {Path(recorder.last_saved_episode).name}")
        print(f"   Frames: {len(controls.replay_trajectory)}")
        print(f"   Duration: {controls.replay_trajectory[-1]['time']:.1f}s")
    
    else:
        print("\n⚠️  No episode to replay (record something first)")


def get_joint_positions(controls, leader, frame_count):
    """Get joint positions - either from replay or from leader arm"""
    if controls and controls.replaying:
        # Replay mode - use recorded joint positions
        if controls.replay_frame_idx < len(controls.replay_trajectory):
            frame = controls.replay_trajectory[controls.replay_frame_idx]
            joint_positions = np.array(frame['joints'])
            controls.replay_frame_idx += 1
            return joint_positions, False  # False = not finished
        else:
            # Replay finished
            controls.replaying = False
            controls.replay_trajectory = None
            controls.replay_frame_idx = 0
            print("\n✅ Replay finished!")
            return None, True  # True = replay finished, skip this iteration
    else:
        # Normal mode - read leader arm joint positions
        try:
            # Show debug output for first 3 frames
            debug = (frame_count < 3)
            joint_positions = leader.read_joint_positions(debug=debug)
            return joint_positions, False
        except Exception as e:
            print(f"\n❌ Error reading leader arm: {e}")
            raise


def print_status_message(frame_count, joint_positions, recorder, contact_info):
    """Print periodic status updates"""
    if frame_count % 90 == 0:  # Every 3 seconds at 30Hz
        status = f"Frame {frame_count:5d} | Joints: " + ", ".join([f"{j:+.2f}" for j in joint_positions])
        if recorder and recorder.recording:
            status += f" | Recording: {len(recorder.trajectory)} frames"
        
        # Add contact force info to status
        if contact_info['num_contacts'] > 0:
            status += f" | Grip: {contact_info['num_contacts']} contacts, {contact_info['total_normal_force']:.2f}N total"
        else:
            status += " | Grip: No contact"
        
        print(status)


def setup_viewer(model, data):
    """Create and configure the MuJoCo viewer"""
    viewer_handle = mujoco.viewer.launch_passive(model, data)
    
    # Set default viewer perspective (nice angled view of workspace)
    viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer_handle.cam.distance = 1.2  # Distance from lookat point
    viewer_handle.cam.azimuth = 135   # Horizontal rotation (degrees)
    viewer_handle.cam.elevation = -25  # Vertical angle (degrees)
    viewer_handle.cam.lookat[:] = [0.4, 0.0, 0.45]  # Look at workspace center
    
    print("🖥️  Official MuJoCo Viewer opened\n")
    print("   Camera controls:")
    print("     - Mouse drag: Rotate view")
    print("     - Mouse scroll: Zoom in/out")
    print("     - Right-click drag: Pan view")
    print("     - Tab: Cycle camera modes\n")
    
    return viewer_handle


def print_startup_banner(args):
    """Print startup information"""
    print("\n" + "="*80)
    print("LEADER ARM → MUJOCO SIMULATION BRIDGE")
    print("="*80)
    print(f"Leader port: {args.leader_port}")
    print(f"Leader ID: {args.leader_id}")
    print(f"Control frequency: {args.hz} Hz")
    print(f"Recording: {'Enabled' if args.record else 'Disabled'}")
    print("="*80 + "\n")


def print_teleoperation_banner(args, recorder):
    """Print teleoperation active banner"""
    print("="*80)
    print("🎮 TELEOPERATION ACTIVE")
    print("="*80)
    print("Move the leader arm → MuJoCo robot follows!")
    if args.record:
        print("\n📹 RECORDING MODE ENABLED")
        print(f"   Trajectories save to: {recorder.output_dir}/")
        if FLASK_AVAILABLE:
            print("\n   🌐 Web Control Panel is running!")
            print("      Open the URL above in your browser")
        print("\n   Available controls:")
        print("     • ▶️ Start Recording - Begin recording episode")
        print("     • ⏹️ Stop Recording - End recording episode")
        print("     • 🔄 Reset Cube - Reset cube to start position")
        print("     • 💾 Save Episode - Save recorded trajectory")
        print("     • 🗑️ Discard Episode - Delete current recording")
        print("\n   💡 Workflow:")
        print("      1. Position robot at start")
        print("      2. Click 'Start Recording' button")
        print("      3. Perform pick-and-place demonstration")
        print("      4. Click 'Stop Recording' button")
        print("      5. Click 'Save Episode' or 'Discard Episode'")
        print("      6. Click 'Reset Cube' for next episode")
        print("      7. Repeat!")
    else:
        print("\nPress Ctrl+C to quit")
        print("(To record episodes, restart with: --record flag)")
    print("="*80 + "\n")


def handle_shutdown(recorder, control_panel, leader, viewer_handle):
    """Cleanup resources on shutdown"""
    print("\n🛑 Shutting down...")
    
    # Stop control panel
    if control_panel:
        control_panel.stop()
    
    # If still recording when quit, offer to save
    if recorder and recorder.recording:
        recorder.stop()
        if recorder.trajectory:
            response = input(f"\n💾 Save Episode {recorder.episode_count + 1}? (y/n): ").strip().lower()
            if response == 'y' or response == 'yes':
                recorder.save(episode_num=recorder.episode_count + 1)
                print("✅ Recording saved!")
            else:
                print("🗑️  Recording discarded")
        
    if recorder:
        print(f"\n📊 Session summary: {recorder.episode_count} episodes recorded")
        
        # Export LeRobot dataset to local folder
        if recorder.save_lerobot and recorder.lerobot_dataset is not None:
            try:
                print("📦 Exporting LeRobot dataset to local folder...")
                # Note: LeRobot v2.1 (0.3.3) saves separate episode files, no finalize() needed
                recorder.export_all_to_local()
            except Exception as e:
                print(f"⚠️  Error exporting dataset: {e}")
    
    # Disconnect leader arm with error handling
    try:
        if leader:
            leader.disconnect()
    except Exception as e:
        print(f"⚠️  Error during leader disconnect: {e}")
    
    # Close viewer
    try:
        if viewer_handle:
            viewer_handle.close()
    except Exception as e:
        print(f"⚠️  Error closing viewer: {e}")
    
    print("✅ Cleanup complete\n")


def main():
    parser = argparse.ArgumentParser(description="Leader arm to MuJoCo bridge")
    parser.add_argument("--leader-port", required=True, help="Leader arm USB port (e.g., /dev/tty.usbmodem58760431551)")
    parser.add_argument("--leader-id", default="my_leader_arm", help="Leader arm calibration ID")
    parser.add_argument("--scene", default=None, help="Complete scene XML (e.g., scenes/grounded_scene.xml)")
    parser.add_argument("--arm-xml", default="scenes/so_arm.xml", help="Robot model XML (if not using --scene)")
    parser.add_argument("--scene-xmls", nargs="+", 
                       default=["scenes/table.xml", "scenes/cube.xml", "scenes/bin.xml", "scenes/camera_c920.xml"],
                       help="Scene XML files (if not using --scene)")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--record", action="store_true", help="Enable trajectory recording")
    parser.add_argument("--hz", type=float, default=30.0, help="Control frequency (Hz)")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration on connect (if already calibrated)")
    
    args = parser.parse_args()
    
    print_startup_banner(args)
    
    # Connect to leader arm
    calibrate = not args.no_calibrate  # Calibrate unless --no-calibrate is used
    leader = LeaderArmReader(args.leader_port, args.leader_id, calibrate=calibrate)
    
    # Load MuJoCo scene  
    print("🔧 Loading MuJoCo scene...")
    if args.scene:
        # Use complete scene file directly
        scene_path = args.scene
        print(f"   Using complete scene: {scene_path}")
        model = mujoco.MjModel.from_xml_path(scene_path)
    else:
        # Use legacy merge approach
        merged_xml = merge_xml_files([args.arm_xml] + args.scene_xmls)
        model = mujoco.MjModel.from_xml_path(merged_xml)
    
    data = mujoco.MjData(model)
    print(f"✅ MuJoCo loaded (nq={model.nq})")
    print(f"   Gravity: {model.opt.gravity}")
    print(f"   Timestep: {model.opt.timestep}\n")
    
    # Initialize recorder and control panel
    recorder = None
    controls = None
    control_panel = None
    
    if args.record:
        # Get repo_id from environment variable or use None (legacy mode)
        repo_id = os.getenv('DATASET_NAME')
        
        # Use overhead c920 camera with LeRobot integration
        recorder = TrajectoryRecorder(
            output_dir="trajectories",
            model=model,
            data=data,
            camera_name="c920",
            repo_id=repo_id,  # Enable LeRobot if DATASET_NAME is set
            fps=int(args.hz),
            robot_type="so101"
        )
        controls = RecordingControls()
        
        if FLASK_AVAILABLE:
            control_panel = RecordingControlPanel(recorder, controls, model, data)
            control_panel.start()
            time.sleep(0.5)  # Give GUI time to initialize
        
        if not CV2_AVAILABLE:
            print("⚠️  OpenCV not installed - video recording disabled")
            print("   Install with: pip install opencv-python")
    
    # Control loop parameters
    dt = 1.0 / args.hz
    
    # Open viewer if requested (official MuJoCo viewer)
    viewer_handle = None
    if args.render:
        viewer_handle = setup_viewer(model, data)
    
    print_teleoperation_banner(args, recorder)
    
    frame_count = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Handle control panel button presses
            if controls:
                handle_control_buttons(controls, recorder, model, data)
            
            # Get joint positions (from replay or leader arm)
            joint_positions, should_skip = get_joint_positions(controls, leader, frame_count)
            if should_skip:
                continue
            
            # Command the robot actuators (proper physics-based control)
            data.ctrl[:6] = joint_positions
            
            # Run multiple physics steps to advance simulation by dt
            # At 30Hz control, dt=0.033s, timestep=0.002s -> need ~17 steps
            n_steps = int(dt / model.opt.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            
            # Update viewer
            if viewer_handle:
                viewer_handle.sync()
            
            # Record frame
            if recorder and recorder.recording:
                cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
                cube_pos = data.xpos[cube_body_id] if cube_body_id >= 0 else None
                recorder.add_frame(joint_positions, cube_pos)
            
            # Check gripper contact forces (for debugging grip issues)
            contact_info = check_gripper_contact_forces(model, data, debug=False)
            
            # Print status occasionally
            frame_count += 1
            print_status_message(frame_count, joint_positions, recorder, contact_info)
            
            # Sleep to maintain control rate
            loop_elapsed = time.time() - loop_start
            sleep_time = dt - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    finally:
        handle_shutdown(recorder, control_panel, leader, viewer_handle)


if __name__ == "__main__":
    main()
