"""Recording and trajectory management for behavior cloning"""

import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Video recording
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Bin geometry for goal conditioning
try:
    from .mujoco_utils import BIN_CENTER, get_cube_to_bin_vector, is_cube_in_bin
    BIN_GEOMETRY_AVAILABLE = True
except ImportError:
    BIN_GEOMETRY_AVAILABLE = False
    # Fallback: mesh bin position from grounded_scene.xml
    BIN_CENTER = np.array([0.495330, -0.193083, 0.420000])

# MuJoCo for video rendering
import mujoco


class RecordingControls:
    """Recording control state for UI buttons"""
    def __init__(self):
        self.start_recording = False
        self.stop_recording = False
        self.reset_cube = False
        self.randomize_cube = False
        self.save_recording = False
        self.discard_recording = False
        self.replay_episode = False
        self.replaying = False
        self.replay_trajectory = None
        self.replay_frame_idx = 0


class TrajectoryRecorder:
    """
    Record trajectories for behavior cloning
    
    Dual-format recorder:
    - Saves to YAML + MP4 (for backward compatibility)
    - Saves to LeRobotDataset (for LeRobot training) if repo_id provided
    """
    
    def __init__(self, output_dir="trajectories", model=None, data=None, camera_name="c920", 
                 repo_id=None, fps=30, robot_type="so101"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.recording = False
        self.trajectory = []
        self.start_time = 0.0
        self.episode_count = self._get_next_episode_number() - 1  # Start at 0, increments on start()
        self.last_saved_episode = None
        
        # Video recording
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.video_writer = None
        self.video_frames = []
        self.video_enabled = CV2_AVAILABLE
        
        # LeRobot dataset integration
        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.lerobot_dataset = None
        self.lerobot_episode_buffer = []
        self.save_lerobot = repo_id is not None
        
        # Check if LeRobot is available
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            import torch
            self.LEROBOT_AVAILABLE = True
        except ImportError:
            self.LEROBOT_AVAILABLE = False
            if self.save_lerobot:
                print("⚠️  LeRobot not installed - LeRobotDataset format disabled")
                print("   Install with: pip install lerobot")
                self.save_lerobot = False
        
        if self.save_lerobot:
            print(f"\n📦 LeRobot recording enabled")
            print(f"   Repo ID: {repo_id}")
            print(f"   FPS: {fps}")
    
    def _get_next_episode_number(self):
        """Find the next episode number by scanning existing files"""
        existing_episodes = list(self.output_dir.glob("episode_*.yaml"))
        if not existing_episodes:
            return 1
        
        # Extract episode numbers from filenames
        episode_numbers = []
        for ep_file in existing_episodes:
            try:
                # episode_001.yaml -> 001 -> 1
                num_str = ep_file.stem.split('_')[1]
                episode_numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        
        return max(episode_numbers) + 1 if episode_numbers else 1
    
    def start(self):
        """Start recording"""
        self.recording = True
        self.trajectory = []
        self.video_frames = []
        self.lerobot_episode_buffer = []
        self.start_time = time.time()
        # Display as "next episode" (will become real episode number only when saved)
        print(f"\n🔴 Recording Episode {self.episode_count + 1} started!")
        if self.video_enabled:
            print(f"   📹 Video recording enabled (camera: {self.camera_name})")
        if self.save_lerobot:
            self._ensure_lerobot_dataset()
    
    def stop(self):
        """Stop recording"""
        self.recording = False
        if self.trajectory:
            print(f"\n⏸️  Episode {self.episode_count + 1} stopped ({len(self.trajectory)} frames, {self.trajectory[-1]['time']:.1f}s)")
        else:
            print(f"\n⏸️  Episode {self.episode_count + 1} stopped (no data recorded)")
    
    def add_frame(self, joint_positions, cube_pos=None):
        """Add a frame to the trajectory and capture video frame"""
        if not self.recording:
            return
        
        timestamp = time.time() - self.start_time
        
        frame = {
            'time': timestamp,
            'joints': joint_positions.copy(),
        }
        
        if cube_pos is not None:
            frame['cube_pos'] = cube_pos.copy()
            # Add goal conditioning: vector from cube to bin
            if BIN_GEOMETRY_AVAILABLE:
                frame['cube_to_bin'] = get_cube_to_bin_vector(cube_pos).tolist()
            else:
                # Fallback if bin_geometry not available
                frame['cube_to_bin'] = (BIN_CENTER - cube_pos).tolist()
        
        self.trajectory.append(frame)
        
        # Capture video frame
        video_frame = None
        if self.video_enabled and self.model and self.data:
            try:
                video_frame = self._capture_video_frame()
            except Exception as e:
                # Don't fail recording if video capture fails
                if len(self.video_frames) == 0:  # Only print once
                    print(f"⚠️  Video capture failed: {e}")
        
        # Build LeRobot frame
        if self.save_lerobot and self.LEROBOT_AVAILABLE:
            import torch
            # State: [joints (6) + cube_pos (3) + cube_to_bin (3)] = 12D
            state = np.concatenate([
                joint_positions,
                cube_pos if cube_pos is not None else np.zeros(3),
                np.array(frame.get('cube_to_bin', [0, 0, 0]))
            ]).astype(np.float32)
            
            lerobot_frame = {
                'observation.state': torch.from_numpy(state),
                'observation.images.front': torch.from_numpy(video_frame) if video_frame is not None else torch.zeros((480, 640, 3), dtype=torch.uint8),
                'task': torch.tensor([0], dtype=torch.int64),  # Task 0: pick and place
                'timestamp': timestamp,
                # Action will be filled in later (next frame's joints)
            }
            
            self.lerobot_episode_buffer.append(lerobot_frame)
    
    def _capture_video_frame(self):
        """Capture a single frame from the MuJoCo camera"""
        # Create renderer if not exists
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Enable all scene visualization options for proper rendering
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            # Print debug info once
            print(f"   📹 Video renderer initialized: 640x480")
        
        # Get camera ID
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        
        if camera_id == -1:
            # List available cameras
            camera_list = []
            for i in range(self.model.ncam):
                cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                camera_list.append(cam_name)
            raise ValueError(f"Camera '{self.camera_name}' not found. Available cameras: {camera_list}")
        
        # Update scene with current physics state
        self._renderer.update_scene(self.data, camera=camera_id)
        
        # Render the frame (RGB uint8)
        frame_rgb = self._renderer.render()
        
        # Check if frame is all black (debugging)
        if len(self.video_frames) == 0:  # First frame
            avg_brightness = frame_rgb.mean()
            if avg_brightness < 1.0:  # Very dark
                print(f"   ⚠️  Warning: First video frame is very dark (brightness: {avg_brightness:.2f})")
                print(f"       Camera: {self.camera_name} (ID: {camera_id})")
        
        # Convert RGB to BGR for OpenCV and add to video frames
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.video_frames.append(frame_bgr)
        
        # Return RGB for LeRobot (expects RGB uint8)
        return frame_rgb
    
    def save(self, filename=None, episode_num=None):
        """Save trajectory - only to LeRobot during recording, local files exported at end"""
        import yaml
        
        if not self.trajectory:
            print("⚠️  No trajectory data to save")
            return None
        
        # If LeRobot recording: ONLY save to LeRobot (fast!), skip local files
        # Local files will be exported in batch at the end via export_all_to_local()
        if self.save_lerobot:
            print(f"\n💾 Episode {self.episode_count + 1}: Saving to LeRobot cache...")
            print(f"   Frames: {len(self.trajectory)}")
            self._save_lerobot(push_to_hub=False)
            
            # Increment saved episode count
            self.episode_count += 1
            
            # Clear buffers
            self.lerobot_episode_buffer = []
            
            return None  # No local file path yet
        
        # Legacy mode: save YAML+MP4 locally (when not using LeRobot)
        save_dir = self.output_dir
        
        if filename is None:
            if episode_num is not None:
                filename = f"episode_{episode_num:03d}.yaml"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"episode_{timestamp}.yaml"
        
        filepath = save_dir / filename
        
        # Convert to serializable format
        traj_data = {
            'metadata': {
                'recorded_at': datetime.now().isoformat(),
                'duration_s': self.trajectory[-1]['time'] if self.trajectory else 0.0,
                'num_samples': len(self.trajectory),
                'device': 'so101_leader_arm'
            },
            'trajectory': []
        }
        
        for frame in self.trajectory:
            traj_data['trajectory'].append({
                'time': float(frame['time']),
                'joints': [float(j) for j in frame['joints']],
                'cube_pos': [float(x) for x in frame.get('cube_pos', [0, 0, 0])],
                'cube_to_bin': [float(x) for x in frame.get('cube_to_bin', [0, 0, 0])]
            })
        
        with open(filepath, 'w') as f:
            yaml.dump(traj_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n💾 Trajectory saved: {filepath}")
        print(f"   Samples: {len(self.trajectory)}")
        print(f"   Duration: {traj_data['metadata']['duration_s']:.2f}s")
        
        # Save video if we have frames
        if self.video_enabled and self.video_frames:
            video_filename = filepath.stem + ".mp4"
            video_path = save_dir / video_filename
            self._save_video(video_path)
        
        # Increment saved episode count
        self.episode_count += 1
        
        # Clear buffers
        self.lerobot_episode_buffer = []
        
        return str(filepath)
    
    def _save_video(self, video_path):
        """Save captured video frames to MP4 file"""
        if not self.video_frames:
            return
        
        try:
            # Get frame dimensions
            height, width = self.video_frames[0].shape[:2]
            
            # Create video writer (30 fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
            
            # Write all frames
            for frame in self.video_frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            # Show relative path if in subfolder
            display_path = video_path.relative_to(self.output_dir) if self.save_lerobot else video_path
            print(f"📹 Video saved: {display_path}")
            print(f"   Frames: {len(self.video_frames)}")
            print(f"   Resolution: {width}x{height}")
        except Exception as e:
            print(f"⚠️  Failed to save video: {e}")
    
    def _ensure_lerobot_dataset(self):
        """Create LeRobotDataset if it doesn't exist"""
        if not self.save_lerobot or not self.LEROBOT_AVAILABLE:
            return
        
        if self.lerobot_dataset is not None:
            return
        
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import shutil
        
        # Check if dataset already exists
        dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / self.repo_id
        
        # Try to load existing dataset
        if dataset_path.exists():
            try:
                print(f"📂 Loading existing LeRobotDataset: {self.repo_id}")
                self.lerobot_dataset = LeRobotDataset(self.repo_id)
                print(f"✅ Loaded! Current episodes: {self.lerobot_dataset.num_episodes}")
                print(f"⚠️  WARNING: Continuing from existing dataset")
                print(f"   If you want to start fresh, delete: {dataset_path}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load existing dataset: {e}")
                print(f"🗑️  Deleting corrupted dataset and creating fresh...")
                try:
                    shutil.rmtree(dataset_path)
                    print(f"✅ Old dataset deleted")
                except Exception as del_error:
                    print(f"❌ Failed to delete old dataset: {del_error}")
                    print(f"   Please manually delete: {dataset_path}")
                    raise RuntimeError(f"Cannot proceed - please manually delete corrupted dataset at {dataset_path}") from del_error
        
        # Create new dataset
        print(f"📂 Creating new LeRobotDataset: {self.repo_id}")
        
        # Define features schema (LeRobot custom format)
        # NOTE: 'task' is NOT included in features - LeRobot adds it automatically!
        # Only include observation and action features here.
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (12,),
                "names": None
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": None
            },
        }
        
        # task will be passed in frame_data to add_frame() but NOT in the features schema
        
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            robot_type=self.robot_type,
        )
        print(f"✅ LeRobotDataset created at: {dataset_path}")
    
    def _save_lerobot(self, push_to_hub=False):
        """Save to LeRobotDataset format"""
        if not self.save_lerobot or not self.LEROBOT_AVAILABLE:
            return
        
        if not self.lerobot_episode_buffer:
            return
        
        import torch
        import pandas as pd
        
        print(f"\n💾 Saving to LeRobotDataset...")
        print(f"   Frames: {len(self.lerobot_episode_buffer)}")
        
        # Fill in actions (next frame's joint positions)
        for i in range(len(self.lerobot_episode_buffer)):
            if i < len(self.lerobot_episode_buffer) - 1:
                # Action = next frame's joint positions
                next_joints = torch.from_numpy(np.array(self.trajectory[i + 1]['joints'], dtype=np.float32))
                self.lerobot_episode_buffer[i]['action'] = next_joints
            else:
                # Last frame: repeat current joints
                current_joints = torch.from_numpy(np.array(self.trajectory[i]['joints'], dtype=np.float32))
                self.lerobot_episode_buffer[i]['action'] = current_joints
        
        # Add episode to dataset (LeRobot 0.3.3 / v2.1 API)
        episode_index_before = self.lerobot_dataset.num_episodes
        
        from PIL import Image as PILImage
        
        try:
            for frame_idx, frame_dict in enumerate(self.lerobot_episode_buffer):
                # Convert torch tensors to numpy arrays / PIL Images (LeRobot requirement!)
                # NOTE: In LeRobot 0.3.3 (v2.1), task is a SEPARATE argument
                frame_data = {
                    'observation.state': frame_dict['observation.state'].cpu().numpy(),  # tensor -> numpy
                    'observation.images.front': PILImage.fromarray(frame_dict['observation.images.front'].cpu().numpy().astype(np.uint8)),  # tensor -> PIL
                    'action': frame_dict['action'].cpu().numpy(),  # tensor -> numpy
                }
                
                # LeRobot 0.3.3 API: add_frame(frame, task) - task is separate!
                task_str = f"task_{int(frame_dict['task'].item()):03d}"  # Convert task ID to string
                self.lerobot_dataset.add_frame(frame_data, task=task_str)
            
            # Save episode (writes complete episode_NNNNNN.parquet file in v2.1)
            self.lerobot_dataset.save_episode()
            
            # VALIDATION: Verify episode was actually saved
            episode_index_after = self.lerobot_dataset.num_episodes
            if episode_index_after != episode_index_before + 1:
                raise RuntimeError(f"Episode count did not increase! Before: {episode_index_before}, After: {episode_index_after}")
            
            print(f"✅ Saved! (Episode {self.episode_count + 1}, {len(self.lerobot_episode_buffer)} frames)")
            print(f"   Total episodes in dataset: {episode_index_after}")
            
        except Exception as e:
            print(f"\n❌ SAVE FAILED!")
            print(f"   Episode: {self.episode_count + 1}")
            print(f"   Frames: {len(self.lerobot_episode_buffer)}")
            print(f"   Error: {e}")
            # Re-raise with context
            raise RuntimeError(f"Failed to save episode {self.episode_count + 1} to LeRobot dataset") from e
        
        # Push to hub if requested
        if push_to_hub:
            print(f"\n📤 Pushing to HuggingFace Hub: {self.repo_id}")
            self.lerobot_dataset.push_to_hub()
            print(f"✅ Dataset available at: https://huggingface.co/datasets/{self.repo_id}")
    
    def export_all_to_local(self):
        """Export all episodes from HF cache to local lerobot_batch_001 folder (v2.1 format)"""
        if not self.save_lerobot or not self.lerobot_dataset:
            return
        
        try:
            import shutil
            
            print(f"\n📦 Exporting all episodes to local folder...")
            
            # Create lerobot_batch_001 subfolder
            lerobot_dir = self.output_dir / "lerobot_batch_001"
            lerobot_dir.mkdir(exist_ok=True)
            
            # Find the HF cache location
            cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / self.repo_id
            
            # v2.1 format: Copy all separate episode parquet files
            data_dir = cache_path / "data" / "chunk-000"
            if data_dir.exists():
                episode_files = sorted(data_dir.glob("episode_*.parquet"))
                total_data_size = 0
                for ep_file in episode_files:
                    dest = lerobot_dir / ep_file.name
                    shutil.copy2(ep_file, dest)
                    total_data_size += dest.stat().st_size
                size_mb = total_data_size / (1024 * 1024)
                print(f"   ✅ Data: {len(episode_files)} episode files ({size_mb:.2f} MB)")
            
            # v2.1 format: Copy all separate episode video files
            video_dir = cache_path / "videos" / "observation.images.front" / "chunk-000"
            if video_dir.exists():
                episode_videos = sorted(video_dir.glob("episode_*.mp4"))
                total_video_size = 0
                for vid_file in episode_videos:
                    dest = lerobot_dir / vid_file.name
                    shutil.copy2(vid_file, dest)
                    total_video_size += dest.stat().st_size
                size_mb = total_video_size / (1024 * 1024)
                print(f"   ✅ Videos: {len(episode_videos)} episode files ({size_mb:.2f} MB)")
            
            # Copy metadata files
            for meta_file in ["info.json", "stats.json", "tasks.parquet"]:
                src = cache_path / "meta" / meta_file
                if src.exists():
                    dest = lerobot_dir / f"meta_{meta_file}"
                    shutil.copy2(src, dest)
                    print(f"   ✅ Meta: {meta_file}")
            
            print(f"\n✅ Export complete! All files in: {lerobot_dir.relative_to(self.output_dir)}")
            print(f"   Total episodes: {self.lerobot_dataset.num_episodes}")
            print(f"   Total frames: {len(self.lerobot_dataset)}")
            
        except Exception as e:
            print(f"⚠️  Export failed: {e}")

