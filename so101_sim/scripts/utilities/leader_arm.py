"""Leader arm interface for SO-101 via LeRobot"""

import time
import numpy as np

# Try to import LeRobot
try:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    print("⚠️  LeRobot not installed or not importable")
    print("   Install with: pip install lerobot")
    LEROBOT_AVAILABLE = False


class LeaderArmReader:
    """Read joint positions from SO-101 leader arm via LeRobot"""
    
    def __init__(self, port, robot_id, calibrate=False):
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is required but not installed")
        
        self.port = port
        self.robot_id = robot_id
        self.robot = None
        
        # Joint ranges for SO-101 in radians (from so_arm.xml)
        # Format: (min, max) for each joint
        self.joint_ranges = {
            "shoulder_pan": (-1.91986, 1.91986),
            "shoulder_lift": (-1.74533, 1.74533),
            "elbow_flex": (-1.69, 1.69),
            "wrist_flex": (-1.65806, 1.65806),
            "wrist_roll": (-2.74385, 2.84121),
            "gripper": (-0.17453, 1.74533),
        }
        
        print(f"📡 Connecting to leader arm...")
        print(f"   Port: {port}")
        print(f"   ID: {robot_id}")
        print(f"   Calibrate: {calibrate}")
        
        # Create LeRobot leader arm instance
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                config = SO101LeaderConfig(
                    port=port,
                    use_degrees=False  # Returns normalized values [-100, 100] for body joints
                )
                config.id = robot_id
                self.robot = SO101Leader(config)
                self.robot.connect(calibrate=calibrate)
                print(f"✅ Leader arm connected!")
                break  # Success!
                
            except Exception as e:
                # Aggressive cleanup on failure
                if self.robot:
                    print(f"🔧 Attempting to clean up failed connection...")
                    try:
                        # Force close the serial port
                        if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'port_handler'):
                            if hasattr(self.robot.bus.port_handler, 'ser'):
                                if self.robot.bus.port_handler.ser:
                                    try:
                                        self.robot.bus.port_handler.ser.close()
                                        print("   └─ Serial port closed")
                                    except:
                                        pass
                    except:
                        pass
                    
                    # Try normal disconnect too
                    try:
                        self.robot.disconnect()
                    except:
                        pass
                    
                    time.sleep(retry_delay)
                
                # If this was the last attempt, give up
                if attempt == max_retries - 1:
                    print(f"\n❌ Failed to connect to leader arm after {max_retries} attempts: {e}")
                    print(f"\nTroubleshooting:")
                    print(f"  1. Check if arm is plugged in: ls -l /dev/tty.usb*")
                    print(f"  2. Unplug and replug the arm's USB cable")
                    print(f"  3. Wait 5 seconds after plugging in")
                    print(f"  4. Make sure no other process is using the arm")
                    raise
                else:
                    print(f"   └─ Retry {attempt + 2}/{max_retries} in {retry_delay}s...")
                    time.sleep(retry_delay)
    
    def _normalized_to_radians(self, normalized_value, joint_range, is_gripper=False):
        """Convert normalized value to actual joint angle in radians
        
        LeRobot uses different normalization modes:
        - Body joints: RANGE_M100_100 returns [-100, 100]
        - Gripper: RANGE_0_100 returns [0, 100]
        
        Args:
            normalized_value: Value in range [-100, 100] or [0, 100]
            joint_range: Tuple of (min_rad, max_rad)
            is_gripper: True if this is the gripper joint (uses 0-100 range)
        
        Returns:
            Joint angle in radians
        """
        min_rad, max_rad = joint_range
        
        if is_gripper:
            # Gripper uses RANGE_0_100: Map [0, 100] to [min_rad, max_rad]
            # 0 -> min_rad (closed), 100 -> max_rad (open)
            return (normalized_value / 100.0) * (max_rad - min_rad) + min_rad
        else:
            # Body joints use RANGE_M100_100: Map [-100, 100] to [min_rad, max_rad]
            # -100 -> min_rad, +100 -> max_rad
            return ((normalized_value + 100) / 200.0) * (max_rad - min_rad) + min_rad
    
    def read_joint_positions(self, debug=False):
        """Read current joint positions from leader arm
        
        Returns:
            np.ndarray: 6D joint positions in radians
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected")
        
        # Read action from leader arm
        # Body joints use RANGE_M100_100 (returns -100 to 100)
        # Gripper uses RANGE_0_100 (returns 0 to 100)
        action = self.robot.get_action()
        
        # Extract and convert positions to radians
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        joint_pos = []
        for name in joint_names:
            is_gripper = (name == "gripper")
            converted = self._normalized_to_radians(action[f"{name}.pos"], self.joint_ranges[name], is_gripper)
            joint_pos.append(converted)
        
        joint_pos = np.array(joint_pos)
        
        if debug:
            print(f"\nDEBUG - Raw values from leader:")
            for name in joint_names:
                raw = action[f"{name}.pos"]
                is_gripper = (name == "gripper")
                converted = self._normalized_to_radians(raw, self.joint_ranges[name], is_gripper)
                range_str = "[0, 100]" if is_gripper else "[-100, 100]"
                print(f"  {name:15s}: {raw:+7.2f} {range_str} -> {converted:+7.4f} rad ({np.degrees(converted):+7.2f}°)")
        
        return joint_pos
    
    def disconnect(self):
        """Disconnect from leader arm and cleanup serial port"""
        if self.robot:
            try:
                # Try normal disconnect first
                self.robot.disconnect()
                print("✅ Leader arm disconnected")
            except Exception as e:
                print(f"⚠️  Error during disconnect: {e}")
            
            # Force cleanup of serial port
            try:
                if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'port_handler'):
                    if hasattr(self.robot.bus.port_handler, 'ser'):
                        # Close the underlying serial connection
                        if self.robot.bus.port_handler.ser and self.robot.bus.port_handler.ser.is_open:
                            self.robot.bus.port_handler.ser.close()
                            print("🔧 Forced serial port closure")
                            time.sleep(0.5)  # Give OS time to release the port
            except Exception as e:
                print(f"⚠️  Error during port cleanup: {e}")

