"""Utility modules for SO-101 teleoperation and recording"""

from .mujoco_utils import (
    reset_cube_position,
    randomize_cube_position,
    merge_xml_files,
    check_gripper_contact_forces,
    BIN_CENTER,
    BIN_INTERIOR,
    BIN_INTERIOR_SIZE,
    is_cube_in_bin,
    get_cube_to_bin_vector,
    get_distance_to_bin,
    get_normalized_bin_distance,
    get_bin_alignment_score,
    compute_reward,
    get_goal_conditioned_state,
)

from .leader_arm import LeaderArmReader
from .recording import RecordingControls, TrajectoryRecorder
from .control_panel import RecordingControlPanel

__all__ = [
    # MuJoCo utilities
    "reset_cube_position",
    "randomize_cube_position",
    "merge_xml_files",
    "check_gripper_contact_forces",
    "BIN_CENTER",
    "BIN_INTERIOR",
    "BIN_INTERIOR_SIZE",
    "is_cube_in_bin",
    "get_cube_to_bin_vector",
    "get_distance_to_bin",
    "get_normalized_bin_distance",
    "get_bin_alignment_score",
    "compute_reward",
    "get_goal_conditioned_state",
    # Leader arm
    "LeaderArmReader",
    # Recording
    "RecordingControls",
    "TrajectoryRecorder",
    # Control panel
    "RecordingControlPanel",
]

