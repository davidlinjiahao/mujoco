"""MuJoCo utility functions for scene management and physics helpers"""

import numpy as np
import mujoco
from pathlib import Path


def reset_cube_position(model, data, position=[0.4, 0.18, 0.42]):
    """
    Reset cube to starting position.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        position: Target position [x, y, z] in meters (default for grounded scene)
    
    Returns:
        bool: True if successful, False if cube not found
    """
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        print("⚠️  Cube body not found in model")
        return False
    
    # Find cube's joint
    cube_joint_id = -1
    for i in range(model.njnt):
        if model.jnt_bodyid[i] == cube_id:
            cube_joint_id = i
            break
    
    if cube_joint_id < 0:
        print("⚠️  Cube joint not found")
        return False
    
    # Get qpos address for this joint
    cube_qpos_idx = model.jnt_qposadr[cube_joint_id]
    
    # Set position (freejoint has 7 DOF: 3 pos + 4 quat)
    data.qpos[cube_qpos_idx:cube_qpos_idx+3] = position
    # Set orientation (quaternion: w, x, y, z) - identity rotation
    data.qpos[cube_qpos_idx+3:cube_qpos_idx+7] = [1, 0, 0, 0]
    
    # Get qvel address and zero velocities (freejoint has 6 DOF velocity)
    cube_qvel_idx = model.jnt_dofadr[cube_joint_id]
    data.qvel[cube_qvel_idx:cube_qvel_idx+6] = 0
    
    # Forward kinematics to update state
    mujoco.mj_forward(model, data)
    return True


def randomize_cube_position(model, data, base_position=[0.4, 0.18, 0.42], max_offset_inches=3.0):
    """
    Randomize cube position within a few inches of the base position.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        base_position: Original cube position [x, y, z] in meters (default for grounded scene)
        max_offset_inches: Maximum random offset in inches (default 3.0)
    
    Returns:
        bool: True if successful, False if failed
    """
    # Convert inches to meters (1 inch = 0.0254 meters)
    max_offset_m = max_offset_inches * 0.0254
    
    # Generate random offsets in x, y (not z to keep it on table)
    random_offset = np.random.uniform(-max_offset_m, max_offset_m, size=3)
    random_offset[2] = 0  # Keep Z (height) constant
    
    # Calculate new position
    new_position = np.array(base_position) + random_offset
    
    # Reset cube to new position
    return reset_cube_position(model, data, new_position)


def check_gripper_contact_forces(model, data, debug=False):
    """
    Check contact forces between gripper and cube.
    Returns dict with contact info for logging.
    """
    contact_info = {
        'num_contacts': 0,
        'total_normal_force': 0.0,
        'max_normal_force': 0.0,
        'contacts': []
    }
    
    # Get cube geom ID
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    
    # Iterate through all active contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        # Check if this contact involves the cube
        if cube_geom_id in (geom1, geom2):
            # Get the other geom (gripper part)
            other_geom = geom2 if geom1 == cube_geom_id else geom1
            other_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            
            # Only log gripper contacts (ignore table/floor)
            if other_geom_name and ('jaw' in other_geom_name.lower() or 'gripper' in other_geom_name.lower()):
                # Get contact force magnitude (normal force)
                # Contact forces are in data.contact[i].frame (contact frame)
                # We need the force from the constraint solver
                normal_force = 0.0
                
                # Find corresponding constraint force
                # Contact forces are stored after joint constraints
                for j in range(data.nefc):
                    if data.efc_type[j] == mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS or \
                       data.efc_type[j] == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL:
                        # This is a simplified approach - actual force calculation is complex
                        # For now, use penetration depth as proxy for force
                        pass
                
                # Use contact distance (penetration) as proxy
                penetration = -contact.dist if contact.dist < 0 else 0.0
                # Rough estimate: force = penetration * contact_stiffness
                # From solimp parameters, stiffness is related to dmin
                estimated_force = penetration * 1000.0  # Rough scaling
                
                contact_info['num_contacts'] += 1
                contact_info['total_normal_force'] += estimated_force
                contact_info['max_normal_force'] = max(contact_info['max_normal_force'], estimated_force)
                
                if debug:
                    contact_info['contacts'].append({
                        'geom': other_geom_name,
                        'penetration': penetration,
                        'force_estimate': estimated_force
                    })
    
    return contact_info


def merge_xml_files(xml_files, output_path="scenes/_merged_teleoperate.xml"):
    """Merge multiple MJCF XML files - save in scenes/ so relative paths to assets/ work"""
    # Save in scenes/ directory so relative paths to assets/ work correctly
    output_path = Path(output_path)
    
    includes = []
    for xml_file in xml_files:
        xml_path = Path(xml_file)
        # Just use the filename since all files are in scenes/
        includes.append(f'  <include file="{xml_path.name}"/>')
    
    includes_str = "\n".join(includes)
    
    merged_xml = f"""<mujoco>
{includes_str}
</mujoco>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(merged_xml)
    
    return str(output_path)


# ============================================================================
# Bin Geometry and Goal Conditioning Utilities
# ============================================================================

# Bin geometry - UPDATED for grounded_scene.xml mesh bin
# Mesh bin position: pos="0.495330 -0.193083 0.420000" (from grounded_scene.xml)
# Approximate dimensions based on mesh scan:
# - Interior: ~15cm x 15cm (slightly smaller than old bin)
# - Height: ~8cm walls

BIN_CENTER = np.array([0.495330, -0.193083, 0.420000], dtype=np.float32)

# Interior bounds (accounting for wall thickness and mesh dimensions)
# Using slightly smaller interior to account for mesh bin walls
BIN_INTERIOR = {
    'x_min': 0.420,   # ~0.495 - 0.075
    'x_max': 0.570,   # ~0.495 + 0.075
    'y_min': -0.268,  # ~-0.193 - 0.075
    'y_max': -0.118,  # ~-0.193 + 0.075
    'z_min': 0.425,   # 0.42 + 0.005 (bottom thickness)
    'z_max': 0.50,    # 0.42 + 0.08 (reasonable height above bottom)
}

# Size of bin interior (for visualization/debugging)
BIN_INTERIOR_SIZE = np.array([
    BIN_INTERIOR['x_max'] - BIN_INTERIOR['x_min'],
    BIN_INTERIOR['y_max'] - BIN_INTERIOR['y_min'],
    BIN_INTERIOR['z_max'] - BIN_INTERIOR['z_min'],
], dtype=np.float32)


def is_cube_in_bin(cube_position, tolerance=0.0):
    """
    Check if cube is inside the bin interior.
    
    Args:
        cube_position: [x, y, z] position of cube center
        tolerance: Extra margin (in meters) to allow near-bin positions
    
    Returns:
        bool: True if cube is in bin
    """
    x, y, z = cube_position
    
    return (BIN_INTERIOR['x_min'] - tolerance <= x <= BIN_INTERIOR['x_max'] + tolerance and
            BIN_INTERIOR['y_min'] - tolerance <= y <= BIN_INTERIOR['y_max'] + tolerance and
            BIN_INTERIOR['z_min'] - tolerance <= z <= BIN_INTERIOR['z_max'] + tolerance)


def get_cube_to_bin_vector(cube_position):
    """
    Get vector from cube to bin center.
    This gives the model directional information about the goal.
    
    Args:
        cube_position: [x, y, z] position of cube center
    
    Returns:
        np.array: [dx, dy, dz] vector pointing toward bin center
    """
    return BIN_CENTER - np.array(cube_position, dtype=np.float32)


def get_distance_to_bin(cube_position):
    """
    Get Euclidean distance from cube to bin center.
    
    Args:
        cube_position: [x, y, z] position of cube center
    
    Returns:
        float: Distance in meters
    """
    return np.linalg.norm(get_cube_to_bin_vector(cube_position))


def get_normalized_bin_distance(cube_position, max_distance=0.5):
    """
    Get normalized distance to bin (0 = at bin, 1 = very far).
    
    Args:
        cube_position: [x, y, z] position of cube center
        max_distance: Maximum expected distance (for normalization)
    
    Returns:
        float: Normalized distance [0, 1]
    """
    distance = get_distance_to_bin(cube_position)
    return np.clip(distance / max_distance, 0.0, 1.0)


def get_bin_alignment_score(cube_position):
    """
    Get how well cube is aligned with bin in each axis (0-1 per axis).
    1.0 = perfectly aligned, 0.0 = outside bin bounds
    
    Args:
        cube_position: [x, y, z] position of cube center
    
    Returns:
        np.array: [x_score, y_score, z_score] alignment scores
    """
    x, y, z = cube_position
    
    # X alignment
    x_center = (BIN_INTERIOR['x_min'] + BIN_INTERIOR['x_max']) / 2
    x_range = (BIN_INTERIOR['x_max'] - BIN_INTERIOR['x_min']) / 2
    x_score = 1.0 - np.clip(abs(x - x_center) / x_range, 0.0, 1.0)
    
    # Y alignment
    y_center = (BIN_INTERIOR['y_min'] + BIN_INTERIOR['y_max']) / 2
    y_range = (BIN_INTERIOR['y_max'] - BIN_INTERIOR['y_min']) / 2
    y_score = 1.0 - np.clip(abs(y - y_center) / y_range, 0.0, 1.0)
    
    # Z alignment
    z_center = (BIN_INTERIOR['z_min'] + BIN_INTERIOR['z_max']) / 2
    z_range = (BIN_INTERIOR['z_max'] - BIN_INTERIOR['z_min']) / 2
    z_score = 1.0 - np.clip(abs(z - z_center) / z_range, 0.0, 1.0)
    
    return np.array([x_score, y_score, z_score], dtype=np.float32)


def compute_reward(cube_position, in_bin=None):
    """
    Compute dense reward signal for cube position.
    Can be used for RL or as auxiliary loss in BC.
    
    Args:
        cube_position: [x, y, z] position of cube center
        in_bin: Optional bool, if None will be computed
    
    Returns:
        float: Reward value (higher is better)
    """
    if in_bin is None:
        in_bin = is_cube_in_bin(cube_position)
    
    if in_bin:
        # High reward for being in bin
        return 10.0
    else:
        # Dense reward based on distance
        distance = get_distance_to_bin(cube_position)
        # Reward decreases with distance: 0 at 0.5m, 5 at bin edge
        reward = max(0.0, 5.0 - distance * 10.0)
        return reward


def get_goal_conditioned_state(joint_positions, cube_position):
    """
    Create goal-conditioned state representation.
    
    Args:
        joint_positions: [6] joint angles
        cube_position: [3] cube xyz position
    
    Returns:
        np.array: [12] state with goal conditioning
                  [joints(6), cube_pos(3), cube_to_bin(3)]
    """
    cube_to_bin = get_cube_to_bin_vector(cube_position)
    return np.concatenate([joint_positions, cube_position, cube_to_bin]).astype(np.float32)

