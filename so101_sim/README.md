# SO-101 MuJoCo Simulation: Base Sim → Grounded Sim → Sim2Real

A MuJoCo simulation environment for the SO-101 robotic arm, designed to test the progression from **base simulation** (simple primitives) → **grounded simulation** (realistic 3D scans) → **sim2real transfer** (deployment on physical robot).

![SO-101 Pick and Place](https://img.shields.io/badge/MuJoCo-Simulation-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![LeRobot](https://img.shields.io/badge/LeRobot-Compatible-success)

## 🎯 Project Goal

Test the hypothesis: **Can we train policies in simulation that transfer to the real world?**

### Three-Stage Pipeline

1. **Base Simulation** → Train on simple geometric primitives (table, cube, bin)
2. **Grounded Simulation** → Test on realistic Polycam 3D scans of the actual workspace
3. **Sim2Real Transfer** → Deploy trained policies on the physical SO-101 robot

## ✅ What Works

- ✅ **Base Simulation**: Simple table/cube/bin environment for rapid iteration
- ✅ **Grounded Simulation**: Polycam-scanned workspace with realistic textures and physics
- ✅ **Leader Arm Teleoperation**: Physical SO-101 leader arm → MuJoCo simulation
- ✅ **LeRobot Integration**: Record demonstrations and train ACT/Diffusion policies
- ✅ **Policy Deployment**: Test trained policies in both base and grounded sims
- ✅ **Camera System**: Logitech C920 camera model with realistic FOV

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate
git clone https://github.com/davidlinjiahao/mujoco.git
cd mujoco/so101_sim

# Install dependencies
pip install -r requirements.txt

# For macOS users
export MUJOCO_GL=glfw

# Login to HuggingFace (for dataset upload)
huggingface-cli login
```

### 1. Test Base Simulation (Optional Scene)

Record demonstrations using simple primitive shapes:

```bash
python3 scripts/teleoperate.py \
    --scene scenes/table.xml \
    --scene scenes/cube.xml \
    --scene scenes/bin.xml \
    --leader-port /dev/tty.usbmodem58760431551 \
    --leader-id my_leader_arm \
    --render \
    --record
```

**Base simulation uses:**
- Simple box geometries for table, cube, and bin
- Fast physics simulation
- Good for rapid policy iteration

### 2. Test Grounded Simulation (Realistic Scene)

Record demonstrations using Polycam-scanned meshes:

```bash
python3 scripts/teleoperate.py \
    --scene scenes/grounded_scene.xml \
    --leader-port /dev/tty.usbmodem58760431551 \
    --leader-id my_leader_arm \
    --render \
    --record
```

**Grounded simulation uses:**
- Real-world 3D scans from Polycam
- Realistic textures and geometry
- Tests if policies generalize to realistic visuals

**Recording Features:**
- 🎮 **Web Control Panel**: Start/stop recording via browser (http://127.0.0.1:5001)
- 📹 **Video Recording**: Automatic MP4 capture from simulated C920 camera
- 🔄 **Episode Management**: Save, discard, or replay episodes
- 🎲 **Cube Randomization**: Randomize cube position for variation
- 💾 **LeRobot Format**: Saves directly to HuggingFace LeRobotDataset format

### 3. Train Policies with LeRobot

Train on your recorded demonstrations:

```bash
# Set your HuggingFace username
HF_USER=$(huggingface-cli whoami | head -n 1)

# Train ACT policy (recommended starting point)
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_sim \
    --policy.type=act \
    --output_dir=outputs/train/act_base_sim \
    --policy.device=mps \
    --wandb.enable=true

# Train Diffusion policy
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_sim \
    --policy.type=diffusion \
    --output_dir=outputs/train/diffusion_base_sim \
    --policy.device=mps
```

**Training time:** ~2-4 hours on M1 Mac for 50k-100k steps

### 4. Deploy and Evaluate

Test trained policies in simulation:

```bash
# Deploy in base simulation
python3 scripts/deploy.py \
    --policy-path outputs/train/act_base_sim/checkpoints/last/pretrained_model \
    --scene scenes/table.xml \
    --scene scenes/cube.xml \
    --scene scenes/bin.xml \
    --episodes 20 \
    --render

# Deploy in grounded simulation (test generalization)
python3 scripts/deploy.py \
    --policy-path outputs/train/act_base_sim/checkpoints/last/pretrained_model \
    --scene scenes/grounded_scene.xml \
    --episodes 20 \
    --render
```

**Key question:** Does a policy trained on simple shapes work on realistic scans?

## 📁 Project Structure

```
so101_sim/
├── scenes/                         # Scene definitions
│   ├── so_arm.xml                  # SO-101 robot model
│   ├── assets/                     # Robot STL mesh files
│   ├── table.xml                   # Base sim: simple table
│   ├── cube.xml                    # Base sim: simple cube
│   ├── bin.xml                     # Base sim: simple bin
│   ├── camera_c920.xml             # Logitech C920 camera
│   ├── grounded_scene.xml          # Grounded sim: realistic scanned scene
│   └── scans/                      # Polycam 3D scans
│       ├── room.obj                # Scanned room environment
│       ├── table.obj               # Scanned table
│       ├── bin.obj                 # Scanned bin (split for hollow collision)
│       ├── cube.obj                # Scanned cube
│       └── *_textures/             # PNG textures for meshes
├── scripts/
│   ├── teleoperate.py              # 🎮 Leader arm teleoperation + recording
│   ├── deploy.py                   # 🤖 Deploy trained policies
│   └── utilities/
│       ├── leader_arm.py           # Leader arm communication
│       ├── recording.py            # LeRobot dataset recording
│       ├── control_panel.py        # Web control interface
│       └── mujoco_utils.py         # MuJoCo helper functions
├── outputs/                        # Training outputs
│   └── train/
│       └── */checkpoints/          # Trained policy checkpoints
├── trajectories/                   # Recorded demonstrations
│   └── lerobot_batch_*/            # LeRobot dataset episodes
└── requirements.txt                # Python dependencies
```

## 🎓 Recommended Workflow

### Phase 1: Base Sim Training (Establish Baseline)

**Goal:** Verify the robot can learn the task in an ideal environment.

1. **Record 50-100 demonstrations** in base simulation:
   ```bash
   python3 scripts/teleoperate.py \
       --scene scenes/table.xml \
       --scene scenes/cube.xml \
       --scene scenes/bin.xml \
       --leader-port /dev/tty.usbmodem58760431551 \
       --render --record
   ```

2. **Train ACT policy:**
   ```bash
   lerobot-train \
       --dataset.repo_id=${HF_USER}/so101_base_sim \
       --policy.type=act \
       --output_dir=outputs/train/act_base_sim
   ```

3. **Evaluate in base sim:**
   ```bash
   python3 scripts/deploy.py \
       --policy-path outputs/train/act_base_sim/checkpoints/last/pretrained_model \
       --scene scenes/table.xml --scene scenes/cube.xml --scene scenes/bin.xml \
       --episodes 20
   ```

**Success criteria:** >70% success rate in base simulation

### Phase 2: Grounded Sim Testing (Test Visual Generalization)

**Goal:** Test if policies trained on simple shapes work with realistic scans.

1. **Deploy base-sim-trained policy in grounded sim:**
   ```bash
   python3 scripts/deploy.py \
       --policy-path outputs/train/act_base_sim/checkpoints/last/pretrained_model \
       --scene scenes/grounded_scene.xml \
       --episodes 20 --render
   ```

2. **If it fails, record demonstrations in grounded sim:**
   ```bash
   python3 scripts/teleoperate.py \
       --scene scenes/grounded_scene.xml \
       --leader-port /dev/tty.usbmodem58760431551 \
       --render --record
   ```

3. **Train on grounded sim data:**
   ```bash
   lerobot-train \
       --dataset.repo_id=${HF_USER}/so101_grounded_sim \
       --policy.type=act \
       --output_dir=outputs/train/act_grounded_sim
   ```

**Key insight:** If base-sim policies don't transfer to grounded sim, sim2real is unlikely to work.

### Phase 3: Sim2Real Transfer (Future Work)

**Goal:** Deploy policies on physical SO-101 follower robot.

**Requirements:**
- Physical SO-101 follower robot
- Intel RealSense D455 camera (match C920 FOV)
- Real-world table, cube, and bin matching simulation

**Process:**
1. Match camera position/FOV to simulation
2. Deploy sim-trained policy on real robot
3. Measure success rate
4. If needed, fine-tune with real-world demonstrations

## 🔧 Configuration

### Camera Setup (Logitech C920)

```xml
<!-- scenes/camera_c920.xml -->
<camera name="c920"
        fovy="43"                    # Vertical FOV (matches real C920)
        pos="0.35 -0.40 0.87"       # Position above workspace
        euler="-25 0 0"/>            # Tilt down 25° toward table
```

**Specifications:**
- Resolution: 640×480 (default) or 640×360
- FOV: 43° vertical (78° horizontal)
- Frame rate: 30 FPS

### Scene Selection

**Base simulation scenes** (simple primitives):
```bash
--scene scenes/table.xml \
--scene scenes/cube.xml \
--scene scenes/bin.xml
```

**Grounded simulation scene** (realistic scans):
```bash
--scene scenes/grounded_scene.xml
```

The grounded scene includes all components (robot, camera, scanned objects).

### Action/Observation Space

**Observations:**
- `observation.state` (12D): `[joints (6) + cube_pos (3) + cube_to_bin (3)]`
- `observation.images.front` (640×480×3): RGB from C920 camera

**Actions:**
- `action` (6D): Target joint positions in radians

## 🏗️ Importing 3D Scans into MuJoCo

This section documents the complete process for importing Polycam 3D scans (or any .obj meshes) into MuJoCo with proper positioning, orientation, and textures. This was used to create `grounded_scene.xml` with realistic scanned objects.

### Prerequisites

```bash
pip install pillow numpy  # For texture conversion and mesh analysis
```

**What you'll need:**
- `.obj` mesh files from Polycam (or any 3D scanner)
- `.mtl` material files with texture references
- Texture image files (usually in separate folders)
- Scale information for each object

### Step 1: Position Alignment

#### Challenge
MuJoCo uses a **Z-up coordinate system**, but most 3D tools (Unity, Blender, Polycam) export **Y-up** meshes. You need to:
1. Convert coordinate systems
2. Align objects to specific heights (grounding)

#### Solution: Measure and Align

**A. Measure mesh bounding boxes:**

```python
def measure_obj(obj_path):
    """Parse OBJ file and get bounding box dimensions"""
    vertices = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    return {
        'xmin': vertices[:, 0].min(), 'xmax': vertices[:, 0].max(),
        'ymin': vertices[:, 1].min(), 'ymax': vertices[:, 1].max(),
        'zmin': vertices[:, 2].min(), 'zmax': vertices[:, 2].max(),
    }
```

**B. Apply grounding rules:**

For a pick-and-place task, we need precise vertical alignment:

```python
# Example: Align table top to robot base height (z=0.42)
table_dims = measure_obj('scenes/scans/table.obj')
table_height = table_dims['zmax'] - table_dims['zmin']
table_scale = 1.5  # From Unity export or manual measurement

# Calculate body position to place table top at z=0.42
desired_table_top_z = 0.42
body_z = desired_table_top_z - (table_height * table_scale)
```

**Our alignment rules:**
- Room floor → `z = 0.0` (ground plane)
- Table top → `z = 0.42` (robot base height)
- Bin bottom → `z = 0.42` (sits on table)
- Cube bottom → `z = 0.42` (sits on table)

**C. Position objects in XML:**

```xml
<body name="table" pos="0.4 0 -0.53" euler="1.570796 0 0">
  <geom type="mesh" mesh="table_scan" material="table_material"/>
</body>
```

### Step 2: Orientation Alignment

#### Challenge
All meshes appear **sideways** when loaded into MuJoCo because:
- OBJ files from Polycam are Y-up oriented
- MuJoCo expects Z-up oriented meshes
- A rotation is needed to convert between coordinate systems

#### Solution: Visual Rotation Test

**A. Create a test file** (`scenes/test_rotations.xml`):

```xml
<mujoco>
  <compiler angle="radian" meshdir="scans"/>
  <asset>
    <mesh name="test" file="table.obj" scale="1.5 1.5 1.5"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    
    <!-- Test different rotations -->
    <body name="no_rotation" pos="0 0 0.5">
      <geom type="mesh" mesh="test" rgba="1 0 0 1"/>  <!-- Red -->
    </body>
    
    <body name="plus_90_x" pos="0.5 0 0.5" euler="1.5708 0 0">
      <geom type="mesh" mesh="test" rgba="0 0 1 1"/>  <!-- Blue -->
    </body>
    
    <body name="minus_90_x" pos="1.0 0 0.5" euler="-1.5708 0 0">
      <geom type="mesh" mesh="test" rgba="0 1 0 1"/>  <!-- Green -->
    </body>
    
    <body name="plus_90_y" pos="1.5 0 0.5" euler="0 1.5708 0">
      <geom type="mesh" mesh="test" rgba="1 1 0 1"/>  <!-- Yellow -->
    </body>
    
    <body name="minus_90_y" pos="2.0 0 0.5" euler="0 -1.5708 0">
      <geom type="mesh" mesh="test" rgba="1 0 1 1"/>  <!-- Magenta -->
    </body>
    
    <body name="plus_90_z" pos="2.5 0 0.5" euler="0 0 1.5708">
      <geom type="mesh" mesh="test" rgba="0 1 1 1"/>  <!-- Cyan -->
    </body>
  </worldbody>
</mujoco>
```

**B. View and identify the correct orientation:**

```bash
mjpython -c "import mujoco; import mujoco.viewer; \
  model = mujoco.MjModel.from_xml_path('scenes/test_rotations.xml'); \
  mujoco.viewer.launch_passive(model, mujoco.MjData(model))"
```

Identify which colored mesh appears upright. For Polycam Y-up meshes, **+90° X rotation (blue)** is typically correct.

**C. Apply the base rotation to all meshes:**

```xml
<!-- All scanned objects need euler="1.570796 0 0" (+90° around X) -->
<body name="room" pos="0.15 0.0 1.02" euler="1.570796 0.000000 0.000000">
  <geom type="mesh" mesh="room_scan" material="room_material"/>
</body>

<body name="table" pos="0.4 0 -0.53" euler="1.570796 0.000000 0.000000">
  <geom type="mesh" mesh="table_scan" material="table_material"/>
</body>
```

**Key insight:** `euler="1.5708 0 0"` = +90° around X-axis = Y-up → Z-up conversion

### Step 3: Creating Hollow Collision Geometry

#### Challenge
MuJoCo treats mesh collisions as **convex hulls**, meaning any mesh becomes a solid object. For a bin, this creates a solid block instead of a hollow container.

#### Solution: Split Mesh into Separate Pieces

**A. Analyze the mesh to find corners:**

```python
import numpy as np

def analyze_bin_mesh(obj_path):
    """Find the actual visual boundaries of the mesh"""
    vertices = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    
    # Find bottom corners (lowest z values)
    bottom_z = vertices[:, 2].min()
    bottom_verts = vertices[vertices[:, 2] < bottom_z + 0.01]  # Within 1cm of bottom
    
    # Get extreme X and Y coordinates
    x_min, x_max = bottom_verts[:, 0].min(), bottom_verts[:, 0].max()
    y_min, y_max = bottom_verts[:, 1].min(), bottom_verts[:, 1].max()
    
    print(f"Bin dimensions: {x_max - x_min:.3f}m × {y_max - y_min:.3f}m")
    print(f"Corner positions: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
```

**B. Split the mesh into 5 pieces (base + 4 walls):**

```python
def split_bin_mesh(input_obj, output_prefix):
    """Split bin.obj into 5 separate collision pieces"""
    vertices = []
    faces = []
    
    # Read mesh
    with open(input_obj) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                faces.append(line)
    
    vertices = np.array(vertices)
    
    # Define bounds for each piece (with overlap margin)
    bounds = measure_obj(input_obj)
    margin = 0.10  # 10cm overlap for continuous walls
    
    pieces = {
        'base': lambda v: v[2] < bounds['zmin'] + margin,
        'wall_x_neg': lambda v: v[0] < bounds['xmin'] + margin,
        'wall_x_pos': lambda v: v[0] > bounds['xmax'] - margin,
        'wall_z_neg': lambda v: v[1] < bounds['ymin'] + margin,
        'wall_z_pos': lambda v: v[1] > bounds['ymax'] - margin,
    }
    
    # Write separate OBJ files for each piece
    for name, condition in pieces.items():
        vertex_mask = np.array([condition(v) for v in vertices])
        # ... write filtered faces to f"{output_prefix}_{name}.obj"
```

**C. Create dual-layer structure in MuJoCo:**

```xml
<body name="bin_scan_hollow" pos="0.495 -0.193 0.42" euler="1.5708 0 0">
  <!-- VISUAL: Complete mesh (textured, no collision) -->
  <geom name="bin_visual" type="mesh" mesh="bin_scan" material="bin_material"
        contype="0" conaffinity="0"/>
  
  <!-- COLLISION: 5 invisible pieces (hollow structure) -->
  <geom name="bin_base_collision" type="mesh" mesh="bin_base" 
        contype="1" conaffinity="1" rgba="0 0 0 0" mass="0.2"/>
  <geom name="bin_wall_x_neg_collision" type="mesh" mesh="bin_wall_x_neg" 
        contype="1" conaffinity="1" rgba="0 0 0 0" mass="0.1"/>
  <geom name="bin_wall_x_pos_collision" type="mesh" mesh="bin_wall_x_pos" 
        contype="1" conaffinity="1" rgba="0 0 0 0" mass="0.1"/>
  <geom name="bin_wall_z_neg_collision" type="mesh" mesh="bin_wall_z_neg" 
        contype="1" conaffinity="1" rgba="0 0 0 0" mass="0.1"/>
  <geom name="bin_wall_z_pos_collision" type="mesh" mesh="bin_wall_z_pos" 
        contype="1" conaffinity="1" rgba="0 0 0 0" mass="0.1"/>
</body>
```

**Result:** Visually seamless bin that physically behaves as hollow.

### Step 4: Applying Textures

#### Challenge
MuJoCo **ignores .mtl files** and doesn't automatically load textures referenced in OBJ files. Everything appears gray without explicit texture definitions.

#### Solution: Extract and Convert Textures

**A. Find texture paths in .mtl files:**

```python
import re

def extract_texture_from_mtl(mtl_path):
    """Extract diffuse texture path from .mtl file"""
    with open(mtl_path) as f:
        content = f.read()
    
    # Find diffuse texture (map_Kd)
    match = re.search(r'map_Kd\s+(.+\.jpg)', content)
    if match:
        return match.group(1).strip()
    return None

# Example
texture_path = extract_texture_from_mtl('scenes/scans/room.mtl')
# Returns: 'room_textures/e5a31366fc076fe2a9c445947a1716c5.jpg'
```

**B. Convert JPEG to PNG** (MuJoCo requirement):

```python
from PIL import Image
import os

def convert_textures_to_png(scan_dir):
    """Convert all JPEG textures to PNG for MuJoCo"""
    texture_folders = ['room_textures', 'table_textures', 'bin_textures', 'cube_textures']
    
    for folder in texture_folders:
        folder_path = os.path.join(scan_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                jpg_path = os.path.join(folder_path, filename)
                png_path = jpg_path.replace('.jpg', '.png')
                
                img = Image.open(jpg_path)
                img.save(png_path, 'PNG')
                print(f"✓ Converted {filename} → {filename.replace('.jpg', '.png')}")

convert_textures_to_png('scenes/scans')
```

**C. Add explicit texture definitions in MuJoCo XML:**

```xml
<asset>
  <!-- Meshes -->
  <mesh name="room_scan" file="room.obj"/>
  <mesh name="table_scan" file="table.obj" scale="1.5 1.5 1.5"/>
  
  <!-- Textures (converted to PNG) -->
  <texture name="room_texture" type="2d" file="scans/room_textures/...png"/>
  <texture name="table_texture" type="2d" file="scans/table_textures/...png"/>
  
  <!-- Materials linking textures to meshes -->
  <material name="room_material" texture="room_texture"/>
  <material name="table_material" texture="table_texture"/>
</asset>

<worldbody>
  <!-- Apply materials to geoms -->
  <body name="room" pos="..." euler="...">
    <geom type="mesh" mesh="room_scan" material="room_material"/>
  </body>
  
  <body name="table" pos="..." euler="...">
    <geom type="mesh" mesh="table_scan" material="table_material"/>
  </body>
</worldbody>
```

**Important notes:**
- Texture paths are relative to the XML file location (not `meshdir`)
- MuJoCo requires PNG format for textures
- Each geom needs `material="..."` attribute to display textures

### Complete Example

See `scenes/grounded_scene.xml` for a full working example with:
- ✅ Properly oriented meshes (+90° X rotation)
- ✅ Precise z-alignment (table at robot base height)
- ✅ Hollow bin collision (split mesh pieces)
- ✅ Polycam textures (PNG format with explicit materials)

### Quick Reference

**Coordinate conversion:**
```
Y-up to Z-up: euler="1.5708 0 0"  (+90° around X)
```

**Testing orientation:**
```bash
mjpython -c "import mujoco; import mujoco.viewer; \
  model = mujoco.MjModel.from_xml_path('scenes/test_rotations.xml'); \
  mujoco.viewer.launch_passive(model, mujoco.MjData(model))"
```

**Texture format:**
```
MuJoCo requires: PNG files
Polycam exports: JPEG files
Solution: Convert with PIL (pillow)
```

## 🐛 Troubleshooting

### Leader Arm Connection Issues

```bash
# Check available ports
ls -l /dev/tty.usb*

# If connection fails, unplug and replug USB
# Wait 5 seconds before reconnecting
```

### Camera Rendering Issues

```bash
# Try different GL backends
export MUJOCO_GL=glfw   # macOS recommended
# or
export MUJOCO_GL=egl    # Linux with GPU
# or
export MUJOCO_GL=osmesa # CPU fallback
```

### Mesh Loading Errors

```bash
# Error: "No such file or directory" for meshes
# Solution: Check that meshdir is set correctly in XML
<compiler angle="radian" meshdir="scans"/>

# Error: Meshes appear gray (no textures)
# Solution: Convert JPEG → PNG and add explicit texture/material definitions
```

### Training Not Improving

**Common issues:**
1. **Insufficient data**: Record more episodes (aim for 50-100+)
2. **Inconsistent demonstrations**: Review videos, ensure smooth motions
3. **Cube not visible**: Check camera view includes workspace
4. **No variation**: Add randomized cube positions
5. **Overfitting to base sim**: Test in grounded sim to verify generalization

## 📚 Resources

- **LeRobot Documentation**: https://huggingface.co/docs/lerobot
- **ACT Policy Paper**: https://arxiv.org/abs/2304.13705
- **SO-101 Robot**: https://github.com/TheRobotStudio/SO-ARM100
- **MuJoCo**: https://mujoco.org
- **Polycam**: https://poly.cam

## 🎯 Research Questions

### Sim-to-Sim Transfer
- ✅ Does training on simple shapes transfer to realistic scans?
- ✅ How much does visual realism matter for pick-and-place?
- 🔬 What's the minimum visual fidelity needed for transfer?

### Sim-to-Real Transfer (Future)
- 🔬 Does grounded sim improve sim2real transfer?
- 🔬 How much real-world data is needed to fine-tune sim policies?
- 🔬 Can we use domain randomization to bridge the gap?

## 📝 License

MIT License - Free for research and commercial use.

## 🙏 Acknowledgments

- **HuggingFace Team** for LeRobot library
- **The Robot Studio** for SO-ARM100 robot model
- **DeepMind** for MuJoCo physics engine
- **Polycam** for accessible 3D scanning

---

**Status**: 🔬 **Active Research** → Testing base sim → grounded sim → sim2real pipeline

**Last Updated**: November 7, 2025
