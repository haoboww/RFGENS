# Tools Documentation

This directory contains utility scripts and tools for processing RF-Genesis data. These tools are improvements and extensions to the original RF-Genesis repository.

## Overview

The tools in this directory provide:
- **Radar data conversion**: Convert radar frames to point clouds
- **Motion expansion**: Expand sparse joint data into continuous motion sequences
- **Batch processing**: Scripts for processing multiple sequences in parallel

## Tools

### 1. `convert_radar_to_pointcloud.py`

**Purpose**: Convert radar signal frames to 3D point clouds with fixed Doppler velocity calculation.

**Key Features**:
- Fixed Doppler resolution calculation (0.1 m/s)
- Support for SMPL data alignment
- Multiple normalization methods
- Visualization capabilities

**Usage**:
```bash
# Basic usage
python tools/convert_radar_to_pointcloud.py \
    --radar_frames ./output/seq_0/radar_frames.npy \
    --output_dir output_pointclouds/sequence_0/radar

# With simple normalization
python tools/convert_radar_to_pointcloud.py \
    --radar_frames ./output/seq_0/radar_frames.npy \
    --output_dir output_pointclouds/sequence_0/radar \
    --simple_normalize

# With visualization
python tools/convert_radar_to_pointcloud.py \
    --radar_frames ./output/seq_0/radar_frames.npy \
    --output_dir output_pointclouds/sequence_0/radar \
    --visualize
```

**Arguments**:
- `--radar_frames`: Path to radar_frames.npy file
- `--output_dir`: Output directory for point clouds
- `--radar_config`: Path to radar config file (default: ../models/TI1843_config.json)
- `--simple_normalize`: Use simple normalization method
- `--visualize`: Create visualization plots
- `--mmbody_labels`: Path to MMBody labels for pelvis positions (optional)

**Output**:
- Point cloud files: `frame_1.npy`, `frame_2.npy`, ... (each with shape (N, 5))
  - Columns: [x, y, z, velocity, intensity]
- Visualization images (if `--visualize` is used)

---

### 2. `advanced_motion_expander.py`

**Purpose**: Expand sparse joint pose data into continuous motion sequences for better radar simulation.

**Key Features**:
- Intelligent limb movement simulation
- Torso stability control
- Natural motion characteristics
- Configurable expansion factors

**Usage**:
```bash
# Basic usage (expand 5x)
python tools/advanced_motion_expander.py \
    -i input_joints.npy \
    -o output_expanded.npy \
    -e 5

# Advanced usage with custom parameters
python tools/advanced_motion_expander.py \
    -i input_joints.npy \
    -o output_expanded.npy \
    -e 11 \
    -s 0.15 \
    -t 0.98 \
    -l 1.3 \
    -d 1.2
```

**Arguments**:
- `-i, --input`: Input joint data file (.npy, shape: (N, 22, 3))
- `-o, --output`: Output file path (.npy)
- `-e, --expansion`: Expansion factor (must be odd, default: 5)
- `-s, --movement-scale`: Movement amplitude scaling (default: 0.1)
- `-t, --torso-stability`: Torso stability factor 0-1 (default: 0.95)
- `-l, --leg-boost`: Leg movement boost factor (default: 1.3)
- `-d, --distal-boost`: Distal joint boost factor (default: 1.2)

**Why Use This Tool?**:
- Radar simulation requires **continuous motion** for accurate Doppler velocity calculation
- Sparse joint data (e.g., 10fps) can cause artifacts in point clouds
- Expanding to 5-11x provides smooth motion for better radar simulation

**See**: `README_advanced_motion.md` for detailed documentation.

---

### 3. `bash_my_run.sh`

**Purpose**: Batch script for running RF-Genesis simulation on multiple joint data files.

**Usage**:
```bash
# Edit the script to set your joint files and names
bash tools/bash_my_run.sh
```

**Configuration**:
Edit the arrays in the script:
- `joint_files`: Array of input joint file paths
- `names`: Array of output sequence names
- `max_jobs`: Maximum parallel jobs (default: 5)

**Example**:
```bash
joint_files=(
    "./my_data/seq_0.npy"
    "./my_data/seq_1.npy"
)
names=(
    "Seq_0"
    "Seq_1"
)
```

---

### 4. `bash_my_convert.sh`

**Purpose**: Batch script for converting multiple radar frame files to point clouds.

**Usage**:
```bash
# Edit the script to set your radar frames and output directories
bash tools/bash_my_convert.sh
```

**Configuration**:
Edit the arrays in the script:
- `radar_frames`: Array of radar_frames.npy file paths
- `output_dirs`: Array of output directory paths
- `max_jobs`: Maximum parallel jobs (default: 10)

**Example**:
```bash
radar_frames=(
    "./output/Seq_0/radar_frames.npy"
    "./output/Seq_1/radar_frames.npy"
)
output_dirs=(
    "output_pointclouds/sequence_0/radar"
    "output_pointclouds/sequence_1/radar"
)
```

---

## Workflow Example

### Complete Pipeline:

1. **Expand motion data** (if needed):
```bash
python tools/advanced_motion_expander.py \
    -i ./my_data/sparse_joints.npy \
    -o ./my_data/expanded_joints.npy \
    -e 11
```

2. **Generate radar frames**:
```bash
python run.py \
    --joint-file ./my_data/expanded_joints.npy \
    --joint-order custom \
    --name my_sequence \
    --no-environment true
```

3. **Convert to point clouds**:
```bash
python tools/convert_radar_to_pointcloud.py \
    --radar_frames ./output/my_sequence/radar_frames.npy \
    --output_dir output_pointclouds/my_sequence/radar \
    --simple_normalize
```

---

## Notes

- All tools assume they are run from the project root directory
- Paths in scripts are relative to the project root
- The `convert_radar_to_pointcloud.py` tool automatically finds the radar config file from the `models/` directory
- For batch processing, adjust `max_jobs` based on your system's memory and CPU capacity

---

## Improvements Over Original RF-Genesis

These tools include several improvements:
1. **Fixed Doppler resolution calculation** - More accurate velocity estimation
2. **Motion expansion** - Better handling of sparse input data
3. **Batch processing support** - Efficient processing of multiple sequences
4. **Better normalization methods** - Improved point cloud alignment

