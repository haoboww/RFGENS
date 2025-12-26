# RF Genesis - Enhanced Version

> **Note**: This is an enhanced version of the original [RF-Genesis](https://github.com/Asixa/RF-Genesis) repository, with added support for custom joint data and related tools.

### [Original Project Page](https://rfgen.xingyuchen.me/) | [Original Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf) 

This repository is based on the original RF-Genesis implementation by [Xingyu Chen](https://xingyuchen.me/) and [Xinyu Zhang](http://xyzhang.ucsd.edu/index.html) (UC San Diego, SenSys 2023).

![teaser](https://rfgen.xingyuchen.me/RFGen/pull.png)

---

## 🆕 Key Enhancements

Compared to the original version, this enhanced version adds the following features:

### Core Features

1. **Custom Joint Data Support** ⭐
   - Direct support for external joint data files (MMBody, COCO, OpenPose, custom formats)
   - New `--joint-file` and `--joint-order` parameters
   - Generate radar data without relying on MDM text prompts

2. **Motion Expansion Tool** ⭐
   - `tools/advanced_motion_expander.py`: Expand sparse joint data into continuous motion sequences
   - Improves radar simulation quality by ensuring motion continuity (important for Doppler velocity calculation)

3. **Point Cloud Conversion Tool** ⭐
   - `tools/convert_radar_to_pointcloud.py`: Fixed version of radar point cloud conversion tool
   - Fixed Doppler velocity calculation issues
   - Supports multiple normalization methods

4. **Batch Processing Scripts**
   - `tools/bash_my_run.sh`: Batch generation of radar data
   - `tools/bash_my_convert.sh`: Batch point cloud conversion

---

## 📁 Project Structure

```
RFGENS/
├── genesis/              # Core RF-Genesis modules (original)
├── models/               # Radar configuration files
├── ext/                  # External dependencies (MDM, etc.)
├── tools/                 # 🆕 Tool scripts directory
│   ├── convert_radar_to_pointcloud.py
│   ├── advanced_motion_expander.py
│   ├── bash_my_run.sh
│   ├── bash_my_convert.sh
│   └── README.md          # Detailed tool documentation
├── my_data/               # Custom joint data
├── output/                # Generated radar frames
└── run.py                 # Main entry point (enhanced)
```

---

## 🚀 Quick Start

### Requirements

- Python 3.10
- conda3 or miniconda3
- CUDA-capable GPU

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RFGENS

# Create conda environment
conda create -n rfgen python=3.10 -y 
conda activate rfgen

# Install dependencies
pip install -r requirements.txt
sh setup.sh
```

### Usage

#### Method 1: Using Text Prompts (Original Method)
```bash
python run.py -o "a person walking back and forth" -e "a living room" -n "hello_rfgen"
```

#### Method 2: Using Custom Joint Data (Enhanced Feature) ⭐
```bash
python run.py --joint-file "./my_data/joint.npy" --joint-order custom --no-environment true --name "my_custom_motion"
```

**Supported joint formats:**
- `default`: Default RF-Genesis format
- `coco`: COCO pose format
- `openpose`: OpenPose format
- `custom`: Custom 22-joint format

---

## 🛠️ Complete Workflow Example

```bash
# Step 1: Expand sparse motion data (recommended for better quality)
python tools/advanced_motion_expander.py \
    -i ./my_data/sparse_joints.npy \
    -o ./my_data/expanded_joints.npy \
    -e 11

# Step 2: Generate radar frames
python run.py \
    --joint-file ./my_data/expanded_joints.npy \
    --joint-order custom \
    --name my_sequence \
    --no-environment true

# Step 3: Convert to point clouds
python tools/convert_radar_to_pointcloud.py \
    --radar_frames ./output/my_sequence/radar_frames.npy \
    --output_dir output_pointclouds/my_sequence/radar \
    --simple_normalize
```

---

## 📚 Documentation

- **Tool Documentation**: See `tools/README.md` for details
- **Motion Expansion Guide**: See `tools/README_advanced_motion.md`
- **Original RF-Genesis**: Refer to the [original repository](https://github.com/Asixa/RF-Genesis)

---

## ⚙️ New Command Line Options

```bash
--joint-file PATH          # Use custom joint data file
--joint-order FORMAT       # Joint format: default/coco/openpose/custom
--no-visualize            # Skip visualization (faster)
--no-environment          # Skip environment generation
```

---

## 📊 Output Structure

```
output/
└── my_sequence/
    ├── obj_diff.npz          # SMPL parameters
    ├── radar_frames.npy      # Radar signal data (N, 3, 4, 128, 256)
    └── output.mp4            # Visualization video (if enabled)

output_pointclouds/
└── my_sequence/
    └── radar/
        ├── frame_1.npy       # Point cloud (N, 5): [x, y, z, velocity, intensity]
        └── ...
```

---

## 📡 Radar Hardware

The current simulation is based on the model of [**Texas Instruments AWR 1843**](https://www.ti.com/product/AWR1843#all) radar, with 3TX 4RX MIMO setup.

The radar configuration can be found in `models/TI1843_config.json` and can be freely adjusted.

---

## 📝 Citation

If you use this code, please cite the original RF-Genesis paper:

```bibtex
@inproceedings{chen2023rfgenesis,
    author = {Chen, Xingyu and Zhang, Xinyu},
    title = {RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models},
    booktitle = {ACM Conference on Embedded Networked Sensor Systems (SenSys '23)},
    year = {2023},
    pages = {1-14},
    address = {Istanbul, Turkiye},
    publisher = {ACM, New York, NY, USA},
    url = {https://doi.org/10.1145/3625687.3625798},
    doi = {10.1145/3625687.3625798}
}
```

---

## 📄 License

This code is distributed under an [MIT LICENSE](LICENSE).

**Note**: This code depends on other libraries, including [CLIP](https://github.com/openai/CLIP), [SMPL](https://smpl.is.tue.mpg.de/), [MDM](https://guytevet.github.io/mdm-page/), and [mmMesh](https://github.com/HavocFiXer/mmMesh), each with their own licenses that must be followed.

---

## 🙏 Acknowledgments

- **Original Authors**: [Xingyu Chen](https://xingyuchen.me/) and [Xinyu Zhang](http://xyzhang.ucsd.edu/index.html) for the excellent RF-Genesis framework
- **Original Repository**: [Asixa/RF-Genesis](https://github.com/Asixa/RF-Genesis)

---

## 🔗 Related Links

- [Original RF-Genesis Project Page](https://rfgen.xingyuchen.me/)
- [Original Paper](https://xingyuchen.me/files/Xingyu.Chen_SenSys23_RFGen.pdf)
- [Original GitHub Repository](https://github.com/Asixa/RF-Genesis)

