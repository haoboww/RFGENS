# Advanced Motion Expander User Guide

## Overview

`advanced_motion_expander.py` is a specialized script designed for advanced motion expansion. It can convert static joint pose data into dynamic, continuous motion sequences. The script simulates realistic human movement characteristics and enables fine-grained limb control.

## Key Features

### 1. Intelligent Limb Movement Control
- **Randomized Arm and Leg Movement Directions**: At the start of each motion sequence, each limb randomly selects a movement direction.
- **Consistent Movement**: Once the direction is determined, the limb moves in the same direction throughout the sequence.
- **Chain-like Joint Movement**: Maintains relative positions between joints to simulate realistic human motion.

### 2. Torso Stability
- **High Stability**: The torso (pelvis, spine, neck, head) remains mostly stationary.
- **Subtle Variations**: Only minimal random changes are added to simulate natural phenomena like breathing.
- **Adjustable Stability**: The stability of the torso can be controlled via parameters.

### 3. Natural Motion Simulation
- **Leg Movement Enhancement**: Leg movement amplitude is greater than that of the arms, simulating real human motion.
- **Distal Joint Enhancement**: Joints farther from the torso move more, in line with human kinematics.
- **Time Progression**: Movement amplitude increases over time, simulating gradual motion.
- **Subtle Randomness**: Adds small random variations to avoid mechanical, perfect movement.

## Body Part Grouping

The script divides the 22 joints into 5 main body parts:

```
(torso): [0, 3, 6, 9, 12, 15]
├── 0: pelvis 
├── 3: spine1 
├── 6: spine2 
├── 9: spine3
├── 12: neck 
└── 15: head 

left_arm: [13, 16, 18, 20]
├── 13: left_collar 
├── 16: left_shoulder
├── 18: left_elbow
└── 20: left_wrist

right_arm: [14, 17, 19, 21]
├── 14: right_collar 
├── 17: right_shoulder
├── 19: right_elbow
└── 21: right_wrist

left_leg [1, 4, 7, 10]
├── 1: left_hip
├── 4: left_knee
├── 7: left_ankle
└── 10: left_foot

right_leg: [2, 5, 8, 11]
├── 2: right_hip
├── 5: right_knee
├── 8: right_ankle
└── 11: right_foot
```

## Usage

### Basic Usage

```bash
python advanced_motion_expander.py -i input.npy -o output.npy -e 8 -s 0.15 -t 0.98
```

### Parameter Description

- `-i, --input`: Input joint data file path (.npy format)
- `-o, --output`: Output file path (.npy format)
- `-e, --expansion`: Expansion factor, how many frames to expand each pose to (default: 5)
- `-s, --movement-scale`: Movement amplitude scaling factor (default: 0.1)
- `-t, --torso-stability`: Torso stability factor 0-1 (default: 0.95)
- `-l, --leg-boost`: Leg movement boost factor (default: 1.3, legs move 30% more than arms)
- `-d, --distal-boost`: Distal joint movement boost factor (default: 1.2, distal joints move 20% more than proximal)

### Parameter Tuning Suggestions

#### Movement Amplitude Scaling Factor (-s)
- **0.05-0.1**: Small movements, suitable for subtle changes in static poses
- **0.1-0.2**: Medium movements, suitable for daily activity simulation
- **0.2-0.3**: Large movements, suitable for dance or sports simulation

#### Torso Stability Factor (-t)
- **0.9-0.95**: Highly stable, suitable for scenarios requiring posture holding
- **0.95-0.98**: Extremely stable, suitable for scenarios requiring a completely still torso
- **0.98-1.0**: Almost completely still

#### Leg Movement Boost Factor (-l)
- **1.0-1.2**: Slight boost, legs move a bit more than arms
- **1.2-1.4**: Medium boost, legs move noticeably more than arms
- **1.4-1.6**: Strong boost, legs move much more than arms

#### Distal Joint Boost Factor (-d)
- **1.0-1.1**: Slight boost, distal joints move a bit more than proximal
- **1.1-1.3**: Medium boost, distal joints move noticeably more than proximal
- **1.3-1.5**: Strong boost, distal joints move much more than proximal


## Output Format

### Input Format
- **File type**: .npy (NumPy array)
- **Data shape**: (N, 22, 3)
  - N: Number of motion frames
  - 22: Number of joints
  - 3: xyz coordinates

### Output Format
- **File type**: .npy (NumPy array)
- **Data shape**: (M*N, 22, 3)
  - M*N: Total number of frames after expansion
  - 22: Number of joints
  - 3: xyz coordinates




