#!/usr/bin/env python3
"""
Advanced Motion Expansion Script
Expand (N, 22, 3) joint motion data to (M*N, 22, 3) continuous motion.
The original frame is placed in the middle, expanding forward and backward.
"""

import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

# Joint name definitions
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

# Body part groupings
BODY_PARTS = {
    'torso': [0, 3, 6, 9, 12, 15],  # Torso: pelvis, spine, neck, head
    'left_arm': [13, 16, 18, 20],    # Left arm: left collar, left shoulder, left elbow, left wrist
    'right_arm': [14, 17, 19, 21],   # Right arm: right collar, right shoulder, right elbow, right wrist
    'left_leg': [1, 4, 7, 10],       # Left leg: left hip, left knee, left ankle, left foot
    'right_leg': [2, 5, 8, 11]       # Right leg: right hip, right knee, right ankle, right foot
}

class AdvancedMotionExpander:
    """Advanced motion expander"""
    
    def __init__(self, movement_scale: float = 0.1, torso_stability: float = 0.95,
                 leg_movement_boost: float = 1.3, distal_joint_boost: float = 1.2):
        """
        Initialize the motion expander
        
        Args:
            movement_scale: Movement amplitude scaling factor
            torso_stability: Torso stability factor (0-1, 1 means completely still)
            leg_movement_boost: Leg movement boost factor (default 1.3, legs move 30% more than arms)
            distal_joint_boost: Distal joint movement boost factor (default 1.2, distal joints move 20% more than proximal)
        """
        self.movement_scale = movement_scale
        self.torso_stability = torso_stability
        self.leg_movement_boost = leg_movement_boost
        self.distal_joint_boost = distal_joint_boost
        
    def expand_motion(self, input_joints: np.ndarray, expansion_factor: int) -> np.ndarray:
        """
        Expand motion data, with the original frame in the middle
        
        Args:
            input_joints: (N, 22, 3) input joint data
            expansion_factor: Expansion factor M (must be odd)
            
        Returns:
            expanded_joints: (M*N, 22, 3) expanded joint data
        """
        if expansion_factor % 2 == 0:
            raise ValueError("Expansion factor must be odd to ensure the original frame is in the middle")
            
        N = input_joints.shape[0]
        expanded_joints = []
        
        for i in range(N):
            current_pose = input_joints[i]
            frames = self._generate_motion_sequence(current_pose, expansion_factor)
            expanded_joints.append(frames)
        
        return np.concatenate(expanded_joints, axis=0)
    
    def _generate_motion_sequence(self, pose: np.ndarray, num_frames: int) -> np.ndarray:
        """
        Generate a motion sequence for a single pose, with the original frame in the middle
        
        Args:
            pose: (22, 3) single pose
            num_frames: number of frames to generate (odd)
            
        Returns:
            frames: (num_frames, 22, 3) motion sequence
        """
        middle_frame = num_frames // 2
        frames = [None] * num_frames
        
        # Place the original frame in the middle
        frames[middle_frame] = pose.copy()
        
        # Generate movement directions for each body part
        movement_directions = self._generate_movement_directions()
        
        # Generate forward frames (from middle to front)
        for frame_idx in range(middle_frame - 1, -1, -1):
            time_factor = (middle_frame - frame_idx) / middle_frame  # 0 to 1
            new_frame = self._generate_next_frame(pose, movement_directions, time_factor, is_forward=True)
            frames[frame_idx] = new_frame
        
        # Generate backward frames (from middle to back)
        for frame_idx in range(middle_frame + 1, num_frames):
            time_factor = (frame_idx - middle_frame) / (num_frames - 1 - middle_frame)  # 0 to 1
            new_frame = self._generate_next_frame(pose, movement_directions, time_factor, is_forward=False)
            frames[frame_idx] = new_frame
        
        return np.array(frames)
    
    def _generate_movement_directions(self) -> Dict[str, np.ndarray]:
        """
        Generate random movement directions for each body part
        
        Returns:
            movement_directions: Dictionary of movement directions for each body part
        """
        directions = {}

        for part in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)  # Normalize
            directions[part] = direction
            
        return directions
    
    def _generate_next_frame(self, base_pose: np.ndarray, 
                           movement_directions: Dict[str, np.ndarray], 
                           time_factor: float, is_forward: bool) -> np.ndarray:
        """
        Generate the next frame
        
        Args:
            base_pose: Base pose
            movement_directions: Movement directions
            time_factor: Time factor (0 to 1)
            is_forward: Whether it is forward expansion
            
        Returns:
            new_frame: New pose frame
        """
        new_frame = base_pose.copy()
        
        # Apply torso stability
        new_frame = self._apply_torso_stability(new_frame, base_pose, time_factor)
        
        # Apply limb movement
        new_frame = self._apply_limb_movement(new_frame, base_pose, movement_directions, time_factor, is_forward)
        
        return new_frame
    
    def _apply_torso_stability(self, new_frame: np.ndarray, base_pose: np.ndarray, 
                              time_factor: float) -> np.ndarray:
        """
        Apply torso stability, keeping the torso mostly still
        
        Args:
            new_frame: New frame
            base_pose: Base pose
            time_factor: Time factor
            
        Returns:
            new_frame: Modified new frame
        """
        # Torso joints remain highly stable
        for joint_idx in BODY_PARTS['torso']:
            # Add very small random variation to simulate breathing or micro-movements
            stability_factor = self.torso_stability
            random_offset = np.random.normal(0, 0.001 * (1 - stability_factor), 3)
            new_frame[joint_idx] = base_pose[joint_idx] + random_offset
            
        return new_frame
    
    def _apply_limb_movement(self, new_frame: np.ndarray, base_pose: np.ndarray,
                            movement_directions: Dict[str, np.ndarray], 
                            time_factor: float, is_forward: bool) -> np.ndarray:
        """
        Apply limb movement
        
        Args:
            new_frame: New frame
            base_pose: Base pose
            movement_directions: Movement directions
            time_factor: Time factor
            is_forward: Whether it is forward expansion
            
        Returns:
            new_frame: Modified new frame
        """
        # Apply movement for each limb
        for part_name, joint_indices in BODY_PARTS.items():
            if part_name == 'torso':
                continue
                
            if part_name in movement_directions:
                direction = movement_directions[part_name]
                
                # Calculate base movement magnitude (increases with time)
                base_movement_magnitude = self.movement_scale * time_factor
                
                # Apply leg boost factor
                if part_name in ['left_leg', 'right_leg']:
                    movement_magnitude = base_movement_magnitude * self.leg_movement_boost
                else:
                    movement_magnitude = base_movement_magnitude
                
                # Use opposite direction for forward and backward
                if not is_forward:
                    direction = -direction
                
                # Apply movement to each joint, maintaining relative positions
                self._apply_chain_movement(new_frame, base_pose, joint_indices, 
                                         direction, movement_magnitude, part_name)
        
        return new_frame
    
    def _apply_chain_movement(self, new_frame: np.ndarray, base_pose: np.ndarray,
                              joint_indices: List[int], direction: np.ndarray,
                              movement_magnitude: float, part_name: str):
        """
        Apply movement to a joint chain, maintaining relative positions
        
        Args:
            new_frame: New frame
            base_pose: Base pose
            joint_indices: List of joint indices
            direction: Movement direction
            movement_magnitude: Movement magnitude
            part_name: Body part name
        """
        if len(joint_indices) < 2:
            return
            
        # The root joint of the chain (closest to torso)
        root_joint = joint_indices[0]
        
        # Apply main movement to the root joint
        new_frame[root_joint] = base_pose[root_joint] + direction * movement_magnitude
        
        # Apply movement to other joints, with distal joints moving more
        for i, joint_idx in enumerate(joint_indices[1:], 1):
            # Distance factor from torso (0 is closest, 1 is farthest)
            distance_factor = i / (len(joint_indices) - 1)
            
            # Apply distal joint boost: farther joints move more
            distal_boost = 1.0 + (self.distal_joint_boost - 1.0) * distance_factor
            
            # Calculate joint movement magnitude
            joint_movement = direction * movement_magnitude * distal_boost
            
            # Add some randomness to simulate natural movement
            random_offset = np.random.normal(0, 0.01, 3)
            new_frame[joint_idx] = base_pose[joint_idx] + joint_movement + random_offset

def main():
    parser = argparse.ArgumentParser(description="Advanced Motion Expansion Generator")
    parser.add_argument("--input", "-i", required=True, help="Input joint data file (.npy)")
    parser.add_argument("--output", "-o", required=True, help="Output file path (.npy)")
    parser.add_argument("--expansion", "-e", type=int, default=5, help="Expansion factor (must be odd)")
    parser.add_argument("--movement-scale", "-s", type=float, default=0.1, 
                       help="Movement amplitude scaling factor (default: 0.1)")
    parser.add_argument("--torso-stability", "-t", type=float, default=0.95,
                       help="Torso stability factor 0-1 (default: 0.95)")
    parser.add_argument("--leg-boost", "-l", type=float, default=1.3,
                       help="Leg movement boost factor (default: 1.3, legs move 30%% more than arms)")
    parser.add_argument("--distal-boost", "-d", type=float, default=1.2,
                       help="Distal joint movement boost factor (default: 1.2, distal joints move 20%% more than proximal)")
    
    args = parser.parse_args()
    
    # Argument validation
    if args.expansion % 2 == 0:
        print("Error: Expansion factor must be odd to ensure the original frame is in the middle")
        return
    
    if args.torso_stability < 0 or args.torso_stability > 1:
        print("Error: Torso stability factor must be between 0 and 1")
        return
    
    if args.movement_scale <= 0:
        print("Error: Movement amplitude scaling factor must be greater than 0")
        return
    
    if args.leg_boost <= 0:
        print("Error: Leg movement boost factor must be greater than 0")
        return
    
    if args.distal_boost <= 0:
        print("Error: Distal joint movement boost factor must be greater than 0")
        return
    
    # Load data
    print(f"Loading input data: {args.input}")
    input_joints = np.load(args.input)
    # print(f"Input shape: {input_joints.shape}")
    
    if input_joints.shape[1:] != (22, 3):
        print("Error: Input data format is incorrect, expected shape (N, 22, 3)")
        return
    
    # Create motion expander
    expander = AdvancedMotionExpander(
        movement_scale=args.movement_scale,
        torso_stability=args.torso_stability,
        leg_movement_boost=args.leg_boost,
        distal_joint_boost=args.distal_boost
    )
    
    # Expand motion
    print(f"Expanding motion using advanced smooth interpolation, factor: {args.expansion}")
    print(f"Movement amplitude scaling: {args.movement_scale}")
    print(f"Torso stability: {args.torso_stability}")
    print(f"Leg movement boost: {args.leg_boost}")
    print(f"Distal joint boost: {args.distal_boost}")
    print(f"Original frame position: frame {args.expansion//2 + 1} (middle)")
    
    expanded_joints = expander.expand_motion(input_joints, args.expansion)
    final = np.concatenate((np.tile(expanded_joints[0, :, :], (2, 1, 1)), expanded_joints), axis=0)

    # Save result
    np.save(args.output, final)
    print(f"Output shape: {final.shape}")
    print(f"Saved to: {args.output}")
    print("Expansion complete!")

if __name__ == "__main__":
    main()