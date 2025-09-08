import sys
import os
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
from termcolor import colored
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from transforms3d.axangles import axangle2mat
import sys
sys.path.append("ext/mdm/")
from visualize.vis_utils import joints2smpl,npy2obj
from model.rotation2xyz import Rotation2xyz
import utils.rotation_conversions as geometry

def euler_to_axis_angle(euler_angles):
    """ Converts a set of Euler angles to axis-angle representation."""
    axis_angle_params = np.zeros_like(euler_angles)

    for i in range(euler_angles.shape[0]):
        for j in range(euler_angles.shape[1]):
            euler = euler_angles[i, j]
            r = R.from_euler('xyz', euler)
            axis_angle = r.as_rotvec()
            axis_angle_params[i, j] = axis_angle

    return axis_angle_params

def process(out_dir):
    filename = out_dir+"/obj_diff_raw.npy"
    print(colored("---[RFGen.ObjDiff]:Runing SMPLify, it may take a few minutes.---", 'yellow'))
    print(colored("---[RFGen.ObjDiff]:This may be optimized in future updates.---", 'yellow'))
    data = np.load(filename,allow_pickle=True)
    motion = data[None][0]['motion'].transpose(0,3, 1, 2)
    
    num_frames = motion.shape[1]
    device='0'
    cuda=True
    
    os.chdir("ext/mdm")
    j2s = joints2smpl(num_frames=num_frames, device_id=device, cuda=cuda)
    os.chdir("../..")
    
    motion_tensor, opt_dict = j2s.joint2smpl(motion[0]) 
    thetas = motion_tensor[0, :-1, :, :num_frames]   
                                                # So basicly this would be the posture of SMPL, 
                                                # it is rot6d, but you can convert it to rotation matrix
                                                # see rotation2xyz
    root_translation = motion_tensor[0, -1, :3, :].cpu().numpy().transpose(1,0)


    thetas_matrix = thetas.transpose(2, 0).transpose(1, 2)
    thetas_matrix = geometry.rotation_6d_to_matrix(thetas_matrix)
    thetas_vec3 = geometry.matrix_to_euler_angles(thetas_matrix,"XYZ")
    thetas_vec3 = thetas_vec3.cpu().numpy()
    final_thetas = euler_to_axis_angle(thetas_vec3)
    smpl_params = final_thetas.reshape(final_thetas.shape[0], -1)
    
    shape_params =np.zeros(10) 
    np.savez(out_dir+'/obj_diff.npz',pose=smpl_params,shape=shape_params, root_translation = root_translation,gender="male")
    


def generate(prompt, out_dir):

    os.chdir("ext/mdm/")
    subprocess.run(
        ['python', '-m', 'sample.generate_rfgen', '--model_path', './save/humanml_trans_enc_512/model000200000.pt', 
         '--text_prompt', prompt, 
         '--output_dir', "../../"+out_dir, 
         '--num_samples', '1', '--num_repetitions', '1'])
    os.chdir("../..")
    process(out_dir)
    


def generate_from_joints(joint_file_path, out_dir):
    """
    Generate obj_diff.npz in SMPL format directly from joint data file.
    Skip MDM and SMPLify steps.
    
    Args:
        joint_file_path: Path to joint data file (.npy format)
        out_dir: Output directory
    """
    print(colored("---[RFGen.ObjDiff]: Converting joint data to SMPL format ---", 'green'))
    
    # Load joint data
    joints = np.load(joint_file_path)  # (N, 22, 3)
    num_frames = joints.shape[0]
    
    print(colored(f"---[RFGen.ObjDiff]: Loaded {num_frames} frames with {joints.shape[1]} joints ---", 'green'))
    
    # According to user's joint definition
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
    ]
    
    # Bone connection definition
    BONE_CONNECTIONS = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
        (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
        (16, 18), (17, 19), (18, 20), (19, 21)
    ]
    
    # Improved joint-to-SMPL conversion
    # 1. Compute root position (use pelvis as root)
    root_translation = joints[:, 0, :].astype(np.float32)  # (N, 3) - pelvis position
    
    # 2. Compute relative joint positions (relative to pelvis)
    relative_joints = joints - joints[:, 0:1, :]  # (N, 22, 3)
    
    # 3. Generate SMPL pose parameters
    # SMPL uses 24 joints, each with 3 rotation parameters (axis-angle)
    pose_params = np.zeros((num_frames, 72), dtype=np.float32)
    
    # Build joint hierarchy based on bone connections
    joint_hierarchy = {}
    for parent, child in BONE_CONNECTIONS:
        joint_hierarchy[child] = parent
    
    # Compute pose parameters for each frame
    for frame_idx in range(num_frames):
        for joint_idx in range(22):  # 22 joints
            if joint_idx == 0:  # pelvis, keep zero rotation
                continue
                
            # Get parent joint index
            parent_idx = joint_hierarchy.get(joint_idx, -1)
            if parent_idx == -1 or parent_idx >= joints.shape[1]:
                continue
                
            # Compute current joint position relative to parent
            if joint_idx < joints.shape[1] and parent_idx < joints.shape[1]:
                joint_pos = relative_joints[frame_idx, joint_idx]
                parent_pos = relative_joints[frame_idx, parent_idx]
                
                # Compute bone direction vector
                bone_vector = joint_pos - parent_pos
                
                if np.linalg.norm(bone_vector) > 1e-6:  # Avoid zero vector
                    # Normalize direction vector
                    bone_direction = bone_vector / np.linalg.norm(bone_vector)
                    
                    # Choose default direction based on joint type
                    if joint_idx in [16, 17, 18, 19, 20, 21]:  # Arm-related joints
                        default_direction = np.array([0, 0, 1])
                    elif joint_idx in [1, 2, 4, 5, 7, 8, 10, 11]:  # Leg-related joints
                        default_direction = np.array([0, 0, 1])
                    elif joint_idx in [3, 6, 9, 12]:  # Spine-related joints
                        default_direction = np.array([0, 1, 0])
                    else:  # Other joints
                        default_direction = np.array([0, 0, 1])
                    
                    # Compute rotation
                    if np.linalg.norm(bone_direction - default_direction) > 1e-6:
                        rotation_axis = np.cross(default_direction, bone_direction)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            cos_angle = np.dot(default_direction, bone_direction)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.arccos(cos_angle)
                            
                            # Convert to axis-angle representation
                            axis_angle = rotation_axis * angle
                            
                            # Place rotation parameters in the corresponding position
                            param_start = joint_idx * 3
                            if param_start + 2 < 72:
                                pose_params[frame_idx, param_start:param_start+3] = axis_angle.astype(np.float32)
    
    # 4. Generate default shape parameters (10-dim, all zeros)
    shape_params = np.zeros(10, dtype=np.float32)
    
    # 5. Save as npz format
    np.savez(out_dir+'/obj_diff.npz',
             pose=pose_params,
             shape=shape_params, 
             root_translation=root_translation,
             gender="male")
    
    print(colored(f"---[RFGen.ObjDiff]: Generated obj_diff.npz with {num_frames} frames ---", 'green'))
    print(colored(f"---[RFGen.ObjDiff]: Root translation range: {np.min(root_translation):.3f} to {np.max(root_translation):.3f} ---", 'green'))
    print(colored("---[RFGen.ObjDiff]: Using custom joint definition with 22 joints ---", 'green'))
    print(colored("---[RFGen.ObjDiff]: Note: Using simplified joint-to-SMPL conversion ---", 'yellow'))
    print(colored("---[RFGen.ObjDiff]: For better results, consider implementing proper inverse kinematics ---", 'yellow'))


def generate_from_joints_advanced(joint_file_path, out_dir, joint_order='default'):
    """
    Advanced version of joint-to-SMPL conversion, supports different joint orders and more accurate conversion.
    
    Args:
        joint_file_path: Path to joint data file (.npy format)
        out_dir: Output directory
        joint_order: Joint order type ('default', 'coco', 'openpose', 'custom')
    """
    print(colored("---[RFGen.ObjDiff]: Advanced joint-to-SMPL conversion ---", 'green'))
    
    # Load joint data
    joints = np.load(joint_file_path)  # (N, 22, 3)
    num_frames = joints.shape[0]
    
    print(colored(f"---[RFGen.ObjDiff]: Loaded {num_frames} frames with {joints.shape[1]} joints ---", 'green'))
    
    # 1. Compute root position (use the first joint as root)
    root_translation = joints[:, 0, :].astype(np.float32)  # (N, 3)
    
    # 2. Compute relative joint positions (relative to root)
    relative_joints = joints - joints[:, 0:1, :]  # (N, 22, 3)
    
    # 3. Define different hierarchies based on joint order
    if joint_order == 'coco':
        # COCO joint order
        joint_hierarchy = {
            0: -1,   # nose
            1: 0,    # left eye
            2: 0,    # right eye
            3: 0,    # left ear
            4: 0,    # right ear
            5: 0,    # left shoulder
            6: 0,    # right shoulder
            7: 5,    # left elbow
            8: 6,    # right elbow
            9: 7,    # left wrist
            10: 8,   # right wrist
            11: 0,   # left hip
            12: 0,   # right hip
            13: 11,  # left knee
            14: 12,  # right knee
            15: 13,  # left ankle
            16: 14,  # right ankle
            17: 9,   # left hand
            18: 10,  # right hand
            19: 15,  # left foot
            20: 16,  # right foot
            21: 0,   # head
        }
    elif joint_order == 'openpose':
        # OpenPose joint order
        joint_hierarchy = {
            0: -1,   # nose
            1: 0,    # neck
            2: 1,    # right shoulder
            3: 2,    # right elbow
            4: 3,    # right wrist
            5: 1,    # left shoulder
            6: 5,    # left elbow
            7: 6,    # left wrist
            8: 1,    # right hip
            9: 8,    # right knee
            10: 9,   # right ankle
            11: 1,   # left hip
            12: 11,  # left knee
            13: 12,  # left ankle
            14: 0,   # right eye
            15: 0,   # left eye
            16: 14,  # right ear
            17: 15,  # left ear
            18: 4,   # right hand
            19: 7,   # left hand
            20: 10,  # right foot
            21: 13,  # left foot
        }
    elif joint_order == 'custom':
        # User-defined 22-joint format
        JOINT_NAMES = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        
        BONE_CONNECTIONS = [
            (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
            (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
            (16, 18), (17, 19), (18, 20), (19, 21)
        ]
        
        # Build joint hierarchy based on bone connections
        joint_hierarchy = {}
        for parent, child in BONE_CONNECTIONS:
            joint_hierarchy[child] = parent
    else:
        # Default format
        joint_hierarchy = {
            0: -1,   # root, no parent
            1: 0,    # spine1
            2: 1,    # spine2
            3: 2,    # spine3
            4: 3,    # left shoulder
            5: 3,    # right shoulder
            6: 4,    # left upper arm
            7: 5,    # right upper arm
            8: 6,    # left forearm
            9: 7,    # right forearm
            10: 8,   # left hand
            11: 9,   # right hand
            12: 3,   # left hip
            13: 3,   # right hip
            14: 12,  # left thigh
            15: 13,  # right thigh
            16: 14,  # left shin
            17: 15,  # right shin
            18: 16,  # left foot
            19: 17,  # right foot
            20: 10,  # left finger
            21: 11,  # right finger
        }
    
    # 4. Generate SMPL pose parameters
    pose_params = np.zeros((num_frames, 72), dtype=np.float32)
    
    # Compute pose parameters for each frame
    for frame_idx in range(num_frames):
        for joint_idx in range(min(22, 24)):  # Limit to 24 joints
            if joint_idx == 0:  # root, keep zero rotation
                continue
                
            # Get parent joint index
            parent_idx = joint_hierarchy.get(joint_idx, -1)
            if parent_idx == -1 or parent_idx >= joints.shape[1]:
                continue
                
            # Compute current joint position relative to parent
            if joint_idx < joints.shape[1] and parent_idx < joints.shape[1]:
                joint_pos = relative_joints[frame_idx, joint_idx]
                parent_pos = relative_joints[frame_idx, parent_idx]
                
                # Compute bone direction vector
                bone_vector = joint_pos - parent_pos
                
                if np.linalg.norm(bone_vector) > 1e-6:  # Avoid zero vector
                    # Normalize direction vector
                    bone_direction = bone_vector / np.linalg.norm(bone_vector)
                    
                    # Choose default direction based on joint type
                    if joint_idx in [6, 7, 8, 9]:  # Arm
                        default_direction = np.array([0, 0, 1])
                    elif joint_idx in [14, 15, 16, 17]:  # Leg
                        default_direction = np.array([0, 0, 1])
                    else:  # Other joints
                        default_direction = np.array([0, 0, 1])
                    
                    # Compute rotation
                    if np.linalg.norm(bone_direction - default_direction) > 1e-6:
                        rotation_axis = np.cross(default_direction, bone_direction)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            cos_angle = np.dot(default_direction, bone_direction)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.arccos(cos_angle)
                            
                            # Convert to axis-angle representation
                            axis_angle = rotation_axis * angle
                            
                            # Place rotation parameters in the corresponding position
                            param_start = joint_idx * 3
                            if param_start + 2 < 72:
                                pose_params[frame_idx, param_start:param_start+3] = axis_angle.astype(np.float32)
    
    # 5. Generate default shape parameters (10-dim, all zeros)
    shape_params = np.zeros(10, dtype=np.float32)
    
    # 6. Save as npz format
    np.savez(out_dir+'/obj_diff.npz',
             pose=pose_params,
             shape=shape_params, 
             root_translation=root_translation,
             gender="male")
    
    print(colored(f"---[RFGen.ObjDiff]: Generated obj_diff.npz with {num_frames} frames ---", 'green'))
    print(colored(f"---[RFGen.ObjDiff]: Root translation range: {np.min(root_translation):.3f} to {np.max(root_translation):.3f} ---", 'green'))
    print(colored(f"---[RFGen.ObjDiff]: Joint order: {joint_order} ---", 'green'))
    print(colored("---[RFGen.ObjDiff]: Note: Using advanced joint-to-SMPL conversion ---", 'yellow'))
