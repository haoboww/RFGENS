
#!/usr/bin/env python3
"""
修复版本的雷达数据转换脚本
主要修复多普勒速度计算问题
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import argparse

# 添加genesis路径（从tools目录回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'genesis'))

from genesis.raytracing.radar import Radar
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud

# SMPL关节名称，骨盆是第0个关节
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

PELVIS_JOINT_IDX = 0  # 骨盆关节索引


class FixedPointCloudProcessCFG(PointCloudProcessCFG):
    """修复版本的点云处理配置"""
    
    def __init__(self, radar: Radar):
        super().__init__(radar)
        # 修复多普勒分辨率计算
        self._fix_doppler_resolution()
    
    def _fix_doppler_resolution(self):
        """修复多普勒分辨率计算"""
        # 对于77GHz雷达人体运动，多普勒分辨率应该在0.1-0.5 m/s范围内
        self.frameConfig.doppler_resolution = 0.1  # 0.1 m/s
        
        print(f"Fixed doppler resolution: {self.frameConfig.doppler_resolution:.6f} m/s")


def load_smpl_data(smpl_file: str) -> np.ndarray:
    """加载SMPL数据，返回关节位置"""
    if not os.path.exists(smpl_file):
        raise FileNotFoundError(f"SMPL file not found: {smpl_file}")
    
    smpl_data = np.load(smpl_file, allow_pickle=True)
    
    if 'pose' in smpl_data and 'shape' in smpl_data:
        print(f"Loading RF-Genesis SMPL data from {smpl_file}")
        print(f"Pose shape: {smpl_data['pose'].shape}")
        print(f"Shape shape: {smpl_data['shape'].shape}")
        
        # 简化的关节位置估计
        num_frames = len(smpl_data['pose'])
        joints = np.zeros((num_frames, 22, 3))
        
        if 'root_translation' in smpl_data:
            root_translation = smpl_data['root_translation']
            for i in range(num_frames):
                joints[i, 0] = root_translation[i]  # 骨盆位置
                # 其他关节位置使用简化的相对位置
                joints[i, 1] = root_translation[i] + np.array([-0.1, 0, -0.1])  # 左髋
                joints[i, 2] = root_translation[i] + np.array([0.1, 0, -0.1])   # 右髋
                joints[i, 3] = root_translation[i] + np.array([0, 0, 0.1])      # 脊柱1
        
        return joints
    else:
        joints = np.load(smpl_file)
        if joints.ndim == 2 and joints.shape == (22, 3):
            joints = joints[None, ...]
        return joints


def convert_radar_to_pointcloud_fixed(radar_frames: np.ndarray, radar_config: str) -> List[np.ndarray]:
    """修复版本的雷达帧到点云转换"""
    print(f"Converting radar frames with shape {radar_frames.shape}")
    
    # 使用修复版本的配置
    radar = Radar(radar_config)
    pointcloud_cfg = FixedPointCloudProcessCFG(radar)
    
    pointclouds = []
    num_frames = radar_frames.shape[0]
    
    for frame_idx in range(num_frames):
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx + 1}/{num_frames}")
        
        frame_data = radar_frames[frame_idx]
        
        try:
            pc = frame2pointcloud(frame_data, pointcloud_cfg)
            pc = np.transpose(pc, (1, 0))
            
            if len(pc) > 0:
                pc = pc[np.abs(pc[:, 3]) > 1e-10]  # 过滤掉接近0的速度
            
            if len(pc) == 0:
                print(f"Warning: No valid points in frame {frame_idx}")
                pc = np.zeros((0, 5))
            else:
                pc = pc[:, :5] 
                
                # 限制点云数量到128个点
                if len(pc) > 128:
                    indices = np.random.choice(len(pc), 128, replace=False)
                    pc = pc[indices]
                elif len(pc) < 128:
                    num_to_fill = 128 - len(pc)
                    if len(pc) > 0:
                        fill_indices = np.random.choice(len(pc), num_to_fill, replace=True)
                        fill_points = pc[fill_indices]
                        pc = np.vstack([pc, fill_points])
                    else:
                        pc = np.zeros((128, 5))
            
            pointclouds.append(pc)
            
            # if frame_idx < 5:  # 打印前5帧的详细信息
            #     print(f"Frame {frame_idx}: {len(pc)} points, velocity range: [{pc[:, 3].min():.6f}, {pc[:, 3].max():.6f}]")
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            pointclouds.append(np.zeros((128, 5)))
    
    return pointclouds


def normalize_pointclouds_simple(pointclouds: List[np.ndarray]) -> List[np.ndarray]:
    print("开始简单归一化点云...")
    normalized_pointclouds = []
    
    for i, pc in enumerate(pointclouds):
        if len(pc) == 0:
            normalized_pointclouds.append(pc)
            continue
        
        normalized_pc = pc.copy()
        
        # 简单归一化：x不动，y统一加3，z统一减1
        # X轴：不动
        # Y轴：统一加3
        normalized_pc[:, 1] -= 3
        # Z轴：统一减1
        normalized_pc[:, 2] += 1
        
        normalized_pointclouds.append(normalized_pc)
        
        if i < 5:
            print(f"Frame {i}: 简单归一化后 PC range x: [{normalized_pc[:, 0].min():.3f}, {normalized_pc[:, 0].max():.3f}], y: [{normalized_pc[:, 1].min():.3f}, {normalized_pc[:, 1].max():.3f}], z: [{normalized_pc[:, 2].min():.3f}, {normalized_pc[:, 2].max():.3f}]")
    
    return normalized_pointclouds



# def save_pointclouds(pointclouds: List[np.ndarray], output_dir: str, prefix: str = "frame", index: int = 0):
def save_pointclouds(pointclouds: List[np.ndarray], output_dir: str, prefix: str = "frame"):
    """保存点云数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(pointclouds)} pointclouds to {output_path}")
    
    for i, pc in enumerate(pointclouds):
        filename = output_path / f"{prefix}_{i+1}.npy"
        # filename = output_path / f"{prefix}_{index}.npy"
        np.save(filename, pc)
        
        if i < 5:
            print(f"Saved {filename}: shape {pc.shape}")
    
    print(f"All pointclouds saved to {output_path}")


def visualize_pointclouds(pointclouds: List[np.ndarray], output_dir: str, num_samples: int = 5):
    """可视化点云数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations for {min(num_samples, len(pointclouds))} pointclouds")
    
    for i in range(min(num_samples, len(pointclouds))):
        pc = pointclouds[i]
        
        if len(pc) == 0:
            continue
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D点云图
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 4], cmap='hot', s=20)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Point Cloud {i} (XYZ)')
        
        # 多普勒速度分布
        ax2 = fig.add_subplot(132)
        ax2.hist(pc[:, 3], bins=20, alpha=0.7)
        ax2.set_xlabel('Doppler Velocity (m/s)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Doppler Distribution {i}')
        
        # 强度分布
        ax3 = fig.add_subplot(133)
        ax3.hist(pc[:, 4], bins=20, alpha=0.7)
        ax3.set_xlabel('Intensity')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Intensity Distribution {i}')
        
        plt.tight_layout()
        plt.savefig(output_path / f"pointcloud_vis_fixed_{i:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert radar frames to pointclouds (Fixed Version)')
    parser.add_argument('--radar_frames', type=str, 
                       default='./output/seq_0/radar_frames.npy',
                       help='Path to radar_frames.npy')
    parser.add_argument('--mmbody_labels', type=str, 
                       default='./data_process/basic_process_copy/mmbody_labels.npy',
                       help='Path to mmbody_labels.npy for pelvis positions')
    parser.add_argument('--radar_config', type=str,
                       default=None,
                       help='Path to radar config file (default: ../models/TI1843_config.json)')
    parser.add_argument('--output_dir', type=str,
                       default='output_pointclouds/seq_0/radar',
                       help='Output directory for pointclouds')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--simple_normalize', action='store_true',
                       help='Use simple normalization (x不动, y+3, z-1)')
    parser.add_argument('--index', type=int,
                       default=0,
                       help='Index of the radar frame')
    
    args = parser.parse_args()
    
    # 设置默认radar_config路径
    if args.radar_config is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.radar_config = os.path.join(project_root, 'models', 'TI1843_config.json')
    
    print("=" * 60)
    print("Radar Frames to Pointcloud Converter (FIXED VERSION)")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.radar_frames):
        print(f"Error: Radar frames file not found: {args.radar_frames}")
        return
    
    if not os.path.exists(args.radar_config):
        print(f"Error: Radar config file not found: {args.radar_config}")
        return
    
    # 加载雷达数据
    print(f"Loading radar frames from: {args.radar_frames}")
    radar_frames = np.load(args.radar_frames)
    print(f"Radar frames shape: {radar_frames.shape}")
    
    # 转换为点云
    pointclouds = convert_radar_to_pointcloud_fixed(radar_frames, args.radar_config)
    
    # 归一化处理
    if args.simple_normalize:
        # 使用简单归一化方法
        print("Using simple normalization: x不动, y+3, z-1")
        pointclouds = normalize_pointclouds_simple(pointclouds)
        print("Pointclouds normalized using simple method")
    else:
        print("Using no normalization")
    
    # 保存点云
    # save_pointclouds(pointclouds, args.output_dir, index=args.index)
    save_pointclouds(pointclouds, args.output_dir)
    
    # 可视化
    # if args.visualize:
    #     visualize_pointclouds(pointclouds, args.output_dir)
    
    print("=" * 60)
    print("Conversion completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of pointclouds: {len(pointclouds)}")
    print(f"Pointcloud shape: {pointclouds[0].shape if pointclouds else 'N/A'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
