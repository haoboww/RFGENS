from tqdm import tqdm

import torch
import numpy as np
from .radar import Radar as TorchRadar
from PIL import Image
import math

# Prefer the high-performance CUDA renderer when available; fall back to the
# existing torch implementation if import fails or is disabled.
try:
    import sys
    import os
    # Add the radar_cuda directory to the path
    radar_cuda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'radar_cuda')
    if os.path.exists(radar_cuda_path):
        sys.path.insert(0, radar_cuda_path)
    from renderer import Radar as CUDARadar  # root-level accelerated radar
except Exception:  # pragma: no cover - fallback path
    CUDARadar = None

torch.set_default_device('cuda')

def calculate_environment_points(environment_pir):
    """
    environment_pir: (H, W, 3) torch tensor, assumed to be on the correct device (e.g., CUDA)
    Returns: (H*W, 3) point cloud tensor in camera space
    """
    H, W, _ = environment_pir.shape
    device = environment_pir.device

    distance = environment_pir[:, :, 0] * 5 + 5  # [H, W]

    fov_rad = math.radians(60)
    fx = W / (2 * math.tan(fov_rad / 2))
    fy = fx
    cx = W / 2
    cy = H / 2

    j = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)  # rows
    i = torch.arange(0, W, device=device).view(1, -1).expand(H, W)  # cols

    x = (i - cx) / fx  # [H, W]
    y = (j - cy) / fy
    z = torch.ones_like(x, device=device)

    xyz = torch.stack((x, y, z), dim=-1) * distance.unsqueeze(-1)  # [H, W, 3]
    points = xyz.reshape(-1, 3)  # [H*W, 3]
    return points

# def create_interpolator(_frames, _pointclouds, environment_pir, frame_rate=30, remove_zeros = True):
#     num_frames = len(_frames)   
#     total_time = num_frames / frame_rate
#     frames = _frames.copy()
#     pointclouds = _pointclouds.copy()

#     if environment_pir != None:
#         # Reduce the size of environment PIR to reduce memory usage
#         environment_pir = environment_pir.resize((64, 64), resample=Image.Resampling.BILINEAR)
#         environment_pir = torch.tensor(np.array(environment_pir),dtype=torch.float32)/255.0
#         environment_points = calculate_environment_points(environment_pir)
#         environment_intensity = environment_pir[:,:,1].flatten()
#     def interpolator(time):
#             if time < 0 or time > total_time:
#                 raise ValueError("Invalid time value")
            
#             frame_index = int(time * frame_rate)
#             if frame_index == num_frames:
#                 return frames[-1]
            
#             t = (time * frame_rate) % 1 # fractional part of time
#             frame1 = frames[frame_index]
#             frame2 = frames[frame_index + 1]

#             pointcloud1 = pointclouds[frame_index]
#             pointcloud2 = pointclouds[frame_index + 1]



#             zero_depth_frame1 = frame1[:,:, 1] == 0  # zero depth pixels
#             zero_depth_frame2 = frame2[:,:, 1] == 0

#             zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
#             zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)


#             frame1[zero_depth_frame1] = frame2[zero_depth_frame1] # replace zero depth pixels with the other frame
#             frame2[zero_depth_frame2] = frame1[zero_depth_frame2]

#             pointcloud1[zero_depth_frame1_flat] = pointcloud2[zero_depth_frame1_flat] # replace zero depth pixels with the other frame
#             pointcloud2[zero_depth_frame2_flat] = pointcloud1[zero_depth_frame2_flat]


#             interpolated_frame = frame1 * (1 - t) + frame2 * t
#             interpolated_pointcloud = pointcloud1 * (1 - t) + pointcloud2 * t

#             flatten_pir  = interpolated_frame.reshape(-1, 3)

#             intensity = flatten_pir[:,0]
#             depth = flatten_pir[:,1]
            
#             mask = (depth > 0.1) & (intensity > 0.1)

#             if environment_pir != None:
#                 combined_intensity = torch.cat((environment_intensity, intensity[mask]), dim=0)
#                 combined_pointcloud = torch.cat((environment_points, interpolated_pointcloud[mask]), dim=0)
#             else:
#                 combined_intensity = intensity[mask]
#                 combined_pointcloud = interpolated_pointcloud[mask]
#             # return flatten_pir[:,1], interpolated_pointcloud[mask]
#             return combined_intensity, combined_pointcloud
        
    
#     return interpolator

def create_interpolator(_frames, _pointclouds, environment_pir, frame_rate=30, remove_zeros=True):
    num_frames = len(_frames)   
    total_time = num_frames / frame_rate
    frames = _frames.copy()
    pointclouds = _pointclouds.copy()

    if environment_pir != None:
        # Fix: Improve environment PIR processing to ensure consistency
        # Use a fixed resize size to avoid inconsistency between frames
        target_size = (64, 64)
        environment_pir = environment_pir.resize(target_size, resample=Image.Resampling.BILINEAR)
        environment_pir = torch.tensor(np.array(environment_pir), dtype=torch.float32)/255.0
        
        # Add: Ensure the number of environment point clouds is consistent
        environment_points = calculate_environment_points(environment_pir)
        environment_intensity = environment_pir[:,:,1].flatten()
        
        # Fix: Filter out invalid points in environment PIR
        valid_env_mask = environment_intensity > 0.1
        if valid_env_mask.any():
            environment_points = environment_points[valid_env_mask]
            environment_intensity = environment_intensity[valid_env_mask]
        else:
            # If there are no valid environment points, set to empty
            environment_points = torch.empty((0, 3), device=environment_pir.device)
            environment_intensity = torch.empty(0, device=environment_pir.device)
    
    def interpolator(time):
        if time < 0 or time > total_time:
            raise ValueError("Invalid time value")
        
        # Loop playback: map time into the SMPL data range
        cycle_time = time % total_time
        frame_index = int(cycle_time * frame_rate)
        
        # Fix: Complete boundary check
        if frame_index >= num_frames - 1:
            # If out of range, return the last frame
            return frames[-1], pointclouds[-1]
        
        t = (cycle_time * frame_rate) % 1  # fractional part of time
        frame1 = frames[frame_index]
        frame2 = frames[frame_index + 1]
        pointcloud1 = pointclouds[frame_index]
        pointcloud2 = pointclouds[frame_index + 1]

        # Fix: Handle point cloud size mismatch
        # Since pathtracer now returns filtered point clouds, we need to recalculate valid depth pixels
        if len(pointcloud1) > 0:
            # If the point cloud is not empty, there are valid depth values
            # Create a mask matching the PIR frame size
            pir_size = frame1.shape[0] * frame1.shape[1]  # e.g. 128*128 = 16384
            
            # For filtered point clouds, we assume the first N pixels are valid
            valid_pixels = min(len(pointcloud1), pir_size)
            
            # Create mask: first valid_pixels are True, the rest are False
            zero_depth_frame1 = torch.zeros(frame1.shape[:2], dtype=torch.bool, device=frame1.device)
            zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
            zero_depth_frame1_flat[:valid_pixels] = True
            
            zero_depth_frame2 = torch.zeros(frame2.shape[:2], dtype=torch.bool, device=frame2.device)
            zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)
            zero_depth_frame2_flat[:valid_pixels] = True
        else:
            # If there is no point cloud, all pixels are invalid
            zero_depth_frame1 = torch.ones(frame1.shape[:2], dtype=torch.bool, device=frame1.device)
            zero_depth_frame2 = torch.ones(frame2.shape[:2], dtype=torch.bool, device=frame2.device)
            zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
            zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)

        # Handle zero depth pixels
        frame1[zero_depth_frame1] = frame2[zero_depth_frame1]
        frame2[zero_depth_frame2] = frame1[zero_depth_frame2]
        
        # Note: Since the point cloud has already been filtered, we do not need to replace the point cloud here
        # We directly use the filtered point cloud for interpolation

        interpolated_frame = frame1 * (1 - t) + frame2 * t
        
        # For point cloud interpolation, we need to ensure both point clouds have the same structure
        if len(pointcloud1) == len(pointcloud2):
            interpolated_pointcloud = pointcloud1 * (1 - t) + pointcloud2 * t
        else:
            # If the number of point clouds is different, use the smaller one
            min_points = min(len(pointcloud1), len(pointcloud2))
            if min_points > 0:
                interpolated_pointcloud = pointcloud1[:min_points] * (1 - t) + pointcloud2[:min_points] * t
            else:
                interpolated_pointcloud = torch.empty((0, 3), device=frame1.device)

        flatten_pir = interpolated_frame.reshape(-1, 3)
        intensity = flatten_pir[:,0]
        depth = flatten_pir[:,1]
        mask = (depth > 0.1) & (intensity > 0.1)

        # Fix: Ensure the number of intensity and point cloud values are consistent
        if mask.any():
            valid_intensity = intensity[mask]
            valid_pointcloud = interpolated_pointcloud
            
            # If the number of point clouds does not match the number of valid intensity values, adjust
            if len(valid_pointcloud) != len(valid_intensity):
                min_count = min(len(valid_pointcloud), len(valid_intensity))
                if min_count > 0:
                    valid_intensity = valid_intensity[:min_count]
                    valid_pointcloud = valid_pointcloud[:min_count]
                else:
                    valid_intensity = torch.empty(0, device=frame1.device)
                    valid_pointcloud = torch.empty((0, 3), device=frame1.device)
        else:
            valid_intensity = torch.empty(0, device=frame1.device)
            valid_pointcloud = torch.empty((0, 3), device=frame1.device)

        if environment_pir != None and len(environment_points) > 0:
            combined_intensity = torch.cat((environment_intensity, valid_intensity), dim=0)
            combined_pointcloud = torch.cat((environment_points, valid_pointcloud), dim=0)
        else:
            combined_intensity = valid_intensity
            combined_pointcloud = valid_pointcloud
        
        return combined_intensity, combined_pointcloud
    
    return interpolator
    

def generate_signal_frames(body_pirs, body_auxs, envir_pir, radar_config, use_cuda_renderer=True):
    """Generate radar signal frames from PIR data.

    Args:
        body_pirs: List of body PIR tensors.
        body_auxs: List of body point cloud tensors (in scene coordinates).
        envir_pir: Environment PIR image (optional).
        radar_config: Path to radar config JSON or dict.
        use_cuda_renderer: When True and slangtorch-based renderer is available, use CUDA kernel.
    """
    interpolator = create_interpolator(body_pirs, body_auxs, envir_pir, frame_rate=30)
    total_motion_frames = len(body_pirs)
    
    # Try to use CUDA renderer first, fall back to Torch if unavailable
    use_cuda = use_cuda_renderer and (CUDARadar is not None)
    radar = None
    if use_cuda:
        try:
            print("[RFGen] Using CUDA radar renderer (frameCuda) - Expected 5-20x speedup")
            radar = CUDARadar(radar_config)
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"[RFGen] CUDA renderer init failed, falling back to Torch: {e}")
            use_cuda = False
    
    if radar is None:
        print("[RFGen] Using Torch radar renderer (frameMIMO)")
        radar = TorchRadar(radar_config)
    
    # Fix: Ensure the number of radar frames does not exceed the time range of SMPL data
    total_motion_time = total_motion_frames / 30.0  # Total duration (seconds) of SMPL data
    total_radar_frame = int(total_motion_time * radar.frame_per_second)
    
    # Additional safety check
    if total_radar_frame > total_motion_frames:
        print(f"Warning: Radar frames ({total_radar_frame}) exceed SMPL frames ({total_motion_frames})")
        total_radar_frame = total_motion_frames
    
    frames = []
    pointcloud_counts = []  # Add: Monitor the number of point clouds
    
    # Performance monitoring
    import time
    start_time = time.time()
    
    renderer_type = "CUDA" if use_cuda else "Torch"
    print(f"[RFGen] Starting radar signal generation with {renderer_type} renderer")
    print(f"[RFGen] Processing {total_radar_frame} radar frames from {total_motion_frames} motion frames")
    
    for i in tqdm(range(total_radar_frame), desc=f"Generating radar frames ({renderer_type})"):
        current_time = i * 1.0 / radar.frame_per_second
        
        # Add: Monitor the number of point clouds output by the interpolator
        try:
            intensity, pointcloud = interpolator(current_time)
            pointcloud_counts.append(len(pointcloud))
            
            # Print point cloud count information every 10 frames
            if i % 10 == 0:
                print(f"Frame {i}: time={current_time:.2f}s, pointcloud_count={len(pointcloud)}")
                
        except Exception as e:
            print(f"Error in frame {i}: {e}")
            pointcloud_counts.append(0)
        
        # Use appropriate method based on renderer type
        if use_cuda:
            frame_mimo = radar.frameCuda(interpolator, current_time)
        else:
            frame_mimo = radar.frameMIMO(interpolator, current_time)
        frames.append(frame_mimo.cpu().numpy())
    
    frames = np.array(frames)
    
    # Performance statistics
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_radar_frame / total_time if total_time > 0 else 0
    
    print(f"\n[RFGen] Radar signal generation completed!")
    print(f"[RFGen] Renderer: {renderer_type}")
    print(f"[RFGen] Total time: {total_time:.2f}s")
    print(f"[RFGen] Processing speed: {fps:.2f} frames/sec")
    print(f"[RFGen] Average time per frame: {total_time/total_radar_frame*1000:.1f}ms")
    
    # Add: Output point cloud count statistics
    if pointcloud_counts:
        print(f"\n[RFGen] Pointcloud count statistics:")
        print(f"  Min: {min(pointcloud_counts)}")
        print(f"  Max: {max(pointcloud_counts)}")
        print(f"  Mean: {np.mean(pointcloud_counts):.1f}")
        print(f"  Std: {np.std(pointcloud_counts):.1f}")
    
    return frames