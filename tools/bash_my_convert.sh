#!/bin/bash
# Batch script for converting multiple radar frames to point clouds
# Usage: Run from project root directory: bash tools/bash_my_convert.sh

# 自动切换到项目根目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查convert_radar_to_pointcloud.py是否存在
if [ ! -f "tools/convert_radar_to_pointcloud.py" ]; then
    echo "Error: convert_radar_to_pointcloud.py not found in tools directory: $PROJECT_ROOT/tools/"
    exit 1
fi

echo "Working directory: $PROJECT_ROOT"

# python tools/convert_radar_to_pointcloud.py --radar_frames ./output/seq_0/radar_frames.npy --output_dir output_pointclouds/sequence_0/radar

# 最多同时运行10个进程

radar_frames=(
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_0/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_1/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_2/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_3/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_4/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_5/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_6/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_7/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_8/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_9/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/try/RFGENS/output/mmbody_10/radar_frames.npy"    
)



output_dirs=(
    "mmbody_pointclouds/sequence_0/radar"
    "mmbody_pointclouds/sequence_1/radar"
    "mmbody_pointclouds/sequence_2/radar"
    "mmbody_pointclouds/sequence_3/radar"
    "mmbody_pointclouds/sequence_4/radar"
    "mmbody_pointclouds/sequence_5/radar"
    "mmbody_pointclouds/sequence_6/radar"
    "mmbody_pointclouds/sequence_7/radar"
    "mmbody_pointclouds/sequence_8/radar"
    "mmbody_pointclouds/sequence_9/radar"
    "mmbody_pointclouds/sequence_10/radar"
)
max_jobs=10

for i in "${!radar_frames[@]}"; do
    radar_frame="${radar_frames[$i]}"
    output_dir="${output_dirs[$i]}"

    python tools/convert_radar_to_pointcloud.py --radar_frames "$radar_frame" --output_dir "$output_dir"  --simple_normalize
    
    while [ "$(jobs -r | grep -c python)" -ge "$max_jobs" ]; do
        sleep 2
    done
done

wait
echo "所有任务已完成"



