#!/bin/bash
# Batch script for converting multiple radar frames to point clouds
# Usage: Run from project root directory: bash tools/bash_my_convert.sh

# python convert_radar_to_pointcloud.py --radar_frames ./output/seq_0/radar_frames.npy --output_dir output_pointclouds/sequence_0/radar

# 最多同时运行10个进程

radar_frames=(
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_5/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_6/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_9/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_10/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_17/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_24/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_29/radar_frames.npy"
    "/store/bhw/codes/RF-Genesis/535/RF-Genesis/output_our_test0904/Seq_34/radar_frames.npy"
)




output_dirs=(
    "our_test_output_pointclouds0904/sequence_5/radar"
    "our_test_output_pointclouds0904/sequence_6/radar"
    "our_test_output_pointclouds0904/sequence_9/radar"
    "our_test_output_pointclouds0904/sequence_10/radar"
    "our_test_output_pointclouds0904/sequence_17/radar"
    "our_test_output_pointclouds0904/sequence_24/radar"
    "our_test_output_pointclouds0904/sequence_29/radar"
    "our_test_output_pointclouds0904/sequence_34/radar"
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



