#!/bin/bash
# Batch script for running RF-Genesis simulation on multiple joint data files
# Usage: Run from project root directory: bash tools/bash_my_run.sh

# 定义要处理的 joint-file 路径和对应的 name

joint_files=(
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_5.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_6.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_9.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_10.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_17.npy"
    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_23.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_24.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_29.npy"
#    "/store/bhw/codes/RF-Genesis/data_process/basic0904/seq_our_34.npy"
)
names=(
    # "Seq_5"
    # "Seq_6"
    # "Seq_9"
    # "Seq_10"
    # "Seq_17"
    "Seq_23"
    # "Seq_24"
    # "Seq_29"
    # "Seq_34"
)

max_jobs=5

for i in "${!joint_files[@]}"; do
    joint_file="${joint_files[$i]}"
    name="${names[$i]}"
    log_file="our_k12log_${name}.log"
    nohup python run.py --joint-file "$joint_file" --joint-order custom --name "$name" --no-visualize true --no-environment true > "$log_file" 2>&1 &
    
    # 检查当前后台运行的python进程数，超过max_jobs则等待
    while [ "$(jobs -r | grep -c python)" -ge "$max_jobs" ]; do
        sleep 10
    done
done

wait
echo "所有任务已完成"
