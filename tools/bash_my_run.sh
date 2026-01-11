#!/bin/bash
# Batch script for running RF-Genesis simulation on multiple joint data files
# Usage: Run from project root directory: bash tools/bash_my_run.sh

# 自动切换到项目根目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查run.py是否存在
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in project root directory: $PROJECT_ROOT"
    exit 1
fi

echo "Working directory: $PROJECT_ROOT"

# 定义要处理的 joint-file 路径和对应的 name

joint_files=(
   "../my_data/seq_0.npy"
   "../my_data/seq_1.npy"
   "../my_data/seq_2.npy"
   "../my_data/seq_3.npy"
   "../my_data/seq_4.npy"
   "../my_data/seq_5.npy"
   "../my_data/seq_6.npy"
   "../my_data/seq_7.npy"
   "../my_data/seq_8.npy"
   "../my_data/seq_9.npy"
   "../my_data/seq_10.npy"
   "../my_data/seq_11.npy"
   "../my_data/seq_12.npy"
   "../my_data/seq_13.npy"
   "../my_data/seq_14.npy"
   "../my_data/seq_15.npy"
   "../my_data/seq_16.npy"
   "../my_data/seq_17.npy"
   "../my_data/seq_18.npy"
   "../my_data/seq_19.npy"
)
names=(
    "Seq_0"
    "Seq_1"
    "Seq_2"
    "Seq_3"
    "Seq_4"
    "Seq_5"
    "Seq_6"
    "Seq_7"
    "Seq_8"
    "Seq_9"
    "Seq_10"
    "Seq_11"
    "Seq_12"
    "Seq_13"
    "Seq_14"
    "Seq_15"
    "Seq_16"
    "Seq_17"
    "Seq_18"
    "Seq_19"
)

max_jobs=5

for i in "${!joint_files[@]}"; do
    joint_file="${joint_files[$i]}"
    name="${names[$i]}"
    log_file="our_1226_log_${name}.log"

    nohup python run.py --joint-file "$joint_file" --joint-order custom --name "$name" --no-visualize true --no-environment true > "$log_file" 2>&1 &
    
    # 检查当前后台运行的python进程数，超过max_jobs则等待
    while [ "$(jobs -r | grep -c python)" -ge "$max_jobs" ]; do
        sleep 10
    done
done

wait
echo "所有任务已完成"