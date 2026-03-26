#!/bin/bash

# watch -n 5 ./monitor.sh

# CONFIG
GPU_LIMIT=90
RAM_LIMIT=90
DISK_LIMIT=90

clear

echo "==================== TESLA MONITOR ===================="
date
echo

# GPU
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r util mem_used mem_total
do
    usage_mem=$((mem_used * 100 / mem_total))
    echo "GPU Util: $util% | Mem: $mem_used/$mem_total MB ($usage_mem%)"

    if [ "$util" -gt "$GPU_LIMIT" ]; then
        echo "⚠️ HIGH GPU USAGE"
    fi
done

echo

# RAM
echo "=== RAM ==="
ram=$(free | awk '/Mem:/ {printf("%.0f"), $3/$2 * 100}')
echo "RAM Usage: $ram%"

if [ "$ram" -gt "$RAM_LIMIT" ]; then
    echo "⚠️ HIGH RAM USAGE"
fi

echo

# DISK
echo "=== DISK ==="
disk=$(quota -s | awk 'NR==3 {gsub("M","",$2); print $2}')
echo "Disk Used: $disk MB"

if [ "$disk" -gt 90000 ]; then
    echo "⚠️ CLOSE TO DISK LIMIT"
fi

echo

# PROCESSES
echo "=== PYTHON PROCESSES ==="
ps aux | grep python | grep -v grep

echo
echo "=== JUPYTER ==="
ps aux | grep jupyter | grep -v grep

echo

# TMUX
echo "=== TMUX ==="
tmux ls 2>/dev/null || echo "No tmux sessions"

echo

# PORTS
echo "=== PORT 8888 ==="
lsof -i :8888 || echo "Free"

echo

echo "======================================================="
