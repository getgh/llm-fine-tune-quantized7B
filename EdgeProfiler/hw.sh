#!/usr/bin/env bash

OUTPUT_FILE="hardware_config.txt"

echo "=== EdgeProfiler Hardware Configuration Collection ===" > $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

###############################################
# 1. Peak FLOPs estimation
###############################################
echo "=== 1. Peak FLOPs Estimation ===" >> $OUTPUT_FILE

CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CORES=$(nproc)
FREQ=$(lscpu | awk -F: '/CPU max MHz/{print $2} /CPU MHz/{print $2}' | head -1 | xargs)

echo "CPU Model: $CPU_MODEL" >> $OUTPUT_FILE
echo "CPU Cores: $CORES" >> $OUTPUT_FILE
echo "Max Frequency: $FREQ MHz" >> $OUTPUT_FILE

PEAK_FLOPS=$(echo "$CORES * $FREQ * 1e6 * 8" | bc)
echo "Estimated Peak FLOPs: $PEAK_FLOPS FLOPs/sec" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

###############################################
# 2. Memory Bandwidth
###############################################
echo "=== 2. Memory Bandwidth ===" >> $OUTPUT_FILE

if command -v stream &> /dev/null; then
    echo "Running STREAM benchmark..." >> $OUTPUT_FILE
    STREAM_OUTPUT=$(stream 2>&1)
    TRIAD_BW=$(echo "$STREAM_OUTPUT" | awk '/Triad:/ {print $2}')
    echo "STREAM Triad Bandwidth: $TRIAD_BW MB/s" >> $OUTPUT_FILE
    MEM_BW=$(echo "$TRIAD_BW * 1e6" | bc)
    echo "Memory Bandwidth: $MEM_BW bytes/sec" >> $OUTPUT_FILE
else
    echo "STREAM benchmark not found." >> $OUTPUT_FILE
    if [ "$EUID" -ne 0 ]; then
        echo "[Warning] dmidecode requires sudo privileges for full memory info." >> $OUTPUT_FILE
    fi
    MEM_TYPE=$(dmidecode -t memory 2>/dev/null | awk -F: '/Type:/{print $2}' | head -1 | xargs)
    MEM_SPEED=$(dmidecode -t memory 2>/dev/null | awk -F: '/Speed:/{print $2}' | head -1 | xargs)
    echo "Memory Type: $MEM_TYPE" >> $OUTPUT_FILE
    echo "Memory Speed: $MEM_SPEED" >> $OUTPUT_FILE
    echo "Estimated Memory Bandwidth: manual calculation required" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

###############################################
# 3. Storage Bandwidth
###############################################
echo "=== 3. Storage Bandwidth ===" >> $OUTPUT_FILE

DISK=$(lsblk -ndo NAME,TYPE | awk '$2=="disk"{print $1}' | head -1)
echo "Testing storage device: /dev/$DISK" >> $OUTPUT_FILE

STORAGE_BW=$(dd if=/dev/$DISK of=/dev/null bs=1M count=1024 iflag=direct 2>&1 \
             | grep -Eo '[0-9\.]+ (MB/s|GB/s)')
echo "Storage Bandwidth: $STORAGE_BW" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

###############################################
# 4. Host-to-Device Bandwidth (PCIe)
###############################################
echo "=== 4. Host-to-Device Bandwidth (PCIe) ===" >> $OUTPUT_FILE

ACCEL=$(lspci | grep -Ei 'nvidia|amd|xilinx|intel.*accel')

if [ -n "$ACCEL" ]; then
    echo "Accelerator detected:" >> $OUTPUT_FILE
    echo "$ACCEL" >> $OUTPUT_FILE
    PCIE_INFO=$(lspci -vv | awk '/LnkCap:/ {print; exit}')
    echo "PCIe Info: $PCIE_INFO" >> $OUTPUT_FILE
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi topo -m >> $OUTPUT_FILE 2>&1
    fi
else
    echo "No accelerator detected. Defaulting to memory bandwidth." >> $OUTPUT_FILE
fi

echo "" >> $OUTPUT_FILE

###############################################
# 5. Network Bandwidth
###############################################
echo "=== 5. Network Bandwidth ===" >> $OUTPUT_FILE

NET_INTERFACE=$(ip route | awk '/default/ {print $5; exit}')

if [ -n "$NET_INTERFACE" ]; then
    NET_SPEED=$(ethtool $NET_INTERFACE 2>/dev/null | awk -F: '/Speed/ {print $2}' | xargs)
    [ -z "$NET_SPEED" ] && NET_SPEED="Unknown"
    echo "Network Interface: $NET_INTERFACE" >> $OUTPUT_FILE
    echo "Network Speed: $NET_SPEED" >> $OUTPUT_FILE
else
    echo "No network interface detected, defaulting to 100 MB/s." >> $OUTPUT_FILE
fi

echo "" >> $OUTPUT_FILE

###############################################
# 6. Python Snippet
###############################################
echo "=== Python HardwareConfig Code ===" >> $OUTPUT_FILE

cat >> $OUTPUT_FILE << EOF
from EdgeProfiler import HardwareConfig

hw = HardwareConfig(
    name='My Isolated System',
    peak_flops=$PEAK_FLOPS,
    memory_bandwidth=${MEM_BW:-0},
    storage_bandwidth='$STORAGE_BW',
    host_to_device_bw='AUTO_OR_MEMORY_BW',
    network_bandwidth=1e8,
    compute_util=0.7,
    memory_util=0.6
)
EOF

echo "" >> $OUTPUT_FILE
echo "Configuration saved to $OUTPUT_FILE"
echo "Review the values and update the Python code as needed."
