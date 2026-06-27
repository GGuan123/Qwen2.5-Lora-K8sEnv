#!/bin/bash
# Container networking experiment for NCCL distributed training
# Compares HostNetwork vs Docker bridge (overlay equivalent)

echo "=== Container Networking Experiment ==="
echo ""

# 1. Docker bridge (simulates overlay CNI)
echo "--- Test 1: Docker Bridge (default, simulates K8s overlay) ---"
docker run --rm --gpus all --network bridge \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=1 \
  nvidia/cuda:12.8.0-base-ubuntu22.04 \
  bash -c "apt update -qq && apt install -y -qq nvidia-nccl-utils 2>/dev/null; echo 'Bridge network latency:'; ping -c 5 -q 8.8.8.8 2>/dev/null | tail -1" 2>&1

echo ""

# 2. Host network (like K8s HostNetwork)
echo "--- Test 2: Host Network (like Kubernetes HostNetwork) ---"
docker run --rm --gpus all --network host \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=1 \
  nvidia/cuda:12.8.0-base-ubuntu22.04 \
  bash -c "echo 'Host network - no NAT, direct host interface'; ip addr show eth0 2>/dev/null | grep inet" 2>&1

echo ""

# 3. NCCL environment variables guide
echo "=== Key NCCL Environment Variables for Container Networking ==="
cat << 'NCCLINFO'

NCCL_SOCKET_IFNAME=eth0    # Which NIC NCCL uses for communication
NCCL_IB_DISABLE=1          # Disable InfiniBand (set 0 for RDMA)
NCCL_DEBUG=INFO            # Verbosity: VERSION|WARN|INFO|TRACE
NCCL_SOCKET_NTHREADS=4     # Number of socket threads
NCCL_NSOCKS_PERTHREAD=4    # Sockets per thread
NCCL_BUFFSIZE=2097152      # Buffer size (default 4MB, tune for BW)
NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA level (0-5)

# For K8s PyTorchJob, set these in container env:
#   env:
#   - name: NCCL_SOCKET_IFNAME
#     value: "eth0"
#   - name: NCCL_DEBUG
#     value: "INFO"
#   - name: NCCL_IB_DISABLE
#     value: "1"

# Multi-node (2+ machines) needs MASTER_ADDR and MASTER_PORT:
#   - name: MASTER_ADDR
#     value: "<worker-0-ip>"
#   - name: MASTER_PORT
#     value: "29500"
NCCLINFO

echo ""
echo "=== Experiment Complete ==="