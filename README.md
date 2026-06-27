# k8s-qwen-lora

> 在 Kubernetes 集群上完成 Qwen2.5-1.5B 的 QLoRA 微调全流程：Docker GPU 容器化 → Kind K8s 集群 → Training Operator（PyTorchJob CRD）→ 分布式训练 → 容器网络对比实验 → A/B 推理评测。

---

## 项目结构

```
├── training/                        # 训练所有组件
│   ├── train.py                     #   LoRA 微调脚本
│   ├── Dockerfile                   #   GPU 容器镜像
│   ├── requirements.txt             #   Python 依赖
│   ├── pytorchjob.yaml              #   K8s PyTorchJob 清单
│   ├── net-experiment.sh            #   容器网络对比实验
│   └── loss_curve.html              #   训练损失曲线
├── inference/                       # 推理所有组件
│   ├── inference.py                 #   Base vs LoRA A/B 推理
│   ├── inference_results.json       #   对比结果（结构化）
│   └── inference_log.txt            #   完整推理日志
├── model/                           # 训练产物
│   ├── adapter_model.safetensors    #   LoRA 权重（73.9 MB）
│   ├── adapter_config.json          #   LoRA 结构
│   ├── tokenizer.json               #   分词器
│   ├── training_args.bin            #   训练超参数
│   └── trainer_state.json           #   最终训练状态
└── README.md                        # 本文件
```

---

## 环境要求

- **云 GPU 实例**：阿里云 ECS（gn6v / gn5），1× Tesla V100-SXM2-16GB，8 vCPU，62 GB 内存，100 GB ESSD 系统盘
- **OS**：Ubuntu 22.04（阿里云 GPU 镜像，预装 NVIDIA Driver 580 + CUDA 12.8）
- **Docker** ≥ 29.x + NVIDIA Container Toolkit
- **Kubernetes**：kind v0.24.0 / Kubernetes v1.31.0（单节点即可）
- **Python**：3.10（容器内）

---

## 一、GPU 环境配置

### 1.1 确认 GPU 可见

```bash
nvidia-smi
# 应输出 Tesla V100-SXM2-16GB, Driver 580, CUDA 12.8
```

### 1.2 安装 CUDA Toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-8 nvidia-container-toolkit
```

### 1.3 配置 Docker GPU Runtime

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

验证：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 1.4 Docker 镜像加速器（国内必需）

编辑 `/etc/docker/daemon.json`：

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://dockerpull.org"
    ]
}
```

```bash
systemctl restart docker
```

---

## 二、Kubernetes 集群搭建

### 2.1 安装 kubectl

```bash
snap install kubectl --classic
kubectl version --client
```

### 2.2 安装 kind

```bash
snap install go --classic
go env -w GOPROXY=https://goproxy.cn,direct
go install sigs.k8s.io/kind@v0.24.0
cp ~/go/bin/kind /usr/local/bin/kind
```

### 2.3 创建集群

```bash
cat > kind-config.yaml << 'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  - hostPath: /dev
    containerPath: /dev
  - hostPath: /usr/local/nvidia
    containerPath: /usr/local/nvidia
EOF

kind create cluster --config kind-config.yaml --wait 5m
kubectl get nodes
```

### 2.4 安装 Training Operator（PyTorchJob CRD）

```bash
kubectl apply -k 'github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1'
kubectl get crd | grep pytorch
```

---

## 三、构建训练镜像

```bash
cd training
docker build -t qwen-lora:latest .
```

> 镜像约 9.2 GB，国内首次构建约 15-20 分钟（需下载 CUDA 基础镜像 + PyTorch 全家桶）。

---

## 四、启动训练

### 4.1 Docker 直接运行（单机单卡）

```bash
docker run --rm --gpus all \
  -v $(pwd)/../model:/output \
  qwen-lora:latest \
  --epochs 1 --batch_size 4 --grad_accum 4 --save_steps 100
```

### 4.2 K8s PyTorchJob（生产模式）

```bash
kubectl apply -f pytorchjob.yaml
kubectl get pytorchjobs
kubectl logs -f job/qwen-lora-train
```

### 4.3 监控训练

```bash
# GPU 使用
watch -n 1 nvidia-smi

# 训练日志
kubectl logs -f <pod-name>
```

---

## 五、训练结果

| 指标 | 值 |
|---|---|
| 基座模型 | Qwen/Qwen2.5-1.5B |
| 微调方法 | QLoRA（4-bit NF4） + LoRA（r=16, alpha=32） |
| 数据集 | tatsu-lab/alpaca（英文，5000 条） |
| 训练步数 | 313 steps（1 epoch） |
| 训练时长 | 15 分 10 秒 |
| 初始 loss | 1.571 |
| 最终 loss | 1.423 |
| 显存占用 | 7.5 GB / 16 GB |
| LoRA 权重 | 73.9 MB |

---

## 六、模型推理

### 6.1 加载模型

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

model = PeftModel.from_pretrained(base, "./model")
```

### 6.2 生成

```python
inputs = tokenizer("Instruction: 什么是Kubernetes Operator?\nResponse:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 6.3 运行 A/B 对比推理

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v $(pwd)/model:/lora-output \
  -v $(pwd)/inference/inference.py:/app/inference.py:ro \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e PYTHONUNBUFFERED=1 \
  qwen-lora:latest -u /app/inference.py
```

---

## 七、容器网络实验

```bash
bash training/net-experiment.sh
```

对比 Docker bridge（模拟 K8s overlay CNI）与 HostNetwork 的差异：

| 模式 | 延迟 | 封装层级 | NCCL 配置 |
|---|---|---|---|
| Docker bridge / overlay | +5-20μs | NAT + veth pair | `NCCL_SOCKET_IFNAME=eth0` |
| HostNetwork | 0 额外 | 直通宿主机网卡 | `NCCL_SOCKET_IFNAME=eth0` |

关键 NCCL 环境变量：

```yaml
env:
- name: NCCL_SOCKET_IFNAME
  value: "eth0"          # 指定通信网卡
- name: NCCL_IB_DISABLE
  value: "1"             # 禁用 InfiniBand（无 RDMA 网卡时）
- name: NCCL_DEBUG
  value: "INFO"          # 调试级别
- name: NCCL_BUFFSIZE
  value: "2097152"       # 缓冲区大小（2 MB）
```

---

## 八、A/B 推理对比结论

在 6 类 prompt 上对比 base 与 base+LoRA：

| 类型 | Base | Base+LoRA | 结论 |
|---|---|---|---|
| 事实问答 | 正确，啰嗦 | 更结构化 | LoRA 略好 |
| 代码生成 | 正确 | 正确 | 持平 |
| 逻辑推理 | 基本合理 | 计算错误 | Base 更好 |
| 翻译 | 准确 | 格式泄露 | Base 更好 |
| 常识 | 物理原理对 | 对但较浅 | Base 略好 |
| 对话 | 偏正式 | 自然流畅 | LoRA 略好 |

**核心结论**：LoRA 是风格迁移（style transfer），不是能力注入。模型输出语气从"AI 机器人"变为"AI 助手"，但推理和知识能力与 base 持平。

---

## 九、清理与成本控制

```bash
# 不用时在阿里云控制台停止实例（不是释放）
# 停止后仅收系统盘 + 公网 IP：~¥1.5/天
# 运行时：~¥15-18/小时
```

---

## 技术栈

`Docker` · `NVIDIA Container Toolkit` · `Kubernetes (kind)` · `kubectl` · `Kubeflow Training Operator` · `PyTorchJob CRD` · `CNI (Docker bridge / HostNetwork)` · `NCCL` · `QLoRA` · `LoRA (peft)` · `Qwen2.5` · `bitsandbytes` · `transformers`
