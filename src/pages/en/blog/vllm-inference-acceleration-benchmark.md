---
layout: ../../../layouts/BlogPost.astro
title: "Qwen3-VL-4B vLLM Inference Benchmark & Acceleration on Jetson AGX Orin"
date: "2026-08-27"
---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Environment Setup](#2-environment-setup)
  - [2.1 Hardware Configuration Specs](#21-hardware-configuration-specs)
  - [2.2 Basic Software Dependencies and Vertical Stack](#22-basic-software-dependencies-and-vertical-stack)
  - [2.3 Qwen3-VL-4B Model Preparation and Conversion](#23-qwen3-vl-4b-model-preparation-and-conversion)
- [3. vLLM Deployment & Configuration](#3-vllm-deployment--configuration)
  - [3.1 Why Choose vLLM: PagedAttention from a Unified Memory Perspective](#31-why-choose-vllm-pagedattention-from-a-unified-memory-perspective)
  - [3.2 Edge Containerized Deployment and Auto-Start Service Orchestration](#32-edge-containerized-deployment-and-auto-start-service-orchestration)
  - [3.3 Multimodal Architecture Integration: Vision-Language Dynamic Access](#33-multimodal-architecture-integration-vision-language-dynamic-access)
- [4. Benchmarking](#4-benchmarking)
  - [4.1 Test Scenarios and Dataset Construction](#41-test-scenarios-and-dataset-construction)
  - [4.2 Evaluation Metric Definitions](#42-evaluation-metric-definitions)
  - [4.3 Baseline Comparison Experiment Matrix](#43-baseline-comparison-experiment-matrix)
- [5. Acceleration Techniques](#5-acceleration-techniques)
  - [5.1 Model Quantization: W4A16 (AWQ-4bit) Ultimate Squeezing](#51-model-quantization-w4a16-awq-4bit-ultimate-squeezing)
  - [5.2 Kernel and Computation Graph Optimization: CUDAGraph and FlashAttention](#52-kernel-and-computation-graph-optimization-cudagraph-and-flashattention)
  - [5.3 KV Cache Virtualization Tuning under Unified Memory Architecture](#53-kv-cache-virtualization-tuning-under-unified-memory-architecture)
  - [5.4 Batch Scheduling and Speculative Multithread Concurrency Architecture](#54-batch-scheduling-and-speculative-multithread-concurrency-architecture)
- [6. Results & Analysis](#6-results--analysis)
  - [6.1 Performance Test Data Overview](#61-performance-test-data-overview)
  - [6.2 Hardware Compute and GPU Memory Bottleneck Deep Dive](#62-hardware-compute-and-gpu-memory-bottleneck-deep-dive)
  - [6.3 Special Challenges of Edge ARM64 SoC](#63-special-challenges-of-edge-arm64-soc)
- [7. Practical Recommendations](#7-practical-recommendations)
  - [7.1 Production Environment Recommended Configuration Guide](#71-production-environment-recommended-configuration-guide)
  - [7.2 Common Production Pitfalls and Avoidance Manual](#72-common-production-pitfalls-and-avoidance-manual)
  - [7.3 Industrial Edge Typical Application Scenarios](#73-industrial-edge-typical-application-scenarios)
- [8. Conclusion & Future Work](#8-conclusion--future-work)
  - [8.1 Core Achievement Summary](#81-core-achievement-summary)
  - [8.2 Existing Engineering Limitations](#82-existing-engineering-limitations)
  - [8.3 Future Optimization Evolution Directions](#83-future-optimization-evolution-directions)
- [9. References](#9-references)
- [10. Appendix](#10-appendix)
  - [10.1 Production Environment Container Startup Scripts and Key Configuration](#101-production-environment-container-startup-scripts-and-key-configuration)
  - [10.2 Online Streaming Measured Log Fragments](#102-online-streaming-measured-log-fragments)

---

## 1. Introduction

Vision-Language Models (VLMs) are reshaping the technical boundaries of industrial vision detection. These models, by integrating text, images, and other multimodal data, show significant advantages in complex scene understanding. **Qwen3-VL-4B**, as the latest multimodal model launched by DAMO Academy of Alibaba, with its **Chinese-English bilingual understanding, high-resolution visual parsing, and contextual reasoning capabilities**, is particularly suitable for industrial applications such as **pantograph fault detection** that need refined visual recognition.

The industrial on-vehicle system uses **NVIDIA Jetson AGX Orin** as the edge AI computing platform, equipped with Ampere architecture GPU (2048 CUDA cores, 64 Tensor cores) and 64GB LPDDR5 memory. Theoretically it has the capability to support medium-scale model inference. But limited by **limited GPU memory (64GB) and ARM64 architecture compatibility restrictions**, directly deploying Qwen3-VL-4B still faces **out-of-memory (OOM) and inference latency** problems.

Traditional inference frameworks (such as HuggingFace Transformers) when running on edge devices often lead to performance bottlenecks due to **memory fragmentation, inefficient KV Cache management, lack of batching optimization**. For example, using native PyTorch to infer Qwen3-VL-4B on Jetson AGX Orin may, due to **insufficient memory** or **inference speed being too slow**, fail to meet **real-time fault detection** requirements. And **vLLM**, as a framework specially optimized for large model inference, through technologies such as **PagedAttention and continuous batching**, significantly improves inference efficiency and memory utilization. But vLLM officially mainly targets **x86 architecture** optimization; its adaptability and performance on **ARM64 platforms (such as Jetson)** still lack systematic verification.

Based on this, we raise the following key questions:

- **Can Qwen3-VL-4B run smoothly on Jetson AGX Orin?**
- **Can vLLM play its performance advantages on edge devices as in the cloud?**
- **Through which optimization means (such as quantization, tensor parallelism) can inference efficiency be further improved?**
- How to reduce the main program cold start time of the pantograph fault detection system's main program to an acceptable range through **microservice containerization technology** (such as preloading, shared memory, lightweight images)?

To answer these questions, we deployed Qwen3-VL-4B on **Jetson AGX Orin**, combined with vLLM for **inference benchmark testing and acceleration optimization**. This article will detail the **deployment process, performance test results, and optimization strategies**, providing a **practical technical reference** for developers who want to deploy **pantograph fault detection systems** on edge devices.

---

## 2. Environment Setup

### 2.1 Hardware Configuration Specs

The test platform uses a hardened on-vehicle edge industrial computer. The following is our hardware configuration:

| Hardware Component | Specs |
| :--- | --- |
| **Device Model** | NVIDIA Jetson AGX Orin 64GB |
| **GPU** | NVIDIA Ampere architecture, 2048 CUDA cores |
| **Compute Specs** | 275 TOPS (INT8) / 138 TFLOPS (FP16) |
| **CPU** | 12-core ARM Cortex-A78AE (64-bit) |
| **Memory** | 64GB LPDDR5 |
| **Storage** | 64GB eMMC 5.1 + 1TB NVMe M.2 SSD |
| **Operating System** | Ubuntu 22.04 + JetPack 6.2.2 |

> **Note**:
>
> - **Power mode**: it is recommended to set Jetson to **MAXN mode** (maximum performance mode):
>
>   ```bash
>   sudo nvpmodel -m 1  # Enable MAXN mode (all cores run at full speed)
>   sudo jetson_clocks  # Fix CPU/GPU frequency (optional, for benchmark testing)
>   ```

---

### 2.2 Software Dependencies

#### **2.2.1 Operating System and Drivers**

Jetson AGX Orin needs to preinstall the **NVIDIA JetPack SDK**, which is the official complete software stack containing CUDA, cuDNN, TensorRT, and other key components.

1. **Check JetPack version**:

   ```bash
   sudo apt-cache show nvidia-jetpack
   ```

   Or check the current version through NVIDIA SDK Manager. This article's test environment uses **JetPack 6.2.2 r36.4** (based on Ubuntu 22.04).

2. **Verify CUDA and cuDNN**:

   ```bash
   nvcc --version  # Check CUDA version (should be 12.x)
   cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2  # Check cuDNN version
   ```

---

#### **2.2.2 Container Runtime Dependencies**

1. **Install Docker and NVIDIA Container Toolkit**:
   Jetson AGX Orin needs **Docker** and **NVIDIA Container Toolkit** to support GPU-accelerated containers.

   ```bash
   # Install Docker
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl enable docker
   sudo systemctl start docker

   # Add current user to docker group (avoid sudo)
   sudo usermod -aG docker $USER
   newgrp docker  # Refresh group permissions (or re-login)

   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update
   sudo apt install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Verify Docker and GPU support**:

   ```bash
   docker --version  # Confirm Docker is installed
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
   ```

   > **Expected output**: should display Jetson AGX Orin's GPU information (such as `NVIDIA Orin`).

---

#### **2.2.3 Pull vLLM Container**

NVIDIA officially provides a vLLM container optimized for the Jetson platform, integrated with:

- CUDA 12.6
- PyTorch (ARM64 compatible version)
- vLLM 0.19.0
- Other dependencies (such as `bitsandbytes`, `sentencepiece`, etc.)

```bash
# Pull container (about 5-10GB, depending on network speed)
docker pull ghcr.io/nvidia-ai-iot/vllm:0.19.0-r36.4.tegra-aarch64-cu126-22.04

# Verify container
docker images | grep vllm
```

> **Note**:
>
> - Container tag `0.19.0-r36.4.tegra-aarch64-cu126-22.04` corresponds to **JetPack 6.0 (r36.4)** and **CUDA 12.6**.
> - If your JetPack version is different, select the corresponding tag (such as `r35.4.1` corresponds to JetPack 5.1.3).

---

#### **2.2.4 Container Startup and Verification**

1. **Start container (interactive mode)**:

   ```bash
   docker run --gpus all -it --rm \
     -v /path/to/models:/models \  # Mount model directory
     -v /path/to/output:/output \  # Mount output directory (optional)
     --name vllm-qwen3 \
     ghcr.io/nvidia-ai-iot/vllm:0.19.0-r36.4.tegra-aarch64-cu126-22.04
   ```

   > **Parameter explanation**:
   >
   > - `--gpus all`: enable GPU support.
   > - `-v /path/to/models:/models`: mount local model directory to `/models` inside container.
   > - `-it`: interactive terminal.
   > - `--rm`: automatically delete after container exits (can be removed for debugging).

2. **Verify environment inside container**:

   ```bash
   # Check CUDA version
   nvcc --version  # Should output 12.6

   # Check PyTorch
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

   # Check vLLM
   python -c "import vllm; print(vllm.__version__)"  # Should output 0.19.0
   ```

---

## **2.3 Model Preparation**

*Based on Jetson AI Lab's AWQ-4bit quantized version*

---

### **2.3.1 Choose the Right Model Version**

Due to Jetson AGX Orin's **memory limits** (32GB/64GB), directly using FP16 precision Qwen3-VL-4B may cause **out-of-memory (OOM)**. Therefore, it is recommended to use the **AWQ-4bit quantized version**, which is optimized by Jetson AI Lab and can significantly reduce memory usage.

| **Model Version** | **Precision** | **Size** | **Memory Usage (during inference)** | **Applicable Scenarios** |
| | --- | --- | --- | --- |
| Qwen/Qwen3-VL-4B | FP16 | ~8GB | ~16-20GB | 64GB version Jetson AGX Orin |
| **cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit** | **AWQ-4bit** | **~2.1GB** | **~6-8GB** | **32GB/64GB version (recommended)** |

> **Why choose AWQ-4bit?**
>
> - **AWQ (Activation-aware Weight Quantization)** is an efficient quantization method that maintains good model performance at **4bit precision**.
> - Tested by Jetson AI Lab official, has **best compatibility and performance** on Jetson platforms.
> - Memory usage is **only 1/3 of FP16**, suitable for edge devices.

---

### **2.3.2 Download AWQ-4bit Model**

#### **Method 1: using Hugging Face Hub (recommended)**

1. **Download model on local host**:

   ```bash
   # Install huggingface_hub
   pip install -U huggingface-hub

   # Download AWQ-4bit model (about 2.1GB)
   huggingface-cli download \
     --resume-download \
     --local-dir /path/to/models/Qwen3-VL-4B-AWQ-4bit \
     cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit
   ```

   > **Note**:
   >
   > - Model repository: [cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit](https://huggingface.co/cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit)
   >
   > - If network is slow, can configure `HF_ENDPOINT` to use mirror site:
   >
   >   ```bash
   >   export HF_ENDPOINT=https://hf-mirror.com
   >   ```

2. **Verify model files**:

   ```bash
   ls -lh /path/to/models/Qwen3-VL-4B-AWQ-4bit
   ```

   > **Expected output**:

   ```
   -rw-r--r-- 1 user user  147B May 10 12:00 config.json
   -rw-r--r-- 1 user user  1.2K May 10 12:00 generation_config.json
   -rw-r--r-- 1 user user  25M May 10 12:00 tokenizer.model
   -rw-r--r-- 1 user user  30K May 10 12:00 tokenizer_config.json
   -rw-r--r-- 1 user user  1.9G May 10 12:00 model.safetensors  # AWQ-4bit weights
   ```

### **2.3.3 Model Directory Structure**

Organize model files into the following structure (compatible with vLLM):

```
/path/to/models/
└── Qwen3-VL-4B-AWQ-4bit/
    ├── config.json              # Model configuration
    ├── generation_config.json   # Generation configuration
    ├── tokenizer.model          # Tokenizer
    ├── tokenizer_config.json    # Tokenizer configuration
    └── model.safetensors        # AWQ-4bit weight files
```

---

### **2.3.4 Verify Model Loading**

Before starting the vLLM service, first verify whether the model can be correctly loaded inside the container.

#### **Step 1: Start container**

```bash
sudo docker run -it --rm \
  --runtime=nvidia \
  --network host \
  -v /path/to/models:/models \  # Mount model directory
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  bash
```

> **Parameter explanation**:
>
> - `--runtime=nvidia`: enable GPU support (equivalent to `--gpus all`).
> - `--network host`: use host network, convenient for subsequent API access.
> - `-v /path/to/models:/models`: mount local model directory to `/models` inside container.

#### **Step 2: Test model loading inside container**

```bash
# Execute inside container
python -c "
from vllm import LLM
import torch

model_path = '/models/Qwen3-VL-4B-AWQ-4bit'

# Load model (AWQ-4bit auto-detected)
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,      # Single GPU
    trust_remote_code=True,      # Qwen3-VL needs this parameter
    dtype=torch.float16,         # AWQ-4bit internally uses FP16 computation
    max_model_len=2048,         # Maximum context length
    gpu_memory_utilization=0.8  # GPU memory utilization (0-1)
)
print('✅ Model loaded successfully!')
print(f'Model type: {llm.get_model_config().model_type}')
print(f'Model size: {llm.get_model_config().hidden_size} hidden size')
"
```

> **Expected output**:
>
> ```
> ✅ Model loaded successfully!
> Model type: qwen3
> Model size: 4096 hidden size
> ```

---

### **2.3.5 Using Jetson AI Lab Provided Service Command**

Jetson AI Lab provides a **one-key startup command**, suitable for production environments. The following is detailed analysis and optimization suggestions:

#### **Basic Service Command**

```bash
sudo docker run -it --rm --pull always \
  --runtime=nvidia \
  --network host \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit \
    --gpu-memory-utilization 0.8
```

> **Parameter explanation**:
>
> - `--pull always`: always pull the latest image (ensure using the latest version).
> - `--runtime=nvidia`: enable GPU support.
> - `--network host`: use host network, convenient for API access.

---

## 3. vLLM Deployment and Configuration

### 3.1 vLLM Introduction
[vLLM](https://github.com/vllm-project/vllm) is a framework specially optimized for **large language model (LLM) inference**, jointly developed by UC Berkeley and other institutions. Its core features include:

| **Feature** | **Description** | **Meaning for Jetson AGX Orin** |
| --- | --- | --- |
| **PagedAttention** | Non-contiguous memory management, reducing memory fragmentation | Significantly reduces GPU memory usage, avoiding OOM errors |
| **Continuous batching** | Dynamic batch processing requests, improving throughput | Improves multi-user concurrent inference efficiency |
| **Multimodal support** | Native support for text and image inputs (such as Qwen3-VL) | Directly integrate multimodal models, no extra adaptation needed |
| **Quantization support** | Supports AWQ, bitsandbytes, and other quantization methods | Seamless integration with Jetson AI Lab's AWQ-4bit models |
| **Tensor parallelism** | Supports multi-GPU inference (Jetson AGX Orin is single GPU, set to 1) | Maximize single GPU performance |

> **Why choose vLLM?**
>
> - On edge devices, traditional inference frameworks (such as Hugging Face Transformers) lead to performance bottlenecks due to **memory fragmentation** and **inefficient KV Cache management**.
> - vLLM stores KV Cache in non-contiguous memory through **PagedAttention**, significantly improving memory utilization.
> - Supports **streaming output** and **batching**, suitable for real-time inference scenarios.

### 3.2 Basic Deployment

**3.2.1 Start vLLM Service**

Use the container provided by Jetson AI Lab to start the inference service for Qwen3-VL-4B-AWQ-4bit:

```bash
# Start production-grade vLLM inference microservice container
docker run -d \
  --name gongwang_vllm_service \
  --restart always \
  --runtime=nvidia \
  --network host \
  --ipc=host \
  -v /models/Qwen3-VL-4B-Instruct-AWQ-4bit:/models/qwen3-vl \
  vllm-jetson-orin:r36.4 \
  vllm serve /models/qwen3-vl \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-VL-4B-Instruct \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager
```

> [!IMPORTANT]
> **Key parameter explanation**:
>
> - `--gpu-memory-utilization 0.80`: on Orin 64GB unified memory, must reserve 20% (about 12GB) physical memory for the Linux system kernel, desktop, and video pull stream buffer, to prevent the system from triggering OOM;
> - `--max-model-len 4096`: carefully plan context window combined with 4 frames 720P visual Tokens and prompt length, to prevent long sequences from occupying GPU memory;
> - `--enforce-eager`: on ARM64 initial startup, avoid some misaligned operator computation graph warmup crashes, improving runtime stability.

---

> **Parameter explanation**:
>
> - `--name gongwang_vllm_service`: name the container, convenient for management.
> - `-d`: run in background.
> - `--runtime=nvidia`: enable GPU support.
> - `--network host`: use host network, convenient for API access.
> - `-v /path/to/models:/models`: mount local model directory.
> - `--gpu-memory-utilization 0.8`: on Orin 64GB unified memory, GPU memory utilization set to 0.8, reserve physical memory for Linux system kernel, desktop, and video pull stream buffer, to prevent the system from triggering OOM.
> - `--max-model-len 2048`: maximum context length (Token count).
> - `--host 0.0.0.0`: listen on all network interfaces.
> - `--port 8000`: API service port.

#### **3.2.2 Verify Service Status**

1. **Check container logs**:

   ```bash
   docker logs vllm-qwen3-vl
   ```

   > **Expected output**:

   ```
   INFO: Starting vLLM server...
   INFO: Model loaded successfully: Qwen3-VL-4B-AWQ-4bit
   INFO: Server listening on http://0.0.0.0:8000
   ```

2. **Test API connectivity**:

   ```bash
   curl http://localhost:8000/v1/models
   ```

   > **Expected output**:

   ```json
   {
     "object": "list",
     "data": [
       {
         "id": "Qwen3-VL-4B-AWQ-4bit",
         "object": "model",
         "created": 1700000000,
         "owned_by": "vllm"
       }
     ]
   }
   ```

## 4. Benchmarking

### 4.1 Test Scenarios and Dataset Construction
The test set is taken from actual high-speed rail operation monitoring video.

### 4.2 Evaluation Metric Definitions
1. **Time to First Token (TTFT)**: latency from multimodal data input to the large model outputting the first Token, mainly reflecting vision encoder (ViT) forward computation and first-word prompt throughput efficiency.
2. **End-to-end single-window diagnosis latency (E2E Latency)**: total time including Stage 1 temporal global classification, Stage 2 fine-grained localization, and post-processing.
3. **Throughput (Throughput & QPS)**: the number of video stream frames (FPS) and diagnosis windows (Windows/s) the system can process per second.
4. **Memory and GPU memory peak (Memory RSS)**: unified memory resident set size and actual KV Cache allocation.

### 4.3 Baseline Comparison Experiment Matrix
We compared three baselines under the same data stream:
1. **Baseline 1: native Transformers (FP16)**: directly call PyTorch 2.4 local model loading.
2. **Baseline 2: TITAN RTX development machine + vLLM official container**: as x86_64 discrete GPU performance reference upper limit.
3. **Optimization solution: Jetson AGX Orin + vLLM containerized accelerated version**: actual measured data on a 50W power-limited embedded platform.

---

## 5. Acceleration Techniques

### 5.1 Model Quantization: W4A16 (AWQ-4bit) Ultimate Squeezing
On memory-constrained devices, the bottleneck of large model inference is often not compute, but **GPU memory read bandwidth (Memory Bandwidth Bound)**.
- **Optimization principle**: keep Vision Encoder using FP16 precision to ensure image features are not distorted; only apply AWQ 4-bit quantization to the linear layer weights of the LLM Decoder.
- **Actual gain**: static weight GPU memory compressed from 8.87GB to 3.48GB, reducing GPU memory load per Token generation by 60%, bringing nearly 2x decoding speedup on Orin.

### 5.2 Kernel and Computation Graph Optimization: CUDAGraph and FlashAttention
On Jetson Orin's Ampere architecture (SM87):
1. **CUDAGraph topology pre-capture**: fix the common Decode sequence execution flow into a CUDA graph during the warmup stage, eliminating complex operator dispatch scheduling latency on the CPU side.
2. **FlashInfer / FlashAttention-2 operator fusion**: rewrote the computation kernel of the attention mechanism, completing Softmax computation in SRAM, significantly reducing global memory reads/writes.

---

## 6. Results and Analysis

### 6.1 Performance Test Data Overview

The following table summarizes end-to-end actual performance metrics under different software/hardware configurations:

| Hardware platform and inference solution | Running mode / precision | TTFT (ms) | Stage 1 time | Stage 2 time | End-to-end steady mean | Memory/GPU memory resident |
| :--- | --- | --- | --- | --- | --- | --- |
| **TITAN RTX (native Transformers)** | Serial / FP16 | ~1,200 ms | 4,300 ms | 3,500 ms (foreign object) | **5,500 ms** | Dynamic fragmented (>20GB) |
| **TITAN RTX (vLLM official container)** | Speculative concurrency / FP16 | **81 ms** | **1,023 ms** | **543 ms** (foreign object) | **1,068 ms** | 17.94 GB (controlled) |
| **Jetson AGX Orin (native Transformers)** | Serial / FP16 | ~1,850 ms | 6,800 ms | 4,200 ms | **8,400 ms** | 28.5 GB (easily OOM) |
| **Jetson AGX Orin (vLLM optimized version)** | Serial / AWQ-4bit | 480 ms | 1,820 ms | 1,150 ms | **2,970 ms** | 14.2 GB (stable) |
| **Jetson AGX Orin (vLLM + speculative concurrency)** | **Speculative concurrency / AWQ-4bit** | **480 ms** | **1,820 ms** | **1,150 ms** | **1,850 ms 🚀** | **14.8 GB (entire machine ample)** |

<p align="center"><em>Table 1: Full-dimension performance benchmark matrix under cross-platform and different optimization solutions</em></p>

### 6.2 Hardware Compute and GPU Memory Bottleneck Deep Dive
Using Jetson's dedicated `tegrastats` tool for full-lifecycle pressure test monitoring:
1. **GPU load and main frequency**: under `MAXN` mode, GPU cores continuously run stably at 1.3GHz full frequency, utilization maintained at 85%~92% during speculative concurrency.
2. **Memory bandwidth bottleneck**: when processing unquantized FP16 visual Tokens, EMC (external memory controller) bandwidth occupancy reached 88%, becoming the main throughput bottleneck; switched to AWQ-4bit, EMC load dropped to 54%, compute-bound regained dominance.
3. **Thermal stability**: under on-vehicle hardened heat sinks, the whole machine ran at high load for a long time, temperature did not exceed 64°C, no frequency reduction risk.

---

## 8. Conclusion and Future Work

### 8.1 Core Achievement Summary
This article fully reviewed the acceleration practice of Qwen3-VL-4B multimodal large model on the Jetson AGX Orin edge platform:
1. Successfully unblocked aarch64 containerized microservice deployment based on vLLM, achieving **0.1 second second-level hot restart self-healing**.
2. Combined with AWQ-4bit quantization and speculative concurrency architecture, end-to-end sliding window latency was greatly compressed, verifying that 4B-scale multimodal large models have all-weather industrial real-time diagnosis feasibility on the edge side.

### 8.2 Existing Engineering Limitations
- **Multi-stream video concurrency bottleneck**: current hardware compute can only support 1~2 streams of 720P video stream second-level temporal inference; under four-way or eight-way on-vehicle panoramic concurrency it still faces compute limits.
- **ARM native support**: currently relies on deeply customized Docker images; the community's automatic distribution mechanism for Jetson native wheels still needs improvement.

### 8.3 Future Optimization Evolution Directions
1. **Try ultimate quantization paradigms**: keep up with the latest FP4 / INT2 asymmetric mixed-precision quantization, further reducing bandwidth requirements.
2. **Explore TensorRT-LLM proprietary engine**: under pure forward fixed-size scenarios, compare extreme operator fusion gains brought by TensorRT-LLM static compilation.
3. **Edge tier migration verification**: plan to further cut and test this optimization pipeline, verifying its degraded operation boundaries on lower power platforms (such as Jetson Orin Nano / NX).

---

## 9. References

1. **vLLM official open-source repository**: [GitHub - vllm-project/vllm](https://github.com/vllm-project/vllm)
2. **NVIDIA Jetson AI Lab practice guide**: [Jetson AI Lab - Generative AI on Jetson](https://www.jetson-ai-lab.com/)
3. **Qwen3-VL official model card**: [HuggingFace - Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen)
4. **PagedAttention original paper**: Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*.
5. **AWQ quantization paper**: Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*.
