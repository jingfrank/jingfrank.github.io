---
layout: ../../layouts/BlogPost.astro
title: "【工业多模态实战·系列五】Jetson AGX Orin 生产级 Docker 容器化与全离线交付实战"
date: "2026-09-04"
---

> **专栏导读**：在动车组车载工业机柜中，边缘计算设备往往处于**物理绝对断网（Air-Gapped）**的极端受限环境。算法在实验室开发机上跑通只是第一步，如何将其封装为无公网依赖、版本严格固化、在 ARM64 边缘硬件上一键可复制的生产级容器，才是决定工程交付成败的生命线。本文为 **【动车组受电弓智能监测实战系列】** 第五篇，深度复盘我们在 NVIDIA Jetson AGX Orin 上推行全离线 Docker 容器化架构的工程实践。
>
> 📌 **系列导航**：
> * **👉 系列一：[Jetson Orin 部署 Qwen-VL 踩坑实录与底层软件栈断层复盘](/blog/jetson-orin-qwenvl-deployment-deepdive)**
> * **👉 系列二：[动静分离两阶段 VLM 架构与跨窗口时序一致性滤波实战](/blog/pantograph-vlm-twostage-algorithm)**
> * **👉 系列三：[从 5.5s 到 1.06s：基于 vLLM 与推测并发的高性能推理优化实战](/blog/vllm-inference-acceleration-benchmark)**
> * **👉 系列四：[从无监督热力图到确定性诊断：FastAD 工业异常检测与空间-时序解耦状态机实战](/blog/fastad-industrial-anomaly-detection-decoupling)**
> * **👉 系列五（本文）：[Jetson AGX Orin 生产级 Docker 容器化与全离线交付实战](/blog/jetson-orin-docker-setup-guide)**

---

## 目录

- [一、业务痛点：高铁车载机柜的“极端交付环境”](#一业务痛点高铁车载机柜的极端交付环境)
- [二、跨架构镜像构建流水线（x86 ➔ ARM64 交付件）](#二跨架构镜像构建流水线x86--arm64-交付件)
  - [1. 用 Skopeo 突破异构架构拉取限制](#1-用-skopeo-突破异构架构拉取限制)
  - [2. 官方 Jetson IoT 底座选型机理](#2-官方-jetson-iot-底座选型机理)
  - [3. 一体化 Dockerfile 与 AWQ 权重内嵌权衡](#3-一体化-dockerfile-与-awq-权重内嵌权衡)
- [三、Tegra 统一内存（UMA）与容器资源调度](#三tegra-统一内存uma与容器资源调度)
  - [1. 物理内存共享下的 OOM-Killer 风险](#1-物理内存共享下的-oom-killer-风险)
  - [2. 关键启动参数：内存利用率与 Eager 模式](#2-关键启动参数内存利用率与-eager-模式)
- [四、容器进程守护与生命周期治理 (entrypoint.sh 深度拆解)](#四容器进程守护与生命周期治理-entrypointsh-深度拆解)
  - [1. 生产级探针机制 (Healthcheck Probe)](#1-生产级探针机制-healthcheck-probe)
  - [2. 信号捕获与优雅退出 (Trap Signal)](#2-信号捕获与优雅退出-trap-signal)
  - [3. 为什么必须采用 `--network host`？](#3-为什么必须采用---network-host)
- [五、边缘端一键部署与生产避坑清单](#五边缘端一键部署与生产避坑清单)
  - [1. 避坑一：`/var/lib/docker` 爆满与 NVMe 软链接迁移](#1-避坑一varlibdocker-爆满与-nvme-软链接迁移)
  - [2. 避坑二：容器内外物理时钟漂移排查](#2-避坑二容器内外物理时钟漂移排查)
  - [3. 避坑三：NVIDIA Container Runtime 丢失问题](#3-避坑三nvidia-container-runtime-丢失问题)
- [六、总结与一键装机极简体验](#六总结与一键装机极简体验)

---

## 一、业务痛点：高铁车载机柜的“极端交付环境”

在实验室中，我们习惯了 `pip install`、`docker pull`、在线下载 Hugging Face 权重。但当算法进入机车机房与车载机柜现场时，开发人员面临的是截然不同的现实：

1. **物理绝对隔离（Air-Gapped Network）**：机车边缘计算主机处于专用局域网甚至单机离线状态，严禁连接任何公网，无法拉取镜像，更无法现场在线安装编译依赖；
2. **底层环境“牵一发而动全身”**：车载 Jetson 主机上同时运行着多个子系统的算法（如受电弓检测、接触网监测、走行部测温等）。若直接在宿主机安装 Python 包或修改 CUDA 驱动环境，极易引发第三方库版本冲突（Dependency Hell）；
3. **现场交付的确定性要求**：机务段现场交付窗口通常只有列车入库检修的短短数小时。如果交付流程依赖现场编译 wheel 或复杂的环境变量调试，一旦失败将严重影响列车发车排班。

**容器化不是锦上添花，而是工业边缘场景下的唯一生存之道。** 我们的目标是：在开发端生成一个包含所有代码、运行时、CUDA 算子和量化大模型权重的**单一独立离线 tar 归档包**，在现场只需一条命令，即可 100% 确定性地拉起整套系统。

![Jetson Docker 交付流水线架构图](/images/blog/jetson-docker-setup-architecture.svg)
*图 1：从 x86 异构交叉打包到车载 Jetson 边缘机柜全离线装机的完整交付流水线*

---

## 二、跨架构镜像构建流水线（x86 ➔ ARM64 交付件）

日常开发与大文件打包通常在高性能 x86 工作站上进行，而目标机是 ARM64（aarch64）架构的 Jetson AGX Orin。这里存在严重的跨架构壁垒。

### 1. 用 Skopeo 突破异构架构拉取限制

当在 x86 机器上执行 `docker pull ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04` 时，Docker 守护进程会默认尝试匹配 `linux/amd64` 清单并报错退出。

我们采用专业容器工具 **Skopeo**，通过强制覆盖目标架构参数，实现了多架构分层的断点续传与本地镜像封包：

```bash
#!/bin/bash
# download_base_image_resilient.sh
DIR_CACHE="./images/vllm_dir_cache"
TAR_OUT="./images/vllm_jetson_orin_base.tar"
SRC_IMAGE="docker://ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"
TAG="ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04"

mkdir -p "${DIR_CACHE}"

# 1. 强制指定 aarch64 分层，利用断点续传同步到目录缓存
skopeo copy \
    --insecure-policy \
    --retry-times 20 \
    --override-arch arm64 \
    --override-os linux \
    "${SRC_IMAGE}" \
    dir:"${DIR_CACHE}"

# 2. 将本地缓存封装为标准 docker-archive tar 包
skopeo copy \
    --insecure-policy \
    dir:"${DIR_CACHE}" \
    docker-archive:"${TAR_OUT}":"${TAG}"
```

### 2. 官方 Jetson IoT 底座选型机理

在基础镜像选型上，千万不能使用社区通用的 x86 版 `vllm/vllm-openai:latest`。Jetson 采用的是 Tegra 架构 SoC，其 GPU 架构为 Ampere（SM87），算子编译需要绑定特定的 Tegra 驱动桩和统一内存（UMA）补丁。

NVIDIA 官方给出的嵌入式软件全栈架构（NVIDIA Embedded Software Stack）清晰展示了从硬件层到应用层的完整层次结构：

![NVIDIA Embedded Software Stack](/images/blog/nvidia-embedded-software-stack.svg)
*图 2：NVIDIA 官方 Jetson 嵌入式软件全栈架构（从底层 BSP、CUDA-X 算子库到上层生成式 AI 与容器运行时）*

从上图可以看出，Jetson 的软件栈深度依赖于底层的 **Jetson Linux (L4T)** 和系统级硬件加速库（CUDA 12.x、cuDNN、TensorRT、VPI）。如果使用非针对 Tegra 优化的普通 Docker 镜像，不仅无法调用 SM87 的 Tensor Core 进行量化加速，甚至在容器启动时就会因缺少 Tegra 驱动用户态挂载而直接报错。

因此，我们选用了 NVIDIA 官方维护的专属容器底座：
`ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04`
* **操作系统**：Ubuntu 22.04 aarch64
* **JetPack 版本**：JetPack 6.x (L4T r36.4)
* **CUDA / cuDNN**：CUDA 12.6 + 专为 Orin SM87 编译优化的 vLLM 核心轮子

### 3. 一体化 Dockerfile 与 AWQ 权重内嵌权衡

在镜像设计时，关于大模型权重的存放有两种方案：
* **方案 A（外部卷挂载）**：镜像只有代码，模型通过 `-v /data/models:/app/models` 挂载；
* **方案 B（一体化内嵌）**：将预下载量化的 AWQ 4-bit 权重直接打入镜像层。

在动车组工况下，我们坚决选择了**方案 B（一体化完整镜像）**。因为现场不同车次的挂载路径、磁盘权限极易出现配置偏差，一体化内嵌能确保镜像自给自足，达到“不可变基础设施”的极致标准。

```dockerfile
# deploy/orin/Dockerfile
FROM ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    VLLM_MODEL_PATH=/app/models/Qwen3-VL-4B-Instruct-AWQ-4bit \
    VLLM_MODEL_NAME=Qwen3-VL-4B-Instruct \
    VLLM_GPU_MEMORY=0.8 \
    VLLM_MAX_MODEL_LEN=4096

WORKDIR /app

# 1. 安装业务依赖（利用清华源高速安装纯 Python 依赖包）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    openpyxl pyyaml Pillow requests opencv-python-headless

# 2. 内嵌预量化的 AWQ-4bit 权重（仅占约 3.5GB）与业务代码
COPY models/Qwen3-VL-4B-Instruct-AWQ-4bit /app/models/Qwen3-VL-4B-Instruct-AWQ-4bit
COPY . /app

# 3. 授权自启动入口
RUN chmod +x /app/deploy/orin/entrypoint.sh

ENTRYPOINT ["/app/deploy/orin/entrypoint.sh"]
```

---

## 三、Tegra 统一内存（UMA）与容器资源调度

### 1. 物理内存共享下的 OOM-Killer 风险

与台式机独立显卡（如 24GB 独立显存 + 64GB 内存）完全不同，**Jetson AGX Orin 采用的是统一内存架构（Unified Memory Architecture, UMA）**。64GB 的物理 LPDDR5 内存池是由 CPU、GPU 和多媒体硬件编解码器共同共享的。

在传统 PC 上，显存爆掉只会导致 PyTorch 抛出 `CUDA out of memory` 异常，进程本身不会退出；但在 Jetson 上，如果 vLLM 分配了过大的显存池，导致宿主机系统可用物理内存耗尽，**Linux 内核的 OOM-Killer 会立即触发，并无情地发送 `SIGKILL` 杀死占用内存最大的大模型进程**！

### 2. 关键启动参数：内存利用率与 Eager 模式

为了在保障推理并发的同时确保系统绝对安全，我们在容器环境变量中固化了以下调优参数：

```bash
# 1. 显存配额严格限制为 80% (留出 20% 约 12GB 给系统与其他应用)
--gpu-memory-utilization 0.8

# 2. 强制使用 Eager 模式，关闭极端激进的 CUDA Graph 预分配
--enforce-eager

# 3. 限制上下文最大长度，防止偶发长序列爆显存
--max-model-len 4096
```

通过 AWQ 4-bit 量化，Qwen3-VL-4B 模型权重本身仅占 **3.5 GB**，KV Cache 预留约 6 GB，整体常驻内存稳稳控制在 10 GB 左右，为后台 OpenCV 视频抽帧、时序追踪器与 FTP 缓存留出了巨大的安全余量。

---

## 四、容器进程守护与生命周期治理 (`entrypoint.sh` 深度拆解)

很多初学者喜欢直接在 `Dockerfile` 末尾写 `CMD ["vllm", "serve", "..."]`。但在工业生产中，这种写法极度脆弱：
1. 大模型在启动时需要经历权重反序列化、CUDA 显存编译、KV Cache 内存预分配等阶段，耗时通常在 30~90 秒；
2. 如果在此期间外部业务进程尝试连接 API，会遭遇大量连接被拒；
3. 当使用 `docker stop` 时，若容器内主进程未正确转发信号，会导致后台 vLLM 进程变成僵尸进程，残存显存无法释放。

我们在 [`entrypoint.sh`](file:///home/ibd/jyb/gongwang_jetson_vllm/deploy/orin/entrypoint.sh) 中实现了一套工业级的看门狗（Watchdog）与探针（Probe）机制：

```bash
#!/bin/bash
# deploy/orin/entrypoint.sh
set -e

echo "🚀 [Jetson AGX Orin] 弓网故障监测系统正在启动..."

# 1. 后台静默拉起 vLLM 服务
vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT:-8000}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY:-0.8}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    --trust-remote-code \
    --enforce-eager &
VLLM_PID=$!

# 2. 生产级就绪探针 (Healthcheck Probe) 轮询
MAX_RETRIES=90
RETRY_COUNT=0
READY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # 通过调用 /v1/models 接口探测模型是否完成编译
    if curl -s "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" | grep -q "${SERVED_MODEL_NAME}"; then
        READY=1
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   ⏳ vLLM 预热与显存编译中 (${RETRY_COUNT}/${MAX_RETRIES})..."
    sleep 3
done

if [ $READY -eq 1 ]; then
    echo "✅ [vLLM 就绪] 大模型服务已成功拉起并处于监听状态！"
fi

# 3. 信号捕获与优雅退出 (Graceful Shutdown)
cleanup() {
    echo "🛑 接收到终止信号，正在退出 vLLM 服务 (PID: $VLLM_PID)..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# 保持前台运行守护
wait "$VLLM_PID"
```

### 3. 为什么必须采用 `--network host`？

在运行容器时，我们没有使用 Docker 默认的 bridge 桥接网络，而是指定了 `--network host`：
* **高吞吐拉流零损耗**：系统需要同时从车载交换机接收多路 20~25 FPS 的 RTSP 高清码流，Host 模式省去了 Docker 内部的 `iptables` NAT 转发开销，极大减轻 CPU 软中断负荷；
* **UDP 车载报警报文直接广播**：车载故障上报需向端口（如 `8080`）发送 `0xFF 0x20` 二进制 UDP 报文，Host 模式下容器与物理机共享网络协议栈，确保网络报文毫秒级送达。

---

## 五、边缘端一键部署与生产避坑清单

在车载机房的真实安装调试中，我们整理出了最具价值的三大避坑指南：

### 1. 避坑一：`/var/lib/docker` 爆满与 NVMe 软链接迁移
* **现象**：Jetson 系统自带的 eMMC 存储通常只有 64GB，系统占用后剩余不足 20GB。在执行 `docker load` 导入 15GB 级别的大模型镜像时，系统瞬间报 `No space left on device`。
* **解法**：车载主机通常外挂了企业级 NVMe 固态硬盘（挂载于 `/hirain/hi_opt/nvme1`）。必须在导入前将 Docker 存储根目录迁移：
  ```bash
  sudo systemctl stop docker
  sudo mkdir -p /hirain/hi_opt/nvme1/docker-data
  sudo cp -r /var/lib/docker/* /hirain/hi_opt/nvme1/docker-data/ 2>/dev/null || true
  sudo rm -rf /var/lib/docker
  sudo ln -s /hirain/hi_opt/nvme1/docker-data /var/lib/docker
  sudo systemctl start docker
  ```

### 2. 避坑二：容器内外物理时钟漂移排查
* **现象**：动车组处于封闭内网，无法连接互联网 NTP 服务器。若容器内时间与物理宿主机存在时区偏差（如容器内为 UTC，宿主机为 CST），落盘生成的告警目录 `%m-%d` 会跨天错位，导致上层 PHM 系统检索不到证据链图片。
* **解法**：在 `docker run` 中显式以只读方式挂载宿主机的时钟配置文件：
  ```bash
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro
  ```

### 3. 避坑三：NVIDIA Container Runtime 丢失问题
* **现象**：执行 `docker run --runtime=nvidia ...` 时提示 `unknown or invalid runtime name: nvidia`。
* **解法**：这是由于 Docker 配置文件未注册 NVIDIA 运行时导致。检查 `/etc/docker/daemon.json`，确保包含如下内容并重启 Docker：
  ```json
  {
      "runtimes": {
          "nvidia": {
              "path": "nvidia-container-runtime",
              "runtimeArgs": []
          }
      },
      "default-runtime": "nvidia"
  }
  ```

---

## 六、总结与一键装机极简体验

通过上述标准化的容器工程改造，我们为现场机务工程师提供了一套近乎“零门槛”的部署脚本 [`deploy_on_jetson.sh`](file:///home/ibd/jyb/gongwang_jetson_vllm/deploy/orin/deploy_on_jetson.sh)：

```bash
# 机务段现场交付只需一条命令：
sudo chmod +x deploy_on_jetson.sh
sudo ./deploy_on_jetson.sh pantograph-vllm-orin-v1.0.tar
```

**运行输出效果**：
```text
======================================================================
🚀 [Jetson AGX Orin 部署] 弓网故障监测系统 (vLLM AWQ 版)
======================================================================
>> [1/3] 正在从 pantograph-vllm-orin-v1.0.tar 加载 Docker 镜像 (docker load)...
Loaded image: pantograph-vllm-orin:v1.0
>> [2/3] 确认本地结果存储路径: /hirain/hi_opt/nvme1/Algo-result
>> [3/3] 启动弓网监测系统容器...
======================================================================
🎉 部署完成！容器 gongwang_monitor_vllm 已在后台运行。
----------------------------------------------------------------------
查看运行日志: sudo docker logs -f gongwang_monitor_vllm
======================================================================
```

在车载极端无外网、高振动、强干扰的工业环境下，**将复杂的深度学习依赖与模型权重黑盒固化在规范的 Docker 镜像中，并通过严谨的探针与统一内存管控实现自动守护**，是工业 AI 真正能够走出实验室、走向列车线路规模化商用最坚实的技术底座。
