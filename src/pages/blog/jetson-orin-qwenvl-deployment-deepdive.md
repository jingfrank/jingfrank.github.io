---
layout: ../../layouts/BlogPost.astro
title: "【工业多模态大模型实战·系列一】Jetson Orin AGX 部署 Qwen-VL 踩坑实录与底层软件栈断层复盘"
date: "2026-08-27"
---

> **专栏导读**：在云端服务器上一行 `pip install` 就能跑通的视觉多模态大模型（VLM），搬到 Jetson Orin AGX 边缘端后，迎接你的是驱动断层、内存爆仓、多路推拉流死锁与算子冲突。本文为 **【动车组受电弓多模态大模型落地实战三部曲】** 的第一篇，深度复盘在 Jetson Orin AGX 64GB 上部署 Qwen-VL 的全过程——七个真实致命报错的排错记录、EasyDarwin + FFmpeg 闭环仿真压测床构建、ARM 架构与 JetPack 5/6 软硬件断层的底层根因，以及边缘端选型路线。
>
> 📌 **系列导航**：
> * **👉 系列一（本文）：[Jetson Orin 部署 Qwen-VL 踩坑实录与底层软件栈断层复盘](/blog/jetson-orin-qwenvl-deployment-deepdive)**
> * **👉 系列二：[动静分离两阶段 VLM 架构与跨窗口时序一致性滤波实战](/blog/pantograph-vlm-twostage-algorithm)**
> * **👉 系列三：[从 5.5s 到 1.06s：基于 vLLM 与推测并发的高性能推理优化实战](/blog/vllm-inference-acceleration-benchmark)**

---

## 目录

- [一、问题：受电弓检测为什么需要 VLM](#一问题受电弓检测为什么需要-vlm)
- [二、底层机制剖析：ARM 架构、JetPack SDK 与版本断层](#二底层机制剖析arm-架构jetpack-sdk-与版本断层)
- [三、踩坑实录：七个致命报错与流媒体治理](#三踩坑实录七个致命报错与流媒体治理)
  - [坑点 1：CRLF 跨平台换行符导致脚本崩溃](#坑点-1crlf-跨平台换行符导致脚本崩溃)
  - [坑点 2：Python 3.8 对 PEP 585 类型注解的解析死锁](#坑点-2python-38-对-pep-585-类型注解的解析死锁)
  - [坑点 3：54GB eMMC 空间耗尽与 /dev/shm 内存盘自救](#坑点-354gb-emmc-空间耗尽与-devshm-内存盘自救)
  - [坑点 4：PyTorch 2.1.0a0 假版本与 Alpha 算子缺失](#坑点-4pytorch-210a0-假版本与-alpha-算子缺失)
  - [坑点 5：PyPI 自动依赖解析拉错轮子](#坑点-5pypi-自动依赖解析拉错轮子)
  - [坑点 6：Transformers 与 Qwen 系列的代际版本锁](#坑点-6transformers-与-qwen-系列的代际版本锁)
  - [坑点 7：多路 4K RTSP 拉流导致 FFmpeg 管道死锁与僵尸进程失控](#坑点-7多路-4k-rtsp-拉流导致-ffmpeg-管道死锁与僵尸进程失控)
- [四、车载存储自治：端-车-NAS 分级存储与 80%➔60% 双水位淘汰闭环](#四车载存储自治端-车-nas-分级存储与-8060-双水位淘汰闭环)
- [五、选型：三条落地路线](#五选型三条落地路线)
- [六、承上启下：从硬件避坑走向算法与推理攻坚](#六承上启下从硬件避坑走向算法与推理攻坚)

---

## 一、问题：受电弓检测为什么需要 VLM

我们在做一个动车组受电弓/弓网智能监控项目。受电弓是列车车顶与接触网滑动的受流部件，在 350 km/h 运行中需要实时检测三类异常：

1. **大火花与严重拉弧**：夜间高压拉弧伴随强反光，光照变化剧烈，传统检测容易误判为故障或漏判；
2. **异物挂网**：羊角与滑板挂塑料袋、风筝线、树枝、羽毛——异物形态千奇百怪，传统封闭集目标检测极度缺乏正负样本，极易漏报；
3. **结构异常与几何倾斜**：受电弓滑板磨损、支臂微小形变涉及空间几何关系理解，传统检测框无法表达"语义合理性"。

传统方案是 YOLO 单阶段目标检测。但在上述场景里，YOLO 的短板很明确：它只能识别见过的类别（封闭集），对没见过的异物形态零泛化能力；它输出的是矩形框，无法理解"滑板和支臂的角度关系是否合理"。

视觉多模态大模型（VLM）提供了另一条路：**开放集零样本泛化能力 + 细粒度图文空间推理**。你不需要为每种异物采集数千张标注样本，用一段文字描述就能让它判断"画面中是否存在非金属异物"。

我们选了 Qwen-VL 系列（Qwen2-VL / Qwen2.5-VL / Qwen3-VL）。算法验证在云端 A100 / TITAN RTX 上很顺利。然后，准备部署到车载工控机——Jetson Orin AGX 64GB——攻坚战开始了。

---

## 二、底层机制剖析：ARM 架构、JetPack SDK 与版本断层

在开始动手排错之前，必须先理清 NVIDIA Jetson 系列板卡的软硬件本质。很多在 x86 服务器上百试百灵的经验，搬到 Jetson 上会瞬间失灵。

### 1. Jetson 的计算基石：ARM (aarch64) 异构 SoC

常规云端训练与推理服务器（如配备 A100 / H100 / RTX 4090 的主机）基于 **x86_64（Intel / AMD）** 架构，CPU 与 GPU 通过 PCIe 总线通信，拥有独立的显存控制器。

而 NVIDIA Jetson 系列（包括 Orin AGX、Orin Nano 等）本质上是 **ARM 架构（ARMv8.2-A / aarch64）的嵌入式 SoC（System on Chip）**：
- **CPU 核心**：Jetson Orin 搭载的是基于 ARM 架构的 12 核 Arm Cortex-A78AE CPU；
- **统一物理内存（Unified Memory Architecture）**：CPU 与 Ampere 架构 GPU 共享同一块 64GB LPDDR5 物理内存，没有独立的 GPU 显存条。显存分配直接走 Tegra 内部的高速互联总线与 NVRM 驱动模块；
- **指令集与二进制不兼容**：所有 x86 预编译的二进制可执行文件、Docker 镜像和 C/C++ 动态链接库，都**无法直接在 Jetson 上运行**，必须使用 aarch64 交叉编译或本地原生编译；
- **PyPI 生态陷阱**：在 aarch64 环境下直接 `pip install torch`，PyPI 默认分发的是面向通用 ARM 服务器（如 AWS Graviton / 鲲鹏）的 `manylinux` 轮子，这些通用轮子**完全没有针对 Tegra 统一内存与 Jetson GPU 做适配**，安装后调用 `torch.cuda.is_available()` 必然返回 `False`。

---

### 2. 什么是 JetPack SDK？

为了让开发者能够在 ARM SoC 上发挥 GPU 算力，NVIDIA 推出了专门的软件开发套件——**NVIDIA JetPack SDK**。

JetPack 并不是一个单纯的 Python 包，而是覆盖了从操作系统内核到上层 AI 加速引擎的**全栈软硬件交付套件**：

![Jetson Orin 软件栈与 ABI 约束](/images/blog/jetson-stack.svg)
*与 x86 服务器「PCIe + 独立显卡」不同，Jetson 是统一内存的 Tegra SoC：容器隔离不了内核 ABI。*

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker 容器用户态空间                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 视觉大模型应用层 (Qwen-VL / Transformers / vLLM / llama.cpp)      │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ 深度学习框架 (PyTorch 2.x / TorchVision / TensorRT)               │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ CUDA Runtime (libcudart.so)                                       │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    │ 运行时动态链接 (ABI 强依赖)        │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 驱动挂载目录 (/usr/lib/aarch64-linux-gnu/tegra/)                  │  │
│  │ libcuda.so.1, libnvrm.so, libnvos.so 等核心驱动库                 │  │
│  └─────────────────────────────────▲─────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ CSV 规则只读挂载 (bind-mount)
┌────────────────────────────────────┴────────────────────────────────────┐
│                        宿主机操作系统与底层驱动                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ NVIDIA Container Runtime (/etc/nvidia-container-runtime/l4t.csv)  │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ 宿主机 L4T BSP 用户态驱动库 (Linux for Tegra User-space Drivers)  │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ Tegra 内核驱动模块 (nvgpu.ko, NVRM, Linux Kernel 5.10 / 5.15)     │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ 硬件底座：Jetson Orin AGX (Arm Cortex-A78AE + 64GB 统一物理内存)  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **L4T (Linux for Tegra)**：包含 Linux 内核（Kernel）、Bootloader、Tegra 硬件抽象层驱动（NVRM / NVGPU 模块）以及专为 Jetson 定制的 Ubuntu 根文件系统；
2. **CUDA Toolkit & cuDNN / TensorRT**：针对 Tegra SoC 定制编译的底层异构加速库与推理引擎；
3. **NVIDIA Container Runtime**：Jetson 专属的容器运行时。在容器启动时，它根据宿主机 `/etc/nvidia-container-runtime/host-files-for-container.d/` 目录下的 CSV 配置文件（如 `l4t.csv`），将宿主机的用户态驱动（`libcuda.so.1`、`libnvrm.so`）**直接只读 bind-mount 挂载进容器**；
4. **官方专供 PyTorch 轮子**：由于 PyPI 上的通用包无法使用，NVIDIA 官方维护了 [PyTorch for Jetson Platform Release Notes](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)，为各个 JetPack/L4T 版本独立编译并发布专属的 `torch-xxx-nv-cp38-linux_aarch64.whl` 预编译包。

---

### 3. JetPack 5 与 JetPack 6 的代际断层剖析

在实际工程交付中，开发者最常踩的坑正是 **JetPack 5** 与 **JetPack 6** 之间的软硬件断层：

| 对比维度 | JetPack 5.x (L4T R35.x) | JetPack 6.x (L4T R36.x) | 软硬件断层影响 |
| :--- | :--- | :--- | :--- |
| **发布年代** | 2022-2023 | 2024 | - |
| **基础操作系统** | Ubuntu 20.04 LTS (Focal) | Ubuntu 22.04 LTS (Jammy) | GLIBC 2.31 升至 2.35，高版本动态库无法向下兼容 |
| **Linux 内核** | Linux 5.10-tegra (专有 BSP) | Linux 5.15 (引入上游通用内核树) | 内核驱动架构大改，模块解耦 |
| **系统 Python** | **Python 3.8** 原生默认 | **Python 3.10** 原生默认 | 现代开源模型库强制依赖 Python 3.10+ PEP 语法 |
| **CUDA 驱动 ABI** | **CUDA 11.4** 驱动接口 | **CUDA 12.2 / 12.6** 驱动接口 | 容器内无法跨大版本直接运行高版本 CUDA Runtime |
| **官方 PyTorch 轮子** | 止步于 `2.0.0` / `2.1.0a0` (nv23.06) | 支持 `2.3.0` / `2.4.0`+，支持 FlashAttention | JP5 官方轮子缺失现代 VLM 必备的注意力与动态切片算子 |
| **现代 VLM 支持度** | 仅支持老版本 Transformers / Qwen 初代 | 原生支持 Qwen2.5-VL、Qwen3-VL、vLLM | 现代模型体系在 JP5 上面临重重版本阻碍 |

#### Qwen-VL 在 Jetson 上的兼容性矩阵

| 模型代际 | 最低 Transformers | 最低 PyTorch / Python | JetPack 5 (R35) | JetPack 6 (R36) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen-VL** | >= 4.31.0 | PyTorch 2.0 / Py3.8 | ✅ 完全原生支持 | ✅ 支持 |
| **Qwen2-VL (2B/7B)** | >= 4.45.0 | PyTorch 2.0 / Py3.8 | ✅ 完美原生支持 (4.46.3) | ✅ 完全原生支持 |
| **Qwen2.5-VL (3B/7B)** | >= 4.49.0 | PyTorch 2.1+ / Py3.10 | ⚠️ 需 llama.cpp / GGUF | ✅ 完全原生支持 |
| **Qwen3-VL (4B)** | >= 4.57.0 | PyTorch 2.3+ / Py3.10 | ❌ 底层驱动与算子断层 | ✅ 完全原生支持 (vLLM 0.7+) |

---

### 4. 算法跃进与工业工控底座的结构性矛盾

这就引出了边缘端部署最棘手的根本死结：

1. **算法端“日新月异”**：2024~2026 年开源的 Qwen-VL 系列（Qwen2-VL / Qwen2.5-VL / Qwen3-VL）及其依赖的 `transformers`（4.45+ 到 4.57+）、`accelerate` 库，在底层大量使用了 Python 3.10+ 的类型泛型语法，并在注意力计算中强制调用 PyTorch 2.2+ 的动态算子；
2. **硬件端“稳定至上”**：工业现场的车载工控机大多由研华、米尔等工控硬件厂商出厂烧录了定制载板的 JetPack 5 BSP（内含特定 GMSL 相机解串芯片、CAN FD、隔离 IO 的驱动与设备树）。若强行现场刷机升级到 JetPack 6，需要拆开机箱按住 Recovery 物理按键，且极可能导致定制外设驱动全部损坏；
3. **驱动绑定无法逃逸**：由于 Docker 依赖宿主机驱动映射，即便你在容器里拉取最新的 Ubuntu 22.04 镜像，一旦加载宿主机的老旧 L4T R35 驱动，底层 CUDA 也会直接报 `CUDA driver version is insufficient`。

**“前线算法团队要跑最新模型，后方工业工控机锁死老旧 BSP”**——理解了这一底座机制，后面遇到的六大致命报错便不再是孤立的 bug，而是这一结构性矛盾在各个层面的必然体现。

---

## 三、踩坑实录：六个致命报错

以下六个故障全部来自真实调试现场。

<!-- SCREENSHOT-SLOT: 可插入一张真实终端报错截图增强现场感（候选：坑点 1 的 bash\r 报错 / 坑点 3 的 no space left on device + df -h 输出）：
![Jetson 现场报错终端](/images/blog/shot-jetson-terminal.png)
-->

### 坑点 1：CRLF 跨平台换行符导致脚本崩溃

**现象**：从 Windows 开发机通过 Git/SCP 将脚本传到 Jetson 后执行 `./autotag` 报错：

```bash
hirain@tegra-ubuntu:~/Desktop/jetson-containers$ ./autotag transformers
/usr/bin/env: bash\r: No such file or directory
```

**根因**：Windows 默认用 `\r\n` (CRLF) 换行，Linux 将 `bash\r` 视作不存在的解释器路径。  
**解法**：一行命令递归清洗：
```bash
find . -type f -exec sed -i 's/\r$//' {} +
```

---

### 坑点 2：Python 3.8 对 PEP 585 类型注解的解析死锁

**现象**：运行 jetson-containers 官方工具脚本时抛出语法异常：
```text
File ".../l4t_version.py", line 465, in <module>
    def _parse_python_ver_and_nogil(s) -> tuple[Version, bool]:
TypeError: 'type' object is not subscriptable
```
**根因**：`tuple[...]` 是 Python 3.9+ 的 PEP 585 泛型语法。JetPack 5 宿主机是 Python 3.8，原生不支持泛型下标。  
**解法**：批量注入 `from __future__ import annotations`，让 Python 3.8 将类型注解当作字符串延迟求值：
```bash
find jetson_containers/ -name "*.py" -exec sed -i '1s/^/from __future__ import annotations\n/' {} +
```

---

### 坑点 3：54GB eMMC 空间耗尽与 /dev/shm 内存盘自救

**现象**：执行 `docker load -i model.tar.gz` 导入 6GB 压缩包时，解压到 95% 中断：
```text
write .../googlenet.caffemodel: no space left on device
```
`df -h` 显示系统盘只剩 18GB，但 `/dev/shm` 有 31GB 空闲。  
**根因**：板载 54GB eMMC 既要放 8GB 模型权重，又要放 6GB 压缩包，解压时还需 15GB 临时空间，三重占用撑爆系统盘。  
**解法**：Orin 64GB 统一内存意味着 `/dev/shm` 是纯内存文件系统，有 31GB 空闲。把压缩包移入内存盘，瞬间释放系统盘空间：
```bash
# 1. 将压缩包移入 31GB 内存盘（系统盘瞬间释放空间）
mv ~/Desktop/pantograph-subsystem-jetson-v2.0.tar.gz /dev/shm/

# 2. 从内存盘极速导入镜像到 Docker（读写速度达数 GB/s）
docker load -i /dev/shm/pantograph-subsystem-jetson-v2.0.tar.gz

# 3. 导入完成后删除内存文件，归还 RAM
rm -f /dev/shm/pantograph-subsystem-jetson-v2.0.tar.gz
```

---

### 坑点 4：PyTorch 2.1.0a0 假版本与 Alpha 算子缺失

**现象**：安装了 JetPack 5 专用的官方 PyTorch 2.1 轮子后，运行新版 Transformers 遭到版本拦截：
```text
UserWarning: Disabling PyTorch because PyTorch >= 2.1 is required but found 2.1.0a0+413****1538.nv23.06...
```
**根因**：Alpha 预览版（`2.1.0a0`）在 PEP 440 版本规则中小于正式版（`2.1.0`），且 `2.1.0a0` 缺失 Qwen-VL 计算 3D-mROPE 所需的动态切片算子。  
**解法**：如使用 Qwen2-VL，降级 Transformers 到 `4.44.0`；若需高版本，使用 jetson-containers 官方预构建镜像隔离。

---

### 坑点 5：PyPI 自动依赖解析拉错轮子

**现象**：容器内执行 `pip install accelerate`，终端突然开始下载通用 ARM 轮子（`torch-2.13.0-cp310-manylinux_2_28_aarch64.whl`），安装后 `CUDA available: False`。  
**根因**：pip 从 PyPI 拉取了针对云端通用 ARM 服务器（如 AWS Graviton）编译的包，无法识别 Jetson Tegra 驱动。  
**解法**：从 requirements.txt 过滤掉底层加速库，单独安装 NVIDIA 原厂 Jetson 专用 Wheel：
```bash
grep -ivE "^torch|^torchvision|^torchaudio|^tensorrt|^pycuda" requirements.txt > requirements_safe.txt
pip install --no-cache-dir torch-xxx-nv-linux_aarch64.whl
pip install -r requirements_safe.txt
```

---

### 坑点 6：Transformers 与 Qwen 系列的代际版本锁

**现象**：在 `transformers 4.46.3` 环境下强行加载 Qwen3-VL 报错：`ImportError: cannot import name 'Qwen3VLForConditionalGeneration'`。  
**根因**：Qwen3-VL 在 `transformers >= 4.57.0` 中才被并入，而 4.57+ 强制依赖 Python 3.10+ 与 PyTorch 2.3+。  
**解法**：遵循兼容性矩阵，在 JetPack 5 环境下运行 Qwen2-VL-2B/7B，或通过升级 JetPack 6 + vLLM 容器跑 Qwen3-VL。

---

### 坑点 7：多路 4K RTSP 拉流导致 FFmpeg 管道死锁与僵尸/孤儿进程失控

大模型跑通只是第一步。在车载边缘端，大模型必须实时消费来自车顶多路工业相机的 **4K @ 25fps RTSP 视频流**（受电弓全景、滑板微距、接触网侧视）。而在多路并发接入与端到端压力测试中，音视频流接入层成为了最隐蔽的“系统吞噬者”。

#### 1. 仿真压测床搭建：EasyDarwin + FFmpeg 闭环推流

动车组实车调试窗口极其稀缺且成本高昂。为了在实验室与工控机上实现 1:1 闭环压测，我们搭建了基于 **EasyDarwin 流媒体服务器 + FFmpeg 循环推流** 的仿真测试环境：

```bash
# 1. 极速启动 EasyDarwin RTSP 服务器容器
docker run -d --name easydarwin -p 554:554 -p 10008:10008 easydarwin/easydarwin:latest

# 2. FFmpeg 按真实帧率无限循环推流 (模拟车顶 4K 工业相机 RTSP 输出)
ffmpeg -re -stream_loop -1 \
  -i pantograph_4k_test.mp4 \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:554/live/camera_01
```

借助该仿真基准，我们不仅解耦了硬件依赖，还能通过注入弱网丢包和突发断流，对系统进行 7×24 小时高压测试。正是在此过程中，多路拉流的深层系统级故障彻底爆发。

#### 2. 故障现象与进程监控分析

在 4 路 4K 工业相机长时间运行与模拟断网重连测试中，执行 `ps -ef | grep ffmpeg` 观察到严重的进程异化现象：

```text
hirain   10421  9820  0 14:20 ?   00:00:00 [ffmpeg] <defunct>   # 僵尸进程 1
hirain   10455  9820  0 14:21 ?   00:00:00 [ffmpeg] <defunct>   # 僵尸进程 2
hirain   11204     1 95 14:25 ?   01:12:00 ffmpeg -rtsp_transport... # 孤儿进程 (PPID=1)
```

系统表现出三重致命症状：
1. **拉流主循环假死（Hanging）**：Python 图像接收端没有抛出任何 Exception 崩溃，但读取图像队列完全停滞，帧率跌至 0；
2. **僵尸与孤儿进程爆发**：断流重连或重启拉流 Worker 后，残留数十个 `<defunct>` 僵尸进程以及脱离管制的孤儿 FFmpeg 进程；
3. **系统算力与内存雪崩**：孤儿 FFmpeg 进程在后台脱缰狂转（PPID=1），12 核 CPU 占用率飙升至 100%，上层大模型推理主进程频繁被系统的 OOM Killer 强杀。

#### 3. 根因深度剖析

| 故障现象 | 对应代码位置 | 底层根因剖析 |
| :--- | :--- | :--- |
| **僵尸进程 (`<defunct>`)** | `_run_loop` 中的 `subprocess.Popen` | FFmpeg 内部会 fork 多个编解码子进程，但 Python 主程序默认只持有直接子进程的引用。当 FFmpeg 内部子进程先退出时，没有父进程调用 `waitpid()` 回收其退出状态，导致进程描述符滞留内核成为僵尸。 |
| **孤儿进程 (`PPID=1`)** | `stop()` 与流异常重启 | 普通 `process.kill()` 仅向直接子进程发送信号，FFmpeg fork 出的子进程/孙进程无法被直接杀死，在主进程退出或重启时逃逸并被 `init` (PID 1) 收养，继续占用 GPU/CPU 狂转。 |
| **管道缓冲区死锁 (Pipe Full)** | `subprocess.Popen(stderr=PIPE)` | Python 开启 `stderr=subprocess.PIPE` 却只在 `stdout` 读图像帧，忽略了 `stderr` 日志。Linux 默认管道缓冲区仅 **64KB**，4K 高码率拉流下几分钟内被填满，导致 FFmpeg 底层 `write()` 永久阻塞挂起。 |
| **优雅停机失效** | `stdin.write(b'q')` | FFmpeg 需要 stdin 是交互式 TTY 或持续可读才能响应按键 `q`，在非交互子进程管道模式下极不可靠。 |
| **异常清理静默失效** | `_cleanup_orphan_ffmpeg` | 脚本内部引用了 `sys.platform` 但文件头**遗漏了 `import sys`**，导致触发 `NameError` 异常，而外层又被 `except Exception: pass` 静默吞掉，使孤儿清理机制形同虚设！ |

#### 4. 破局：工业级 5 重递进式进程生命周期治理架构

针对上述顽疾，我们在工业拉流录制核心模块（`segment_recorder.py`）中重构了完整的流生命周期守护机制：

```python
import os
import sys
import signal as _signal
import threading
import subprocess
import atexit
import time

# 1. 全局活跃进程组注册表：用于 atexit 兜底强力清理
_active_pgids = set()
_active_pgids_lock = threading.Lock()

def _cleanup_all_active_pgids():
    """进程退出时的最终安全网：强制杀死所有遗留的 FFmpeg 进程组"""
    with _active_pgids_lock:
        for pgid in list(_active_pgids):
            try:
                os.killpg(pgid, _signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        _active_pgids.clear()

atexit.register(_cleanup_all_active_pgids)


class IndustrialRTSPWorker:
    def __init__(self, cam_cfg: dict):
        self.cam_cfg = cam_cfg
        self.process = None
        self._process_pgid = None
        self.lock = threading.Lock()

    def _kill_process_group(self, sig=_signal.SIGTERM):
        """向整个进程组发送信号，确保 FFmpeg 及其所有子孙进程全部被杀掉"""
        pgid = self._process_pgid
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def _unregister_pgid(self):
        """从全局注册表中注销进程组"""
        pgid = self._process_pgid
        if pgid is not None:
            with _active_pgids_lock:
                _active_pgids.discard(pgid)
            self._process_pgid = None

    def _reap_zombies(self):
        """主动收割所有已经退出的僵尸子进程 (<defunct>)"""
        if sys.platform.startswith("win"):
            return
        while True:
            try:
                # WNOHANG: 非阻塞回收任意已终止的子进程
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid <= 0:
                    break
            except (ChildProcessError, OSError):
                break

    def start_stream(self):
        env = os.environ.copy()
        env["TZ"] = "Asia/Shanghai"
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",          # 彻底关闭冗余日志，防止 stderr 溢出死锁
            "-rtsp_transport", "tcp",      # 强制 TCP 传输，杜绝 UDP 丢包花屏
            "-stimeout", "5000000",       # 5秒网络/读帧超时熔断机制
            "-i", self.cam_cfg["rtsp_url"],
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        with self.lock:
            # 核心机制 1：start_new_session=True (底层调用 os.setsid())
            # 让 FFmpeg 在独立的新进程组中运行，确保后续能通过 os.killpg 一网打尽
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # 核心机制 2：重定向至 DEVNULL，杜绝管道写满死锁
                stdin=subprocess.PIPE,
                env=env,
                start_new_session=True
            )
            # 记录 PGID 并加入全局注册表
            self._process_pgid = os.getpgid(self.process.pid)
            with _active_pgids_lock:
                _active_pgids.add(self._process_pgid)

    def stop_stream(self):
        """5 层阶梯式平滑关闭与进程树彻底清理"""
        with self.lock:
            # Level 1: 尝试向 stdin 写入 'q' 触发 FFmpeg 正常转码收尾
            if self.process and self.process.poll() is None:
                try:
                    if self.process.stdin:
                        self.process.stdin.write(b'q')
                        self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

            # Level 2: 短暂等待自行优雅退出 (2s)
            if self.process and self.process.poll() is None:
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

            # Level 3: 发送 SIGTERM 优雅终止整个进程组
            if self.process and self.process.poll() is None:
                self._kill_process_group(_signal.SIGTERM)
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

            # Level 4: 发送 SIGKILL 强杀整个进程组（绝不给孤儿进程逃逸机会）
            if self.process and self.process.poll() is None:
                self._kill_process_group(_signal.SIGKILL)
                try:
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass

            # Level 5: process.kill() 与注销清理
            if self.process and self.process.poll() is None:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass

            self._unregister_pgid()
            self.process = None

        # 核心机制 3：主动收割僵尸进程，归还系统内核资源
        self._reap_zombies()
```

#### 5. 治理收益总结
1. **0 僵尸残留**：引入 `_reap_zombies()`（基于 `os.waitpid(-1, WNOHANG)`），在每次断流、重连和停机后自动收割退出进程，彻底消除 `<defunct>`；
2. **0 孤儿逃逸**：通过 `start_new_session=True` 将 FFmpeg 隔离在独立进程组，停止或异常重启时调用 `os.killpg(pgid, SIGKILL)` 进行整树销毁；
3. **0 管道死锁**：显式指定 `-loglevel quiet` 并将 `stderr=subprocess.DEVNULL`，阻断 64KB 管道缓冲区溢出；
4. **全局异常安全**：借助 `atexit` 全局注册表，即便主 Python 程序遭受未捕获异常崩溃，也能在退出阶段强力清理所有后台流媒体进程。

---

## 四、车载存储自治：端-车-NAS 分级存储与 80%➔60% 双水位淘汰闭环

在高速动车组车载环境中，AI 系统面临着另一个常被忽视的物理红线——**工控机本地存储极度紧缺与视频/诊断大图海量爆发的结构性矛盾**：
- **工控机存储告急**：Jetson Orin AGX 板载 eMMC 仅 54GB（扣除 Ubuntu OS、CUDA 驱动、Docker 镜像与大模型权重后，可用空间不足 10GB）；
- **数据吞吐巨大**：4 路车顶 4K 工业相机连续录制切片视频，配合 VLM 阶段性抓拍的 4K 结构化诊断原图，单日产生的数据量高达数百 GB。若直接堆积在工控机本地，几小时内便会引发 `No space left on device`，导致 Docker 崩溃、TensorRT 引擎反序列化失败以及系统死锁。

为此，我们在 `ftp_uploader.py` 中设计了 **「端-车-NAS 分级存储架构与容量自治管理机制」**：

```text
┌─────────────────────────────────────────────────────────────┐
│  Jetson Orin AGX 车载工控机 (本地存储零积压)                │
│  - 4K 切片视频生成 (segment_recorder.py)                    │
│  - VLM 阶段诊断大图生成 (Algo-result)                      │
│  - 异步上传 Worker: upload_and_delete()                     │
│  - 上传成功即刻原子删除本地文件 ──► 本地磁盘使用率始终 < 10%│
└──────────────────────────────┬──────────────────────────────┘
                               │ FTP (被动模式 PASV / 线程安全连接池)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  车载 NAS 集中存储 (720GB NVMe/SATA 阵列)                   │
│  - 运行目录: /nvme1/Algo-result                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 80% 高水位触发 (576GB) ──► 启动 FIFO 时间戳递归淘汰   │  │
│  │                                 │                     │  │
│  │                                 ▼                     │  │
│  │ 60% 低水位回落 (432GB) ──► 停止淘汰并递归修剪空目录   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1. 核心机制一：`upload_and_delete` 本地零积压外溢
工控机本地只做短暂的内存与切片中转。一旦文件生成完毕，异步工作线程立即将其推送到车载 NAS，并严格校验上传结果。只有确认 NAS 完整接收并落盘后，才原子执行 `os.remove(local_path)` 释放本地 inode 与数据块：

```python
def upload_and_delete(self, local_path: str, remote_path: str) -> bool:
    """上传到 NAS 成功后立即删除容器内本地文件，确保工控机本地零积压"""
    ok = self.upload_file(local_path, remote_path)
    if not ok:
        return False
    try:
        os.remove(local_path)
        print(f"[FTP] 已删除容器本地文件: {local_path}")
        return True
    except OSError as e:
        print(f"[FTP] 上传成功但删除本地失败: {local_path}, 错误: {e}")
        return False
```

### 2. 核心机制二：NAS 远程 80% ➔ 60% 双水位 FIFO 容量淘汰算法
车载 NAS（如 720GB）并非无限容量，在长期 7×24 小时无人值守运行下同样会写满。传统方案往往依赖外部数据库记录文件生命周期，但在车载恶劣断电环境下数据库极易损坏。我们设计了**基于文件路径天然时间戳排序的无数据库自治淘汰引擎**：

```python
class FtpUploader:
    def __init__(self, ftp_cfg: dict):
        self.cleanup_threshold = float(ftp_cfg.get("cleanup_threshold", 0.80))       # 80% 触发清理
        self.cleanup_target_ratio = float(ftp_cfg.get("cleanup_target_ratio", 0.60)) # 60% 清理目标
        self.max_storage_size_gb = float(ftp_cfg.get("max_storage_size_gb", 720.0))  # NAS 容量 720GB
        self.clean_paths = ftp_cfg.get("clean_paths", ["nvme1/Algo-result"])
        self._lock = threading.Lock()

    def run_remote_capacity_cleanup(self):
        """按容量双水位阈值自治清理 NAS 远程磁盘"""
        max_bytes = int(self.max_storage_size_gb * (1024 ** 3))
        cleanup_threshold_bytes = max_bytes * self.cleanup_threshold   # 576 GB
        cleanup_target_bytes = max_bytes * self.cleanup_target_ratio     # 432 GB

        with self._lock:
            ftp = self._connect()
            try:
                for base_clean_path in self.clean_paths:
                    all_files = []
                    # 递归遍历远程 FTP 目录收集文件大小与路径
                    self._collect_ftp_files_recursive(ftp, base_clean_path, all_files)
                    
                    total_bytes = sum(f["size"] for f in all_files)
                    if total_bytes >= cleanup_threshold_bytes:
                        # 核心设计：路径天然包含时间戳 (如 .../2026-08-27/142000_cam01.mp4)
                        # 直接按路径升序排序，严格保证最旧的文件排在最前面进行 FIFO 淘汰
                        all_files.sort(key=lambda x: x["path"])

                        for file_info in all_files:
                            if total_bytes <= cleanup_target_bytes:
                                break  # 成功回落至 60% 以下，立即终止淘汰
                            try:
                                ftp.delete("/" + file_info["path"])
                                total_bytes -= file_info["size"]
                            except Exception as e:
                                print(f"[FTP-Clean] 删除远程文件失败: {e}")

                        # 递归深度修剪无文件的空日期目录
                        self._remove_empty_ftp_dirs(ftp, base_clean_path)
            finally:
                ftp.quit()
```

### 3. 工业级设计细节与收益
1. **轻量无状态（Stateless & Zero-DB）**：无需引入 Redis/MySQL，直接利用规范化的时间目录路径（`YYYY-MM-DD/HHMMSS.mp4`）完成字符串级自然时序排序，天然抵御非正常断电导致的数据索引损坏；
2. **防网络与防火墙阻断**：显式配置 `set_pasv(True)` 被动模式传输，适配车载专用交换机与网关防火墙的端口隔离策略；
3. **空目录级联修剪（`_remove_empty_ftp_dirs`）**：淘汰文件的同时递归向上删除空的日期目录，防止几万个空文件夹造成 NAS 系统的 `inode` 耗尽与文件检索变慢；
4. **端到端 7×24h 零维护闭环**：工控机磁盘占用率常年稳定在 8% 以下，NAS 容量常年维持在 60%~80% 的健康动态平衡区间。

---

## 五、选型：三条落地路线

![JetPack 5 宿主机三条落地路线](/images/blog/three-routes.svg)
*决策维度只有一个：这台车载工控机，允不允许动它的操作系统。*

1. **路线 A（推荐长线演进）**：升级 JetPack 6.1 + vLLM 官方镜像，享受 PagedAttention 算子加速，Qwen3-VL 毫秒级推理；
2. **路线 B（免刷机高精度）**：编译 llama.cpp，指定 Ampere 架构硬加速（`CUDA_ARCHITECTURES=87`），显存仅需 3.5GB；
3. **路线 C（保交付快速上线）**：`transformers 4.46.3` + `Qwen2-VL-2B` 原生容器部署。

---

## 六、承上启下：从硬件避坑走向算法与推理攻坚

解决了边缘端“能不能跑”的基础环境问题后，我们迎来了真正的工业级核心挑战：

1. **算法层面**：大模型单帧推理存在 5%~10% 的光影误报，如何设计**时序一致性滤波算法**将误报率压降至接近 0？
2. **性能层面**：在车载工控机上，如何将滑动窗口端到端诊断时延从 **5.5 秒极限压缩至 1.06 秒**？

👉 **请继续阅读系列第二篇：[《动车组受电弓开集缺陷检测：动静分离两阶段 VLM 架构与跨窗口时序一致性滤波实战》](/blog/pantograph-vlm-twostage-algorithm)**！
