---
layout: ../../../layouts/BlogPost.astro
title: "Hardware Video Encoding and Decoding on NVIDIA Jetson Orin"
date: "2026-09-28"
lastUpdated: "2026-09-29"
author: "Jing Frank"
authorTitle: "Edge AI Engineer"
ogImage: "/images/blog/edge-video-encoding-benchmark.svg"
tags: ["Jetson", "GStreamer", "Edge Computing", "Hardware Acceleration"]
---

## Table of Contents

- [Introduction](#introduction)
- [1. Background and Problem](#1-background-and-problem)
  - [1.1 Video Processing Challenges in Edge Computing Scenarios](#11-video-processing-challenges-in-edge-computing-scenarios)
    - [1.1.1 Typical Business Process](#111-typical-business-process)
    - [1.1.2 Business Classification of Video Segments](#112-business-classification-of-video-segments)
- [2. Encoding and Decoding Hardware Architecture of Jetson Orin](#2-encoding-and-decoding-hardware-architecture-of-jetson-orin)
  - [2.1 NVENC: Hardware Video Encoder](#21-nvenc-hardware-video-encoder)
    - [2.1.1 NVENC Architecture](#211-nvenc-architecture)
  - [2.2 NVDEC: Hardware Video Decoder](#22-nvdec-hardware-video-decoder)
    - [2.2.1 NVDEC Architecture](#221-nvdec-architecture)
- [3. Jetson Orin Software Stack and API](#3-jetson-orin-software-stack-and-api)
  - [3.1 Official Four-Layer Interface System and GStreamer Selection](#31-official-four-layer-interface-system-and-gstreamer-selection)
  - [3.2 NVIDIA Official Multimedia API](#32-nvidia-official-multimedia-api)
    - [3.2.1 Jetson Multimedia API (NvMM)](#321-jetson-multimedia-api-nvmm)
  - [3.3 FFmpeg: Hardware-Accelerated Encoding and Decoding](#33-ffmpeg-hardware-accelerated-encoding-and-decoding)
    - [3.3.1 Principle of FFmpeg Hardware Acceleration](#331-principle-of-ffmpeg-hardware-acceleration)
  - [3.4 GStreamer: Streaming Media Processing Framework](#34-gstreamer-streaming-media-processing-framework)
    - [3.4.1 GStreamer Plugins on Jetson Orin](#341-gstreamer-plugins-on-jetson-orin)
    - [3.4.2 GStreamer Hardware Acceleration Pipeline Examples](#342-gstreamer-hardware-acceleration-pipeline-examples)
  - [3.5 NVIDIA DeepStream SDK](#35-nvidia-deepstream-sdk)
- [4. Practical Application: Building Efficient Video Processing Pipelines](#4-practical-application-building-efficient-video-processing-pipelines)
  - [4.1 RTSP Stream Hardware Decoding + OpenCV Processing (based on GStreamer)](#41-rtsp-stream-hardware-decoding-opencv-processing-based-on-gstreamer)
    - [4.1.1 Scenario Description](#411-scenario-description)
    - [4.1.2 Solution Comparison](#412-solution-comparison)
    - [4.1.3 GStreamer Pipeline Design](#413-gstreamer-pipeline-design)
    - [4.1.4 Python Implementation](#414-python-implementation)
  - [4.2 OpenCV Generated Video + Hardware Encoding to H.264 (Zero-Copy Optimization)](#42-opencv-generated-video-hardware-encoding-to-h264-zero-copy-optimization)
    - [4.2.1 Scenario Description](#421-scenario-description)
    - [4.2.2 Solution Analysis](#422-solution-analysis)
    - [4.2.3 Implementation Solution](#423-implementation-solution)
- [5. Performance Comparison Analysis](#5-performance-comparison-analysis)
  - [5.1 Frontend Decoding Performance Comparison](#51-frontend-decoding-performance-comparison)
    - [5.1.1 Test Scenario](#511-test-scenario)
    - [5.1.2 Key Data Comparison](#512-key-data-comparison)
    - [5.1.3 Technical Interpretation](#513-technical-interpretation)
    - [5.1.4 Applicable Scenario Suggestions](#514-applicable-scenario-suggestions)
  - [5.2 Backend Encoding Performance Comparison](#52-backend-encoding-performance-comparison)
    - [5.2.1 Test Scenario](#521-test-scenario)
    - [5.2.2 Key Data Comparison](#522-key-data-comparison)
    - [5.2.3 Technical Interpretation](#523-technical-interpretation)
    - [5.2.4 Applicable Scenario Suggestions](#524-applicable-scenario-suggestions)

## Introduction

In the wave of edge computing, **real-time video processing** has become a core requirement in rail transit, smart transportation, and industrial quality inspection. Imagine: a train speeding on the tracks, where the on-vehicle monitoring system needs to **detect foreign objects on the tracks within milliseconds** and trigger emergency braking; or on an industrial production line, where the system must **identify defects in every product in real time** to ensure zero defective products leave. In these scenarios, **video encoding and decoding** is no longer a simple multimedia task but a key link determining the system's **performance, power consumption, and stability**.

However, in early development practices, many engineers habitually use **OpenCV's default interfaces** (such as `cv2.VideoCapture` and `cv2.VideoWriter`) to handle video streams. This approach runs well on **personal computers or lab environments**, but once deployed on **resource-constrained embedded industrial computers** (such as NVIDIA Jetson Orin), it exposes **system-level conflicts**:

- **CPU usage spikes to 100%**, causing AI inference frame rates to drop sharply.
- **Memory copy overhead is too large**, and latency fails to meet millisecond-level requirements.
- **Cannot utilize hardware encoding and decoding units** (NVENC/NVDEC), making power consumption and performance hard to balance.
- **Poor multi-stream concurrency**, unable to support 4K/8K multi-stream video processing.

These problems directly lead to **high latency, power consumption exceeding limits, poor stability**, and even **frame loss or crashes at critical moments**, bringing **safety hazards and economic losses** to actual applications.

For these pain points, the **NVIDIA Jetson Orin series** provides **hardware-level video encoding and decoding capabilities** (NVENC/NVDEC), achieving **high concurrency and high stability** in video processing under the premise of **low power consumption and low latency**. But how to **correctly use** these hardware units, how to **replace OpenCV's default solution**, and how to **optimize the end-to-end video pipeline** remain challenges for many developers.

**This is exactly the core goal of this article**:

- ✅ **Deeply analyze** the hardware encoding and decoding architecture of Jetson Orin (principles, performance, interfaces of NVENC/NVDEC).
- ✅ **Practically demonstrate** how to use **FFmpeg and GStreamer** to replace OpenCV and build efficient video processing pipelines.
- ✅ **Optimization tips** on how to use **zero-copy, multi-stream concurrency, intelligent segmentation** and other means to improve system performance.
- ✅ **Troubleshooting** how to deal with common **compatibility, stability, and power consumption** problems during encoding and decoding.

## 1. Background and Problem

### 1.1 Video Processing Challenges in Edge Computing Scenarios

In edge computing scenarios such as **rail transit on-vehicle monitoring, smart transportation, and industrial quality inspection**, video data processing needs have three major characteristics: **high real-time performance, high reliability, and high concurrency**. Taking rail transit as an example, the on-vehicle monitoring system needs to complete defect detection (such as track cracks, foreign object intrusion, equipment failure, etc.) within **milliseconds** and feed the results back to the driver or the backend in real time to ensure driving safety.

#### 1.1.1 Typical Business Process

The tasks undertaken by the edge computing host in the above scenarios can be abstracted as an **end-to-end video processing pipeline**, as shown in the following figure:

```mermaid
flowchart LR
    A[Camera RTSP] -->|1. Video stream acquisition| B[Video decoding]
    B -->|2. Raw frame data| C[Algorithm inference]
    C -->|3. Detection results| D[Result rendering HUD]
    D -->|4. Overlay OSD| E[Video encoding]
    E -->|5. Encoded stream| F[Disk segmentation]
    E -->|5. Encoded stream| G[Network transmission]
```

<p align="center"><em>Figure 1: Edge-side video and algorithm processing pipeline</em></p>

**Process analysis**:

1. **Camera RTSP**: high-definition industrial cameras (such as 4K@30fps) transmit video streams to the edge device via RTSP/RTMP protocols.
2. **Video decoding**: decode compressed video streams (such as H.264/H.265) into raw frame data (YUV/RGB) for subsequent algorithm processing.
3. **Algorithm inference**: run **YOLOv8, Faster R-CNN** and other object detection models, or **U-Net, Segment Anything** and other segmentation models to identify defects or targets.
4. **Result rendering HUD**: overlay detection results (such as bounding boxes, confidence, alert information) onto the original frame to form a **HUD (Head-Up Display)**, displayed to the operator in real time.
5. **Video encoding**: re-encode the overlaid frames to H.264/H.265 format to reduce storage and transmission bandwidth.
6. **Disk segmentation/network transmission**:
   - **Disk segmentation**: store the encoded video into **segment files** (such as `.mp4`/`.ts`) by time or event, for subsequent analysis or evidence collection.
   - **Network transmission**: push the encoded stream to a backend server or cloud (such as a Kubernetes cluster) for centralized storage or further analysis.

#### 1.1.2 Business Classification of Video Segments

In the above pipeline, **video segmentation** is a key link, usually corresponding to the following two types of business tasks:

| **Segment Type** | **Trigger Condition** | **Storage Strategy** | **Purpose** | **Example Scenario** |
| ---------------- | ------------------------ | ------------------------ | ---------------- | ---------------- |
| **Routine inspection segment** | Time-based trigger (e.g., every 5 minutes) | Time-based segmentation (e.g., 5 min/segment) | Post-event review, daily monitoring | Rail transit on-vehicle monitoring, industrial inspection |
| **Event evidence segment** | Algorithm detects anomaly (e.g., defect) | Event-based segmentation (including 10 seconds before and after the anomaly) | Accident analysis, liability determination | Track crack detection, traffic violation capture |

**Example**:

- In **rail transit**, the system will:
  - Generate a **routine inspection segment** every **5 minutes** (such as `20261109_100000_100500.mp4`) for daily inspection.
  - When **track cracks** are detected, automatically trigger an **event evidence segment** (such as `20261109_100320_crack_001.mp4`), containing 10 seconds of video before and after the crack appears, for maintenance personnel analysis.
- In **industrial quality inspection**, the system will:
  - **Continuously store** videos on the production line, but only retain complete segments when **defective products** are detected; other videos can be overwritten (saving storage space).

## 2. Encoding and Decoding Hardware Architecture of Jetson Orin

This chapter will introduce Jetson Orin series **NVENC (hardware encoder) and NVDEC (hardware decoder)**, including their architecture, performance, supported encoding/decoding standards, and collaboration mechanism with GPU and CPU.

The **GPU core count, memory size, and encoding/decoding capabilities** of Jetson AGX Orin are as follows:

| **Model** | **GPU Architecture** | **CUDA Core Count** | **CPU** | **Memory** | **NVENC Count** | **NVDEC Count** | **Max Encoding Resolution** | **Max Decoding Resolution** | **Power** |
| ----------- | ----------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
| **Jetson AGX Orin** | Ampere | 8192 | 12-core ARM Cortex-A78AE | 64GB LPDDR5 | 3 | 4 | 8K@60fps | 8K@60fps | 45W/60W |

> **Note**:
>
> - **NVENC count**: the number of hardware units that can encode simultaneously. For example, Jetson AGX Orin can encode **3 streams of 8K@30fps** at the same time.
> - **NVDEC count**: the number of hardware units that can decode simultaneously. For example, Jetson AGX Orin can decode **4 streams of 8K@30fps** at the same time.
> - **Maximum resolution**: depends on the bandwidth of the hardware units and the GPU memory bandwidth.

### 2.1 NVENC: Hardware Video Encoder

#### 2.1.1 NVENC Architecture

NVENC (NVIDIA Encoder) is a **dedicated hardware encoding unit** in NVIDIA GPUs, used to encode raw video frames into compressed formats (such as H.264/H.265). The NVENC architecture includes the following key components:

1. **Motion Estimation unit (Motion Estimation, ME)**:
   - Hardware-accelerated motion vector search, used to reduce inter-frame redundancy.
   - Supports **multiple search algorithms** (such as Diamond Search, Hexagon Search).
2. **DCT/IDCT unit**:
   - Hardware-accelerated **Discrete Cosine Transform (DCT)** and **Inverse Discrete Cosine Transform (IDCT)**, used to convert spatial-domain pixel blocks into frequency domain.
3. **Entropy Coding unit**:
   - Hardware-accelerated **CABAC (Context-Adaptive Binary Arithmetic Coding)**, used to further compress data.
4. **Bitstream output unit**:
   - Pack encoded data into standard bitstreams (such as H.264/H.265 NAL units).

### 2.2 NVDEC: Hardware Video Decoder

#### 2.2.1 NVDEC Architecture

NVDEC (NVIDIA Decoder) is a **dedicated hardware decoding unit** in NVIDIA GPUs, similar to NVENC, it achieves efficient video decoding through hardware acceleration. The NVDEC architecture includes the following key components:

1. **Decode Engine**:
   - Hardware-accelerated decoding core, supporting multiple video encoding standards (such as H.264, H.265, VP9, etc.).
   - Can process multiple decoding tasks in parallel, improving concurrency capability.
2. **Post-Processing unit**:
   - Supports **Deblocking**, **Denoising**, **Color space conversion (YUV to RGB)**, and other operations.
   - Can directly output **frames in GPU memory**, avoiding CPU participation.
3. **Memory Controller**:
   - Manages data interaction with **Unified Memory**, supporting zero-copy technology.
   - Ensures decoded frames can be used directly for GPU computation (such as AI inference).

## 3. Jetson Orin Software Stack and API

In the previous chapter, we deeply explored the hardware encoding and decoding architecture of Jetson Orin, including the principles, performance, and concurrency capability of NVENC and NVDEC. However, realizing hardware capabilities depends on **software stack support**. This chapter will introduce the **software stack and APIs** on Jetson Orin, including **NVIDIA official multimedia API, FFmpeg, GStreamer, Jetson Multimedia API**, etc., and provide specific code examples to help developers efficiently call hardware encoding and decoding capabilities in actual projects.

### 3.1 Official Four-Layer Interface System and GStreamer Selection

In an [NVIDIA official technical blog](https://developer.nvidia.com/blog/nvidia-jetpack-7-2-1-adds-agentic-video-skills-and-t3000-emulation/) technical review (Table 1) for JetPack video workflows, the Jetson platform provides four different levels of programming interfaces:

| Interface Name | Positioning and Control Granularity | Typical Application Scenario |
| ------------ | ---- | --- |
| **[GStreamer](https://gstreamer.freedesktop.org/)** | High-level pipeline assembly, modular multimedia graph | Suitable for complete applications needing to combine network pull streams, hardware transforms, filters, and container encapsulation (MP4/MKV) |
| **[V4L2 (Video4Linux2)](https://docs.kernel.org/userspace-api/media/v4l/v4l2.html)** | Low-level device and buffer control | Suitable for scenarios needing fine control over camera hardware device nodes, driver registers, or custom driver tuning |
| **[Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)** | C/C++ low-level direct call interface | For low-level micro parameter control of NVENC/NVDEC hardware features, network transmission and encapsulation need to be developed manually |
| **[PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec)** | Python GPU memory direct call interface | Specifically designed for AI training and inference, directly handing decoded frames to PyTorch as CUDA device pointers |

Under this division of labor system, GStreamer (high-level multimedia pipeline framework) is the optimal solution for this project scenario. The core of the selection decision is actual business complexity:

1. **Input side needs network fault tolerance**: RTSP pull streams not only involve H.264 decoding but also need to handle network packet loss, TCP/UDP protocol negotiation, and automatic reconnection.
2. **Middle layer has Python business logic**: algorithm detection results need to overlay locomotive number, carriage number, and other Chinese font watermarks, and temporarily store in memory circular buffers.
3. **Output side needs compliant container encapsulation**: generated accident evidence videos must be standard MP4 container format and directly parseable by ground recording servers.

If pure low-level C++ interfaces were used, the team would need to write a large amount of protocol parsing and timestamp reorganization code manually. Therefore, the multimedia components based on GStreamer are the most deterministic choice for engineering deployment.

### 3.2 NVIDIA Official Multimedia API

NVIDIA provides **proprietary multimedia APIs** for directly calling Jetson Orin's hardware encoding and decoding units. These APIs include **Jetson Multimedia API (also called NvMM API)** and **Argus API**, suitable for scenarios needing **maximum performance and lowest latency**.

#### 3.2.1 Jetson Multimedia API (NvMM)

**Jetson Multimedia API** is a **C/C++ API** provided by NVIDIA for accessing multimedia functions on Jetson platforms, including:

- **Hardware encoding/decoding** (NVENC/NVDEC)
- **Camera input** (through MIPI-CSI2 or GMSL)
- **Video rendering** (through EGL/OpenGL)

**Core components**:

| **Component** | **Role** | **Corresponding Hardware Unit** |
| --- | ---- | --- |
| **NvMM** | Multimedia memory management API, used to allocate and manage GPU memory | GPU memory |
| **NvVideo** | Video encoding/decoding API, supports NVENC/NVDEC | NVENC/NVDEC |
| **NvBuffer** | Buffer management API, used to share data between CPU and GPU | Unified memory |
| **Argus** | Camera API, used to obtain raw frames from MIPI-CSI2 cameras | ISP, camera interface |

### 3.3 FFmpeg: Hardware-Accelerated Encoding and Decoding

FFmpeg is an **open-source, cross-platform multimedia framework**, widely used for video **decoding, encoding, transcoding, streaming**, and other tasks. On Jetson Orin, FFmpeg can call **NVENC/NVDEC** through **NVIDIA's hardware acceleration plugins** (such as `h264_nvenc`, `hevc_nvenc`, `h264_cuvid`) to achieve efficient hardware encoding and decoding.

#### 3.3.1 Principle of FFmpeg Hardware Acceleration

FFmpeg supports hardware acceleration through the following ways:

1. **Encoding acceleration**:
   - Use **NVENC** FFmpeg plugins (such as `h264_nvenc`, `hevc_nvenc`) for hardware encoding.
   - Example: `-c:v h264_nvenc` means using H.264 hardware encoding.
2. **Decoding acceleration**:
   - Use **NVDEC** FFmpeg plugins (such as `h264_cuvid`, `hevc_cuvid`) for hardware decoding.
   - Example: `-c:v h264_cuvid` means using H.264 hardware decoding.
3. **Zero-copy**:
   - Enable CUDA hardware acceleration through `-hwaccel cuda` and `-hwaccel_device 0`, reducing memory copies.

### 3.4 GStreamer: Streaming Media Processing Framework

**GStreamer** is an **open-source streaming media framework**, widely used in **real-time video processing, streaming media transmission, multimedia applications**, and other scenarios. On Jetson Orin, GStreamer supports hardware-accelerated encoding and decoding through **NVIDIA's plugins** (such as `nvv4l2h264dec`, `nvv4l2h265enc`).

#### 3.4.1 GStreamer Plugins on Jetson Orin

| **Plugin** | **Type** | **Description** | **Corresponding Hardware Unit** | **Supported Formats** |
| --- | --- | ---- | --- | --- |
| `nvv4l2h264dec` | Decoder | H.264 hardware decoder | NVDEC | H.264 Baseline/Main/High |
| `nvv4l2h265dec` | Decoder | H.265 hardware decoder | NVDEC | H.265 Main/Main10 |
| `nvv4l2vp8dec` | Decoder | VP8 hardware decoder | NVDEC | VP8 |
| `nvv4l2vp9dec` | Decoder | VP9 hardware decoder | NVDEC | VP9 |
| `nvv4l2av1dec` | Decoder | AV1 hardware decoder (JetPack 5.1+) | NVDEC | AV1 Main |
| `nvv4l2h264enc` | Encoder | H.264 hardware encoder | NVENC | H.264 Baseline/Main/High |
| `nvv4l2h265enc` | Encoder | H.265 hardware encoder | NVENC | H.265 Main/Main10 |
| `nvv4l2av1enc` | Encoder | AV1 hardware encoder (JetPack 5.1+) | NVENC | AV1 Main |
| `nvvidconv` | Filter | Video format conversion (YUV↔RGB) | GPU | YUV420/RGB |
| `nvdsosd` | Filter | Overlay OSD on video (such as detection boxes) | GPU | RGB |
| `nvstreammux` | Filter | Multi-stream merging (for multi-stream video synchronization) | GPU | YUV420 |
| `nvstreamdemux` | Filter | Multi-stream separation | GPU | — |

#### 3.4.2 GStreamer Hardware Acceleration Pipeline Examples

The core advantage of GStreamer is its **modular pipeline architecture**, which can flexibly build complex video processing pipelines. Here are some typical **hardware acceleration pipeline examples** covering **decoding, AI inference, encoding, streaming**, and other scenarios.

### 3.5 NVIDIA DeepStream SDK

**NVIDIA DeepStream SDK** is a software development kit designed specifically for **real-time streaming media analysis**, particularly suitable for **Jetson platforms**. It integrates **hardware-accelerated video encoding/decoding, AI inference, multimedia processing**, and other functions, and can efficiently build **end-to-end video analysis pipelines**, such as:

- **Smart transportation** (vehicle/pedestrian detection, license plate recognition)
- **Security surveillance** (intrusion detection, face recognition)
- **Industrial quality inspection** (defect detection, product classification)
- **Medical imaging** (lesion detection, medical image analysis)

DeepStream SDK is based on **GStreamer and TensorRT**, providing **C/C++ API** and **Python API** (through PyDeepStream), supporting **zero-copy, multi-stream concurrency, low latency**, and other features.

## 4. Practical Application: Building Efficient Video Processing Pipelines

In the previous chapters, we deeply explored Jetson Orin's hardware architecture, software stack, and APIs. Starting from this chapter, we will demonstrate through **practical cases** how to apply this knowledge to actual projects. This chapter provides **two complete practical examples**, covering from **basic video processing** to **complex AI video analysis pipelines**, helping you build efficient, low-latency video processing systems on Jetson Orin.

### 4.1 RTSP Stream Hardware Decoding + OpenCV Processing (based on GStreamer)

#### 4.1.1 Scenario Description

**Requirements**:

- Read a **1080p@30fps** H.264 video stream from an RTSP camera.
- Use **hardware decoding (NVDEC)** to decode the video.
- Convert to BGR frames and hand to **OpenCV** or **AI inference module** for real-time image processing (such as object detection, anomaly detection, etc.).
- **Goal**: latency < 20ms, CPU usage < 10%.

**Application scenarios**:

- Smart surveillance (face detection, motion detection)
- Industrial inspection (defect identification, dimension measurement)
- Video analysis (color correction, denoising)

#### 4.1.2 Solution Comparison

| **Solution** | **Advantages** | **Disadvantages** | **Applicable Scenarios** |
| --- | --- | --- | --- |
| **Pure OpenCV** (`cv2.VideoCapture`) | Simple and easy to use | High CPU usage (90%+), high latency (50-100ms) | Lab prototype |
| **FFmpeg + OpenCV** | Hardware acceleration | Need manual memory copy management | Medium complexity applications |
| **GStreamer + appsink** | **Zero-copy, low latency** | Steeper learning curve | **Production environment** |

**Conclusion**: GStreamer + appsink is the best choice, because:
✅ **Hardware decoding** (NVDEC) significantly reduces CPU usage
✅ **Zero-copy** mechanism reduces memory copy overhead
✅ **Seamless integration with OpenCV** for subsequent processing
✅ **Low latency** meets real-time requirements

#### 4.1.3 GStreamer Pipeline Design

**Pipeline parsing**:

```bash
gst-launch-1.0 \
  rtspsrc location=rtsp://user:pass@192.168.1.100:554/stream latency=0 protocols=tcp drop-on-latency=true ! \
  rtph264depay ! h264parse ! \
  nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! \
  nvvidconv ! 'video/x-raw, format=BGRx' ! \
  appsink drop=1 max-buffers=1 sync=false
```

| **Component** | **Role** | **Key Parameters** |
| --- | ---- | --- |
| `rtspsrc` | RTSP stream source | `latency=0`: low latency mode<br>`protocols=tcp`: use TCP protocol (more stable)<br>`drop-on-latency=true`: drop frames when latency is too high |
| `rtph264depay` | RTP H.264 depayloader | — |
| `h264parse` | H.264 parser | — |
| `nvv4l2decoder` | **Hardware decoder (NVDEC)** | `enable-max-performance=1`: enable maximum performance mode<br>`drop-frame-interval=0`: do not drop frames |
| `nvvidconv` | Video format conversion | — |
| `video/x-raw, format=BGRx` | Set output format to BGRx (OpenCV compatible) | — |
| `appsink` | Pass data to application | `drop=1`: allow frame dropping<br>`max-buffers=1`: maximum buffer count<br>`sync=false`: asynchronous mode (reduce latency) |

> **Why choose BGRx format?**
>
> - OpenCV defaults to **BGR** format.
> - BGRx is a **4-channel format** (BGR + Alpha), compatible with OpenCV's `cv::Mat`.
> - Avoids additional format conversion overhead.

#### 4.1.4 Python Implementation

**Solution: using OpenCV + GStreamer**

```python
#!/usr/bin/env python3
import cv2

# GStreamer pipeline string
gstreamer_pipeline = (
    "rtspsrc location=rtsp://user:pass@192.168.1.100:554/stream latency=0 protocols=tcp drop-on-latency=true ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "appsink drop=1 max-buffers=1 sync=false"
)

# Open video stream
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Define processing function
def process_frame(frame):
    # Add your image processing logic here
    # For example: object detection, image enhancement, etc.
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Gaussian blur
    return frame

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process frame
        processed_frame = process_frame(frame)

        # Display results
        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

> **Advantages**:
>
> - Code is more concise and easy to understand.
> - Directly uses OpenCV's `VideoCapture` interface.
> - No need to manually manage GStreamer callbacks.

> **Disadvantages**:
>
> - May have **slightly additional latency** (because OpenCV internally performs some processing).

### 4.2 OpenCV Generated Video + Hardware Encoding to H.264 (Zero-Copy Optimization)

#### 4.2.1 Scenario Description

**Requirements**:

- Use **OpenCV** to generate or process **1920x1080@25fps** BGR format video frames.
- Convert frames to **NVMM memory format** (zero-copy) through **GStreamer**.
- Use **hardware encoder (NVENC)** to encode to H.264.
- Save as **MP4 file**.
- **Goal**: CPU usage < 10%, GPU usage < 20%, encoding latency < 5ms.

**Application scenarios**:

- AI video synthesis (such as generating videos, data augmentation)
- Encoding and storage after video processing (such as video with detection results)
- Real-time video recording (such as surveillance recording)

#### 4.2.2 Solution Analysis

**Why choose this solution?**

| **Solution** | **Advantages** | **Disadvantages** | **Applicable Scenarios** |
| --- | --- | --- | --- |
| **Pure OpenCV** (`cv2.VideoWriter`) | Simple and easy to use | CPU encoding (high usage), cannot use hardware acceleration | Lab prototype |
| **FFmpeg CLI** | Supports hardware acceleration | Need manual inter-process communication management | Command-line tool |
| **GStreamer + appsrc** | **Zero-copy, hardware acceleration** | Need to integrate OpenCV and GStreamer | **Production environment** |

**Core advantages**:
✅ **Zero-copy**: avoid CPU-GPU memory copying through `memory:NVMM`.
✅ **Hardware encoding**: use `nvv4l2h264enc` to significantly reduce CPU usage.
✅ **High performance**: encoding latency < 5ms, supporting real-time processing.
✅ **Flexibility**: seamless integration with OpenCV.

#### 4.2.3 Implementation Solution

**1. GStreamer Pipeline Design**

Pipeline parsing:

```bash
gst-launch-1.0 -e \
  appsrc is-live=false format=time ! 'video/x-raw, format=BGR, width=1920, height=1080, framerate=25/1' ! \
  queue ! videoconvert ! 'video/x-raw, format=BGRx' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM), format=NV12' ! \
  nvv4l2h264enc bitrate=4000000 ! \
  h264parse ! mp4mux ! \
  filesink location=output.mp4 sync=false
```

| **Component** | **Role** | **Key Parameters** |
| --- | --- | --- |
| `appsrc` | Application source, receives OpenCV frames | `is-live=false`: non-real-time mode<br>`format=time`: use timestamps<br>`video/x-raw, format=BGR`: input format is BGR |
| `queue` | Buffer queue, preventing blocking | — |
| `videoconvert` | Color space conversion (BGR → BGRx) | — |
| `nvvidconv` | GPU-accelerated format conversion | `video/x-raw(memory:NVMM), format=NV12`: convert to NVMM memory NV12 format |
| `nvv4l2h264enc` | **Hardware encoder (NVENC)** | `bitrate=4000000`: 4 Mbps bitrate |
| `h264parse` | H.264 parser | — |
| `mp4mux` | MP4 muxer | — |
| `filesink` | File output | `location=output.mp4`: output file path<br>`sync=false`: asynchronous mode (improve performance) |
| `-e` | Automatically close pipeline when GStreamer exits | — |

> **Key points**:
>
> - **`memory:NVMM`**: use NVIDIA's unified memory to achieve **zero-copy**.
> - **`format=NV12`**: optimal input format for NVENC.
> - **`is-live=false`**: non-real-time mode, suitable for file processing.

**2. Python Implementation (Complete Code)**

**Solution: using OpenCV + GStreamer**

```python
#!/usr/bin/env python3
import cv2
import numpy as np

# GStreamer pipeline string
gstreamer_pipeline = (
    "appsrc is-live=false format=time caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=25/1 ! "
    "queue ! videoconvert ! video/x-raw,format=BGRx ! "
    "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
    "nvv4l2h264enc bitrate=4000000 ! "
    "h264parse ! mp4mux ! "
    "filesink location=output.mp4 sync=false"
)

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(gstreamer_pipeline, cv2.CAP_GSTREAMER, 0, 25, (1920, 1080))

if not out.isOpened():
    print("Error: Could not open GStreamer pipeline")
    exit()

# Define frame generation function
def generate_frame(frame_count):
    # Create a 1920x1080 gradient frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Generate gradient effect
    for i in range(1080):
        color = int((i / 1080) * 255)
        frame[i, :, :] = [color, 255 - color, 128]

    # Add dynamic elements (such as moving circle)
    center_x = (frame_count * 10) % 1920
    cv2.circle(frame, (center_x, 540), 100, (0, 0, 255), -1)

    # Add text
    cv2.putText(frame, f"Frame: {frame_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

# Main loop
try:
    for frame_count in range(300):  # Generate 300 frames (12 seconds)
        frame = generate_frame(frame_count)
        out.write(frame)
        print(f"Writing frame {frame_count}")
finally:
    out.release()
    print("Video saved to output.mp4")
```

> **Advantages**:
>
> - Code is more concise and easy to understand.
> - Directly uses OpenCV's `VideoWriter` interface.
> - No need to manually manage GStreamer callbacks.

> **Disadvantages**:
>
> - May have **slightly additional latency** (because OpenCV internally performs some processing).

## 5. Performance Comparison Analysis

*(Frontend decoding (CPU software decoding vs NVDEC hardware decoding) + Backend encoding (CPU software encoding vs NVENC hardware encoding))*

### 5.1 Frontend Decoding Performance Comparison

#### 5.1.1 Test Scenario

- **Input source**: single-stream **1080P/4K H.264 RTSP network stream**.
- **Decoding solutions**:
  - **CPU software decoding**: OpenCV + FFmpeg (`libx264`/`libx265`).
  - **NVDEC hardware decoding**: GStreamer + `nvv4l2decoder`.
- **Statistical method**: run continuously and take the mean.

#### 5.1.2 Key Data Comparison

| **Metric** | **CPU Software Decoding** | **NVDEC Hardware Decoding** | **Optimization Gain** | **Analysis** |
| --- | --- | --- | --- | --- |
| **Decoding FPS** | 28 FPS | **120+ FPS** | ↑ **328%** | Hardware decoder has strong parallel processing capability; 4K streams can be decoded at line rate |
| **CPU Utilization** | 158% | **7.4%** | ↓ **95.3%** | CPU drops from full load to idle, freeing resources for post-processing |
| **Single-frame Decoding Latency** | 24.5 ms | **4.2 ms** | ↓ **82.9%** | Hardware acceleration has low latency, suitable for real-time systems |
| **CPU Memory Usage** | 1.82 GB | **0.65 GB** | ↓ **64.3%** | Hardware decoding reduces memory copies, lowering memory pressure |
| **4K Stability** | Occasional jitter | **Extremely stable** | Eliminate jitter | Hardware decoder timing is precise, no CPU scheduling interference |
| **Maximum stable stream count** | 2 streams | **8+ streams** | ↑ **300%** | NVDEC supports multi-stream concurrent decoding |

#### 5.1.3 Technical Interpretation

**Why does NVDEC hardware decoding perform excellently?**

1. **Dedicated hardware unit**:
   - NVDEC is a **dedicated video decoder** inside NVIDIA GPUs, supporting H.264/H.265/VP9 and other formats.
   - The decoding process is **completely completed by hardware**; the CPU only handles data transmission.
2. **Zero-copy mechanism**:
   - Frames decoded by hardware are directly saved in **GPU memory**, avoiding CPU memory copies.
   - Combined with `nvvidconv`, **zero-copy post-processing** (such as scaling, color conversion) is achievable.
3. **Multi-stream concurrency support**:
   - Jetson Orin's NVDEC supports **multi-stream 4K concurrent decoding** (such as 8 streams of 1080P or 4 streams of 4K).
   - CPU software decoding is limited by core count, usually **2 streams of 4K are already full load**.

**Limitations of CPU software decoding**

- **CPU bottleneck**: the decoding process occupies a lot of CPU resources, leading to **low frame rates and high latency**.
- **Memory pressure**: soft decoding needs frequent memory reads/writes, increasing **memory bandwidth usage** and **power consumption**.
- **Timing jitter**: CPU scheduling uncertainty leads to **large decoding latency fluctuations**.

#### 5.1.4 Applicable Scenario Suggestions

| **Scenario** | **Recommended Solution** | **Reason** |
| --- | --- | --- |
| **Real-time monitoring (single-stream 4K)** | NVDEC hardware decoding | Low latency, high FPS, stable timing |
| **Multi-stream concurrency (8+ streams)** | NVDEC hardware decoding | Strong hardware concurrency capability |
| **Low-power edge devices** | NVDEC hardware decoding | Low CPU usage, better power consumption |
| **High compatibility requirements** | CPU software decoding | Suitable for non-NVIDIA platforms |
| **Debug/development environment** | CPU software decoding | No hardware acceleration needed, easy to debug |

### 5.2 Backend Encoding Performance Comparison

#### 5.2.1 Test Scenario

- **Input source**: 200+ frames (about 7 seconds) 1080P@30FPS rendered frames.
- **Encoding solutions**:
  - **CPU software encoding**: FFmpeg (`libx264`/`libx265`).
  - **NVENC hardware encoding**: FFmpeg (`h264_nvenc`/`h265_nvenc`).
- **Target bitrate**: 4.0 Mbps (CBR).
- **GOP structure**: 30 frames (I-frame interval 1 second).

#### 5.2.2 Key Data Comparison

| **Metric** | **CPU Software Encoding** | **NVENC Hardware Encoding** | **Optimization Gain** | **Analysis** |
| --- | --- | --- | --- | --- |
| **Encoding FPS** | 7.7 FPS | **37.2 FPS** | ↑ **383%** | Hardware encoder has strong parallel processing capability |
| **CPU Utilization** | 420% | **2.6%** | ↓ **99.4%** | CPU completely idle, available for other tasks |
| **Single-frame Encoding Latency** | 129.2 ms | **26.9 ms** | ↓ **79.2%** | Hardware acceleration has low latency, suitable for real-time encoding |
| **Encoding throughput** | 7.7 FPS | **37.2 FPS** | ↑ **383%** | NVENC supports high-concurrency encoding |
| **File bitrate** | 4.0 Mbps | 4.0 Mbps | — | Bitrate consistent under CBR mode |
| **Video quality** | PSNR 38.2 dB | **PSNR 39.6 dB** | **Better quality** | NVENC has better quality at high bitrates |
| **Multi-stream concurrency** | 1 stream | **4 streams** | ↑ **300%** | NVENC supports multi-stream concurrent encoding |

#### 5.2.3 Technical Interpretation

**Why does NVENC hardware encoding perform excellently?**

1. **Dedicated hardware unit**:
   - NVENC is a **dedicated hardware encoder** inside NVIDIA GPUs, supporting H.264/H.265.
   - The encoding process is **completely completed by hardware**; the CPU only handles data transmission.
2. **Efficient compression algorithms**:
   - NVENC uses **NVIDIA-optimized encoding algorithms**, with better quality at the same bitrate.
   - Supports **B-frames, multi-reference frames**, and other advanced features, improving compression efficiency.
3. **Multi-stream concurrency support**:
   - Jetson Orin's NVENC supports **multi-stream concurrent encoding** (such as 4 streams of 1080P@30FPS).
   - CPU software encoding is limited by core count, usually **1 stream of 1080P is already full load**.

**Limitations of CPU software encoding**

- **CPU bottleneck**: the encoding process occupies a lot of CPU resources, leading to **low frame rates and high latency**.
- **High power consumption**: CPU frequent computation leads to **increased power consumption**, unsuitable for edge devices.
- **Quality loss**: at the same bitrate, software encoding **PSNR is slightly lower than hardware encoding**.

#### 5.2.4 Applicable Scenario Suggestions

| **Scenario** | **Recommended Solution** | **Reason** |
| --- | --- | --- |
| **Real-time streaming (low latency)** | NVENC hardware encoding | Low latency, high FPS |
| **Multi-stream recording** | NVENC hardware encoding | Supports 4+ streams concurrent encoding |
| **High-quality storage** | NVENC hardware encoding | Higher PSNR, better quality |
| **Debug/development environment** | CPU software encoding | No hardware acceleration needed, easy to debug |
| **Storage space sensitive** | CPU software encoding (x265) | x265 has higher compression ratio (but slower) |

## References

- [NVENC Video Encoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html) — Encoding standards supported by NVENC
- [NVDEC Video Decoder API Programming Guide — Video Decoder Capabilities](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#document-video-decoder-capabilities) — Decoding standards supported by NVDEC
