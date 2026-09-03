---
layout: ../../layouts/BlogPost.astro
title: "【工业多模态大模型实战·系列二】动静分离两阶段 VLM 架构与跨窗口时序一致性滤波实战"
date: "2026-08-27"
---

> **专栏导读**：视觉语言大模型（VLM）为工业界带来了强大的零样本开放词表（Open-Vocabulary）识别能力，但在高速移动与强光反射的车载环境下，大模型面临**“多帧 Token 爆炸导致高延迟”**与**“单帧静态幻觉导致高虚警”**两大死穴。本文为 **【动车组受电弓多模态大模型落地实战三部曲】** 的第二篇，深度解析我们自研的“动静分离”两阶段架构设计、强负样本 Prompt 工程，以及将误报率压降 95%+ 的跨窗口空间-时序一致性追踪器。
>
> 📌 **系列导航**：
> * **👉 系列一：[Jetson Orin 部署 Qwen-VL 踩坑实录与生态代沟剖析](/blog/jetson-orin-qwenvl-deployment-deepdive)**
> * **👉 系列二（本文）：[动静分离两阶段 VLM 架构与跨窗口时序一致性滤波实战](/blog/pantograph-vlm-twostage-algorithm)**
> * **👉 系列三：[从 5.5s 到 1.06s：基于 vLLM 与推测并发的高性能推理优化实战](/blog/vllm-inference-acceleration-benchmark)**

---

## 目录

- [一、业务痛点：开集未知异物检测与传统算法的局限](#一业务痛点开集未知异物检测与传统算法的局限)
- [二、“动静分离”两阶段多模态架构设计](#二动静分离两阶段多模态架构设计)
- [三、抗虚警攻坚：跨窗口时序一致性追踪器](#三抗虚警攻坚跨窗口时序一致性追踪器)
- [四、时钟漂移排错：从物理墙上时钟到窗口步长解耦](#四时钟漂移排错从物理墙上时钟到窗口步长解耦)
- [五、实验评测与消融分析](#五实验评测与消融分析)

---

## 一、业务痛点：开集未知异物检测与传统算法的局限

在高速铁路 350 km/h 运行下，受电弓碳滑板直接接触 2.5 万伏高压接触网供电。若受电弓挂上塑料编织袋、风筝线、飞鸟羽毛等外来杂物，可能引发局部高温拉弧、击穿绝缘甚至引发断线恶性事故。

![传统闭集检测器与开放词表 VLM 的对比](/images/blog/closed-vs-open-vlm.svg)
*闭集检测器的能力边界，正是开放词表 VLM 的切入点——但 Token 爆炸与静态幻觉两大挑战随之而来。*

> **💡 模型选型小插曲**：在方案初期，我们曾尝试使用更轻量的 Qwen2-VL（2B 与 7B 版本）。但实测表明，未经微调的 Qwen2-VL 对受电弓微小异物几乎全部漏检（全判为 Normal），不具备可用性。因此团队果断放弃并选用了空间定位与细节感知能力显著提升的 **Qwen3-VL-4B**。

为了兼顾**时序动态故障（大火花、镜头脏污）**与**细粒度静态故障（微小异物）**，我们设计了一套**“动静分离”两阶段多模态推理流水线**。

---

## 二、“动静分离”两阶段多模态架构设计

![动静分离两阶段多模态推理流水线](/images/blog/twostage-pipeline.svg)
*整体流水线：Stage 1 管「动态过程」，Stage 2 管「静态细节」，时序追踪器守住「告警门槛」。*

### 1. Stage 1 动态时序分类 Prompt
```python
PROMPT_STAGE1 = (
    "你是受电弓故障诊断专家。请观察图像，判断：\n"
    "1. 火花(spk)：有无刺眼强光或电弧放电？\n"
    "2. 脏污(dirt)：画面是否影响了视野，极度模糊、失焦、或被水珠/雾气/污泥遮挡？\n"
    "请先给出一句话的推理分析(reasoning)，描述画面的清晰度，然后再输出JSON。\n"
    "示例 JSON:\n"
    '{"reasoning": "画面整体极度模糊，无法看清受电弓细节，判定为脏污。", "spk": 0, "dirt": 1}'
)
```

### 2. Stage 2 强负样本空间约束 Prompt
大模型在没有先验约束时，容易把碳滑板自身的左右弧形弓角、金属包边螺栓误判为“异物”。我们设计了极具工程针对性的强负样本提示词：

```python
PROMPT_STAGE2_FOREIGN = (
    "你是一个严谨的动车组受电弓视觉检测专家。\n"
    "【检测任务】仅检测挂在受电弓最上方碳滑板上的外来异物（如白色羽毛、塑料编织袋碎片、缠绕棉纱线等）。\n"
    "若发现异物，使用XML格式输出其中心点坐标：<point>x, y</point>（0到1000归一化坐标）。\n"
    "【严格负样本排除】：\n"
    "受电弓本体所有机械结构（弧形弯曲弓角、线夹、绝缘子、螺栓、卡扣、金属包边）绝不是异物！\n"
    "列车高速运动造成的灯光拖影、接触网吊弦线绝不是异物！若画面正常无外来垃圾，必须严格直接回复 'Normal'。"
)
```

---

## 三、抗虚警攻坚：跨窗口时序一致性追踪器

大模型单帧由于局部光影反光，可能偶发在螺栓处吐出一个 `<point>`。如果“单帧命中即告警”，系统误报率将高达 10% 以上。

我们提出 **`TemporalConsistencyTracker`（空间-时序一致性滤波追踪器）**：单帧命中只是「疑似」，跨窗口复核通过才算「确诊」。

![TemporalConsistencyTracker 跨窗口一致性滤波](/images/blog/temporal-tracker.svg)
*同一个视频里的两条命运轨迹：真异物两个窗口即确诊告警，瞬态噪点永远停在「疑似」并被静默淘汰。*

核心实现如下：

```python
class ForeignObjectCandidate:
    """疑似候选实体"""
    def __init__(self, point, first_seen_time, window_id):
        self.point = point                      # (x, y) 归一化坐标
        self.first_seen_time = first_seen_time
        self.last_seen_time = first_seen_time
        self.last_window_id = window_id
        self.hit_count = 1                      # 连续命中计数

    def distance_to(self, other_point):
        dx = self.point[0] - other_point[0]
        dy = self.point[1] - other_point[1]
        return (dx * dx + dy * dy) ** 0.5


class TemporalConsistencyTracker:
    def __init__(self, match_distance=100.0, confirm_hits=2, max_window_gap=2, ttl_seconds=30.0):
        self.match_distance = match_distance    # 空间欧氏距离阈值 (<100px 视为同个实体)
        self.confirm_hits = confirm_hits        # 晋升确诊所需命中数 (2 次)
        self.max_window_gap = max_window_gap    # 允许的最大窗口空缺步长 (2 个窗口)
        self.ttl_seconds = ttl_seconds
        self.candidate = None

    def update(self, current_point, now, current_window_id):
        status = "CLEARED"
        confirmed_point = None

        if current_point is not None:
            if self.candidate is None:
                # 首次检出：标记为疑似 (SUSPECT)，暂不告警
                self.candidate = ForeignObjectCandidate(current_point, now, current_window_id)
                status = "SUSPECT"
            else:
                dist = self.candidate.distance_to(current_point)
                gap = current_window_id - self.candidate.last_window_id

                if dist <= self.match_distance and gap <= self.max_window_gap:
                    # 空间一致且时序连续：累加命中
                    self.candidate.hit_count += 1
                    self.candidate.point = current_point
                    self.candidate.last_seen_time = now
                    self.candidate.last_window_id = current_window_id

                    if self.candidate.hit_count >= self.confirm_hits:
                        status = "CONFIRMED"  # 连续 2 次命中，晋升正式告警！
                        confirmed_point = self.candidate.point
                    else:
                        status = "SUSPECT"
                else:
                    # 坐标跳变或跨度过大：重置候选
                    self.candidate = ForeignObjectCandidate(current_point, now, current_window_id)
                    status = "SUSPECT"
        else:
            if self.candidate is not None:
                gap = current_window_id - self.candidate.last_window_id
                if gap > self.max_window_gap or (now - self.candidate.last_seen_time) > self.ttl_seconds:
                    self.candidate = None  # 瞬时误报自动淘汰

        return status, confirmed_point, self.candidate
```

---

## 四、时钟漂移排错：从物理墙上时钟到窗口步长解耦

在系统集成实测中，我们遇到了一个极其隐蔽的致命 Bug：**真实异物在视频中明明持续存在，但时序追踪器始终无法确诊告警！**

### 1. 故障根因复盘
* 早期我们在追踪器中设定了物理墙上时钟 `TTL = 6.0s`；
* 在端到端实时流中，大模型推理耗时 5.0 秒，加上排队和网络开销，相邻两个窗口之间的实际物理耗时达到了 **7.0~9.5 秒**；
* 当窗口 $N+1$ 带着真异物到达时，`now - candidate.last_seen_time` 已经超过了 6.0 秒，**候选目标被错误地判定为超时淘汰**，导致计数器每次都在 1 和 0 之间震荡，永远无法累加到 2 次命中！

### 2. 解决方案：窗口序号步长（Window Gap TTL）数学解耦
我们重构了过期淘汰条件，引入 `gap = current_window_id - last_window_id <= 2`：
* 无论 GPU 算力波动导致单次推理耗时花了 1 秒还是 10 秒，**滑动窗口步长差值永远是严格递增的整数步长**；
* 彻底解耦了硬件时钟漂移对算法决策的干扰！

---

## 五、实验评测与消融分析

### 1. 标准羽毛测试帧定位精度验证 (`foreign_obj.png`)

<!-- SCREENSHOT-SLOT: 插入真实检测效果截图，建议 foreign_obj.png 的可视化版本（模型输出 <point>487, 328</point> 画在原图上，与人工标注真值对比）：
![羽毛异物定位效果](/images/blog/shot-feather-detection.png)
-->

* **真实异物**：受电弓碳滑板中心附着的白色羽毛；
* **模型输出**：`<point>487, 328</point>`；
* **定位精度**：与人工标注真值误差仅 14 个像素（归一化误差 < 1.4%），100% 命中在羽毛核心。

### 2. 10 分钟典型视频消融实验矩阵

| 评测维度 | 无时序滤波 (`rtsp_twostage_vlm.py`) | 启用时序滤波 (`run_full_video_test.py`) |
| :--- | :--- | :--- |
| **真异物检出率 (Recall)** | 100% (第 35 秒捕获) | **100% (第 35~39 秒连续确诊)** |
| **单帧瞬时虚警次数** | **14 次** (严重误报警) | **0 次 (95%+ 瞬态噪点被成功过滤)** |
| **抗反光鲁棒性** | 弱（强光亮斑易误判） | **极强（结合强负样本 Prompt 零误判）** |

---

## 六、承上启下：算法做好了，如何把 5.5 秒的耗时压到 1 秒内？

算法架构与时序滤波解决了“准不准”和“报不报假警”的问题，但 5.5 秒的单窗口耗时依然无法满足实时处理需求。

👉 **请继续阅读系列第三篇：[《从 5.5s 到 1.06s：基于 vLLM 与推测并发的高性能推理优化实战》](/blog/vllm-inference-acceleration-benchmark)**！
