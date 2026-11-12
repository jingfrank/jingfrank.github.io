---
layout: ../../../layouts/BlogPost.astro
title: "Zhimou System"
date: "2026-11-04"
---

**Zhimou** is built on top of **Lingmou**, with a multimodal vision-language large model (Qwen3-VL) integrated on top, giving the system a "smart brain". Using the large model's scene-prior understanding, it performs "expert-style" secondary review of candidate anomalies output by Lingmou, significantly reducing false alarms while maintaining high recall.

## 1. "Lingmou + Multimodal Large Model Review" Collaborative Architecture

![Collaborative architecture of Lingmou + Multimodal Large Model Review](/images/blog/pantograph/zhimou-collaborative-architecture.png)

*Figure 1: Collaborative architecture of Lingmou + Multimodal Large Model Review*

**Lingmou** is responsible for fast perception of multiple types of pantograph faults and candidate anomaly detection. **Zhimou** further combines the original image, candidate anomaly regions, and detection results to comprehensively understand and semantically analyze the visual features, spatial relationships, pantograph structural features, and surrounding scenes of abnormal targets, extending capability from visual anomaly detection toward scene-level semantic understanding and intelligent assisted decision-making. This strengthens comprehensive review capability under complex line environments and background interference, and further improves fault recognition accuracy and alarm reliability.

- **Candidate anomaly review**: receives candidate anomaly images and related detection information output by Lingmou, combines normal pantograph structural features and fault judgment requirements, performs semantic understanding and secondary judgment on suspected abnormal targets, and distinguishes real faults from non-fault targets.
- **Complex background interference identification**: targets background factors in the line such as high-voltage lines, antennas, tunnel roofs, catenary support structures, and shadows that easily visually overlap with the pantograph. Using the large model's scene-context understanding capability, it comprehensively analyzes the positional relationships and semantic associations between targets and the pantograph, reducing false alarms caused by background interference.
- **Review result generation**: based on multimodal analysis results, it forms assisted review conclusions for candidate anomalies, outputting anomaly categories, judgment basis, and handling recommendations, providing semantic-level supplementary information for system alarm decision-making.

The system adopts the Qwen3-VL multimodal vision-language large model [7]. The model uses a multimodal architecture consisting of a vision encoder (Vision Encoder), a vision-language fusion module (Vision-Language Merger), and a large language model (LLM). It encodes image information into visual features, fuses them with text information, and achieves joint modeling of visual perception and language understanding.

Qwen3-VL enhances information interaction between visual features and the language model through a multi-layer visual feature fusion mechanism (DeepStack), and adopts an improved multimodal rotary position encoding (Interleaved-MRoPE) to strengthen modeling of image spatial information and video temporal information [7]. The model supports image, text, and video multimodal inputs and can complete visual semantic understanding and multimodal reasoning under complex scenes.

![Qwen3-VL multimodal model architecture](/images/blog/pantograph/qwen3-vl-architecture.jpg)

*Figure 2: Qwen3-VL multimodal vision-language large model technical architecture [7]*

Compared with traditional dedicated vision detection algorithms, Qwen3-VL can not only recognize visual targets in images but also combine natural language instructions and scene context to comprehensively analyze semantic relationships between targets. By introducing the multimodal large model into the traditional vision detection pipeline, Zhimou forms a layered detection architecture: dedicated vision algorithms are responsible for candidate anomaly discovery, and the multimodal large model is responsible for semantic review, improving the system's anomaly review capability under complex open environments.

---

## References

* [7] Qwen Team. Qwen3-VL Technical Report. arXiv 2025, arXiv:2511.21631.
