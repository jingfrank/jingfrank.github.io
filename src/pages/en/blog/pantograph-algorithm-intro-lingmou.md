---
layout: ../../../layouts/BlogPost.astro
title: "Lingmou System"
date: "2026-11-04"
---

Like an untiring "eagle-eye sentinel" on the roof, **Lingmou** is an intelligent detection vision system built specifically for pantograph cameras, reacting to pantograph-catenary faults in milliseconds with fast response and high sensitivity. The system deeply integrates three algorithms: YOLO transient object detection, FastAD anomaly detection, and HRNet pose estimation. This multi-algorithm design ensures timely response and low missed detection under highly dynamic driving conditions.

## 1. Multi-algorithm Fusion for Pantograph Fault Recognition

The six types of pantograph faults have very different visual appearances and span a wide range of target scales. Add strong and rapidly changing lighting and background interference on the roof, and vision detection faces very high challenges.

- **Camera dirt**: lacks stable geometric features.
- **Large sparks**: millisecond-level transient high-brightness targets, easily affected by direct sunlight, lighting, and overexposure, and must be detected and alarmed within an extremely short time.
- **Suspended foreign objects**: open-ended categories, small scale, irregular shapes, swaying in the wind; similar to the catenary and background structures.
- **Pantograph structural anomalies**: diverse fault appearances.
- **Pantograph tilt**: must be judged by combining pose and reference relationships. Normal driving vibration, curves, and view-angle changes can produce similar tilt-like behavior.
- **Abnormal pantograph floating**: a dynamic trajectory anomaly rather than a single-frame static anomaly; the boundary between normal fluctuations and faulty floating is blurry.

A single model is hard to detect all of the above faults. Therefore, Lingmou adopts a divide-and-conquer strategy: it uses the YOLO object detection network to capture highly dynamic transient targets, introduces the FastAD anomaly detection model to locate unknown local morphological deviations, and cooperates with the HRNet pose estimation algorithm to solve key-component geometric relationships, achieving specialized and precise recognition of multiple fault types.

![Multi-algorithm fusion architecture of the Lingmou system](/images/blog/pantograph/lingmou-architecture.png)

*Figure 1: Multi-algorithm fusion and coordination architecture of the Lingmou system*

### 1.1 YOLO: Large Sparks and Abnormal Pantograph Floating

For pantograph large sparks and abnormal floating, the system uses the YOLO real-time object detection algorithm to achieve fast localization and recognition of spark discharge targets, as well as detection of the abnormal floating state.

**Large spark detection**: uses a deep convolutional neural network to extract local texture, shape, and saliency features of electric arcs, achieving localization and recognition of pantograph large spark targets.

**Abnormal pantograph floating detection**: combines the onboard TCMS raising/lowering pantograph state signal with vision detection results for joint judgment. When the system detects that the pantograph is in the lowered state, it tracks the actual height of the pantograph recognized by vision in real time. When the height exceeds the limit and continuously does not match the standard lowered height, it is judged as abnormal floating, and an alarm is triggered once an anomaly is detected.

The system adopts YOLO11 [1, 2], the latest generation single-stage object detection model released by Ultralytics in 2024. YOLO11 uses an end-to-end object detection framework that achieves category prediction and bounding box localization of specific targets in an image through a single forward propagation. The model consists of a feature extraction network (Backbone), a feature fusion network (Neck), and a detection head (Head), and improves detection capability for targets of different sizes through multi-scale feature extraction and fusion [1]. Compared with traditional two-stage detection methods, YOLO11 has the characteristics of fast inference speed and a good balance between detection accuracy and computational efficiency, making it suitable for roof-based real-time per-frame visual detection at 30 FPS.

![YOLO11 model architecture](/images/blog/pantograph/yolo11-architecture.png)

*Figure 2: YOLO11 object detection model architecture [1, 2]*

### 1.2 FastAD: Suspended Foreign Objects, Pantograph Structural Anomalies, Camera Dirt

For appearance anomalies such as pantograph suspended foreign objects, structural shape anomalies, and camera lens dirt, the system adopts the FastAD unsupervised vision anomaly detection algorithm. By modeling the feature distribution of normal samples, it achieves detection and localization of anomalous regions.

**Suspended foreign object detection**: based on the visual feature distribution of normal pantographs and the area around the pantograph head, it detects local regions that differ significantly from the normal appearance, helping to discover foreign object suspension on the pantograph head and its surroundings.

**Pantograph structural anomaly detection**: uses the structural shape, local texture, and spatial features of a normal pantograph to build a normal distribution model. By analyzing the deviation between the image to be detected and the normal feature distribution, it identifies structural anomaly regions of the pantograph.

**Camera dirt detection**: based on the image feature distribution under normal imaging conditions, it detects local texture, brightness, and imaging feature anomalies caused by lens dirt, occlusion, and other factors, achieving detection and localization of camera-dirt regions.

The system adopts the FastAD unsupervised vision anomaly detection framework [3]. FastAD uses an ImageNet-pretrained WideResNet-50-2 as the feature extraction network [4], extracting deep visual features of the image to build two-level feature descriptors at the patch level and global level. The algorithm uses channel-wise empirical cumulative distribution functions to statistically describe feature distributions and builds multivariate Gaussian distribution models separately. During inference, it uses Mahalanobis distance to measure the deviation between the features of the image to be detected and the normal feature distribution, generating anomaly scores and anomaly region heatmaps to detect local structural anomalies and global appearance anomalies [3, 5].

![FastAD unsupervised vision anomaly detection process and architecture](/images/blog/pantograph/fastad-architecture.jpg)

*Figure 3: FastAD unsupervised vision anomaly detection framework process and architecture [3, 5]*

Compared with traditional supervised detection methods that rely on large numbers of anomaly samples and fault category annotations, FastAD only needs normal samples to complete feature distribution modeling. It does not need to exhaustively annotate all potential anomaly types, so it can adapt to vision detection scenarios with diverse anomaly types and difficulty collecting defect samples. At the same time, combining local and global feature modeling lets the algorithm balance local anomaly localization with overall appearance anomaly detection, with characteristics such as low training cost, fast inference speed, and flexible deployment.

### 1.3 HRNet: Pantograph Tilt

For possible structural tilting during pantograph operation, the system adopts the HRNet high-resolution feature learning network to build a pantograph keypoint detection and pose estimation model. Through key structure localization and geometric relationship analysis, it achieves recognition of the pantograph tilt state.

**Keypoint localization**: uses a deep convolutional neural network to extract pantograph visual features at different levels, precisely localizes key structure positions of the pantograph, and obtains spatial coordinate information for the keypoints.

**Tilt state assessment**: based on the relative positions between keypoints, it calculates the direction angle and geometric deviation of key pantograph structures and compares them with the normal pose baseline. When the structure tilt exceeds the set threshold, it is judged as a pantograph tilt anomaly.

The system adopts HRNet (High-Resolution Network) as the pose estimation model [6]. HRNet was originally proposed by Sun et al. in 2019. Its core characteristic is that it maintains high-resolution feature representations throughout the network computation and achieves fusion of features at different scales through parallel multi-resolution feature branches and repeated information exchange, thereby retaining richer spatial positional information [6]. In keypoint detection tasks, the model localizes target key structure positions by outputting keypoint heatmaps.

![HRNet high-resolution network architecture](/images/blog/pantograph/hrnet-architecture.png)

*Figure 4: HRNet high-resolution network structure [6]*

Compared with traditional network structures that first reduce feature map resolution and then gradually restore spatial resolution, HRNet can continuously retain high-resolution information during feature extraction, has good spatial localization capability, and is suitable for fine-grained structural keypoint detection of pantographs and vision tasks such as pose abnormality assessment.

---

## References

* [1] Bento, J.; Paixão, T.; Alvarez, A.B. Performance Evaluation of YOLOv8, YOLOv9, YOLOv10, and YOLOv11 for Stamp Detection in Scanned Documents. *Applied Sciences*, 2025, 15(6), 3154. DOI: 10.3390/app15063154.
* [2] Jocher, G.; Qiu, J. Ultralytics YOLO11, 2024. Available online: https://github.com/ultralytics/ultralytics.
* [3] Jing, Y.; Ding, G. FastAD: A Deployable Framework for Real-Time Visual Anomaly Detection. In *Artificial Neural Networks and Machine Learning – ICANN 2026*, Lecture Notes in Computer Science; Springer Nature Switzerland: Cham, 2026; Vol. 17090, pp. 269–281. DOI: [10.1007/978-3-032-38401-0_22](https://doi.org/10.1007/978-3-032-38401-0_22).
* [4] Zagoruyko, S.; Komodakis, N. Wide Residual Networks. In *Proceedings of the British Machine Vision Conference (BMVC)*, York, UK, 19–22 September 2016; pp. 87.1–87.12. DOI: 10.5244/C.30.87.
* [5] Defard, T.; Setkov, A.; Loesch, A.; Audigier, R. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization. In *Pattern Recognition. ICPR International Workshops and Challenges*; Springer: Cham, Switzerland, 2021; pp. 475–489. DOI: 10.1007/978-3-030-68799-1_35.
* [6] Sun, K.; Xiao, B.; Liu, D.; Wang, J. Deep High-Resolution Representation Learning for Human Pose Estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 16–20 June 2019; pp. 5693–5703.
