---
layout: ../../layouts/BlogPost.astro
title: "灵眸系统"
date: "2026-11-04"
---

本次测试对受电弓视频故障监测系统的两个算法版本进行对比评测，分别为灵眸与智眸。两个算法的简介如下。

## 1.1  灵眸系统

“灵眸”如同不知疲倦的车顶“鹰眼哨兵”，是专为受电弓摄像头打造的智能检测视觉系统，以单帧毫秒级的反应速度，对弓网故障响应快、敏感度高。系统深度融合了 YOLO 瞬态目标检测、FastAD 异常检测与 HRNet 姿态评估多个算法，多算法设计确保了高动态行车工况下的及时响应与低漏报。

### 1.1.1  多算法融合的受电弓故障识别

6类受电弓故障视觉形态各异、目标尺度跨度大，叠加车顶剧烈变化的光照与背景强干扰，视觉检测面临极高挑战。

画面脏污缺乏稳定几何特征；大火花为毫秒级瞬态高亮目标，易受太阳直射、灯光和过曝干扰，且需在极短时间内完成检出与报警；悬挂异物异物种类开放、尺度小、形状不定、随风飘动；与接触网、背景结构相似；受电弓结构异常故障形态多样；受电弓倾斜需结合姿态和参照关系判断，正常行车振动、弯道和视角变化会产生类似倾斜表现；异常飘弓是动态轨迹异常而非单帧静态异常，正常波动与故障飘弓边界模糊。

单一模型很难通检上述故障。为此，灵眸系统采用分而治之的策略：利用目标检测网络YOLO捕捉高动态瞬态目标，引入异常检测模型FastAD定位未知的局部形态偏差，并配合姿态估计算法HRNet解算关键构件几何关系，实现对多类故障的专项精准识别。

![灵眸系统多算法融合架构](/images/blog/pantograph/lingmou-architecture.png)

*图 1：灵眸系统多算法融合协同架构*

#### YOLO：大火花、异常飘弓

针对受电弓大火花与异常飘弓，系统采用 YOLO 实时目标检测算法，实现大火花放电目标的快速定位识别，以及异常飘弓状态的检测。

大火花检测：利用深度卷积神经网络提取电弧的局部纹理、形态及显著性特征，实现弓网大火花目标的定位与识别。

异常飘弓检测：结合车载 TCMS 升降弓状态信号与视觉检测结果进行联合判定。当系统检测到受电弓处于降弓状态时，实时追踪视觉识别得到的受电弓实际高度；当高度超限且持续与标准落弓高度不匹配时判定为异常飘弓，并在检测到异常后触发报警。

系统采用 Ultralytics 于 2024 年发布的新一代单阶段目标检测模型 YOLO11 [1, 2]。YOLO11 采用端到端的目标检测框架，通过单次前向传播实现图像中特定目标的类别预测与边界框定位。模型由特征提取网络（Backbone）、特征融合网络（Neck）和检测头（Head）组成，通过多尺度特征提取与融合，提升对不同尺寸目标的检测能力 [1]。相较于传统两阶段检测方法，YOLO11 具有推理速度快、检测精度与计算效率平衡较好等特点，适用于车顶 30 FPS 逐帧实时视觉检测任务。

![YOLO11 模型架构](/images/blog/pantograph/yolo11-architecture.png)

*图 2：YOLO11 目标检测模型架构 [1, 2]*

#### FastAD：悬挂异物、受电弓结构异常、画面脏污

针对受电弓悬挂异物、结构形态异常及摄像头画面脏污等外观异常，系统采用 FastAD 无监督视觉异常检测算法，通过正常样本特征分布建模，实现异常区域的检测与定位。

悬挂异物检测： 基于正常受电弓及弓头周边区域的视觉特征分布，检测与正常外观存在显著差异的局部区域，辅助发现弓头及其周边的异物悬挂现象。

受电弓结构异常检测： 利用正常受电弓的结构形态、局部纹理及空间特征建立正常分布模型，通过分析待检测图像与正常特征分布之间的偏离程度，识别受电弓结构异常区域。

画面脏污检测： 基于正常成像条件下的图像特征分布，检测由镜头污渍、遮挡等因素引起的局部纹理、亮度及成像特征异常，实现摄像头画面脏污区域的检测与定位。

系统采用FastAD 无监督视觉异常检测框架 [3]。FastAD 以 ImageNet 预训练的 WideResNet-50-2 为特征提取网络 [4]，通过提取图像的深层视觉特征，分别构建局部图像块（Patch-level）与全局特征图（Global-level）两级特征描述。算法利用通道级经验累积分布函数对特征分布进行统计描述，并分别建立多元高斯分布模型。在推理阶段，采用马氏距离（Mahalanobis Distance）度量待检测图像特征与正常特征分布之间的偏离程度，生成异常评分及异常区域热力图，实现局部结构异常与全局外观异常的检测 [3, 5]。

![FastAD 无监督视觉异常检测流程与架构](/images/blog/pantograph/fastad-architecture.jpg)

*图 3：FastAD 无监督视觉异常检测框架流程与架构 [3, 5]*

相较于依赖大量异常样本及故障类别标注的传统监督检测方法，FastAD 仅需正常样本即可完成特征分布建模，无需对所有潜在异常类型进行穷举标注，能够适应异常类型多样、缺陷样本难以收集的视觉检测场景。同时，局部与全局特征建模相结合，使算法兼顾局部异常定位与整体外观异常检测，具有训练成本低、推理速度快及部署灵活等特点。

#### HRNet：受电弓倾斜

针对受电弓运行过程中可能出现的结构倾斜问题，系统采用 HRNet 高分辨率特征学习网络，构建受电弓关键点检测与姿态评估模型，通过关键结构位置定位及几何关系分析，实现受电弓倾斜状态的识别。

关键点定位： 利用深度卷积神经网络提取受电弓不同层级的视觉特征，对受电弓关键结构位置进行精确定位，获取关键点的空间坐标信息。

倾斜状态评估： 根据关键点之间的相对位置关系，计算受电弓关键结构的方向角及几何偏差，并与正常姿态基准进行比较。当结构倾斜程度超过设定阈值时，判定为受电弓倾斜异常。

系统采用 HRNet（High-Resolution Network）高分辨率网络作为姿态估计模型 [6]。HRNet 最初由 Sun 等人于 2019 年提出，其核心特点是在网络计算过程中始终保持高分辨率特征表示，并通过并行的多分辨率特征分支及反复的信息交换，实现不同尺度特征的融合，从而保留更加丰富的空间位置信息 [6]。在关键点检测任务中，模型通过输出关键点热力图（Keypoint Heatmap），实现目标关键结构位置的定位。

![HRNet 高分辨率网络架构](/images/blog/pantograph/hrnet-architecture.png)

*图 4：HRNet 高分辨率网络结构 [6]*

相较于先降低特征图分辨率、再逐步恢复空间分辨率的传统网络结构，HRNet 能够在特征提取过程中持续保留高分辨率信息，具有较好的空间定位能力，适用于受电弓细粒度结构关键点检测及姿态异常评估等视觉任务。

---

### 参考文献

* [1] Bento, J.; Paixão, T.; Alvarez, A.B. Performance Evaluation of YOLOv8, YOLOv9, YOLOv10, and YOLOv11 for Stamp Detection in Scanned Documents. *Applied Sciences*, 2025, 15(6), 3154. DOI: 10.3390/app15063154.
* [2] Jocher, G.; Qiu, J. Ultralytics YOLO11, 2024. Available online: https://github.com/ultralytics/ultralytics.
* [3] Gao, S.; Liu, X.; Gao, S.; et al. Detection and Assessment of Pantograph-Catenary Arcing Based on High-Speed Vision. *IEEE Transactions on Instrumentation and Measurement*, 2021, 70: 1-11.
* [4] Zagoruyko, S.; Komodakis, N. Wide Residual Networks. In *Proceedings of the British Machine Vision Conference (BMVC)*, York, UK, 19–22 September 2016; pp. 87.1–87.12. DOI: 10.5244/C.30.87.
* [5] Defard, T.; Setkov, A.; Loesch, A.; Audigier, R. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization. In *Pattern Recognition. ICPR International Workshops and Challenges*; Springer: Cham, Switzerland, 2021; pp. 475–489. DOI: 10.1007/978-3-030-68799-1_35.
* [6] Sun, K.; Xiao, B.; Liu, D.; Wang, J. Deep High-Resolution Representation Learning for Human Pose Estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 16–20 June 2019; pp. 5693–5703.
