# MiC Scan-to-BIM Alignment & Semantic Dimension Measurement

**Repository for Chapter 4 of the paper**  

![image])(./Scan-BIM-measurement/asset/scan-bim.png)

## 简介

本仓库提供针对钢结构MiC模块的Scan-BIM对齐与基于BIM语义先验的关键尺寸自动化测量完整实现，包括：

- 基于谱特征分析的分层BIM参考点云构建
- 分级几何约束的Scan-BIM对齐方法（姿态锁定 + 位置锁定 + 鲁棒微调）
- BIM测量模板解析与自适应ROI构建
- 基于法向约束与M-估计的局部鲁棒几何拟合
- 关键尺寸自动提取与偏差分析流程

主要目标是将扫描点云与设计BIM模型稳定对齐，并利用BIM语义先验实现工程尺寸的自动化、结构化提取。
