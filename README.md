# EA-MCM: Energy-Anchored Multi-Context Masking for Transient Bioacoustics

[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-0066CC.svg)](https://2026.ieeeicassp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

> **论文状态：** ICASSP 2026 Submission
> **论文标题：** *Beyond Random Masking: Energy-Anchored Contextual Representation Learning for Transient Bioacoustics*
> **作者：** Xian Tang<sup>1,\*</sup>, Wei Cao<sup>1,\*</sup>, Jia Wang<sup>1,†</sup>
> <sup>1</sup> College of Informatics, Huazhong Agricultural University
> <sup>\*</sup> Equal contribution. <sup>†</sup> Corresponding author.

---

## 📌 更新日志

- **2026.10**：初始版本发布，包含**完整可运行源码**（模型 / 9 种掩码策略 / 6 大基准数据集接口 / 全套训练评测工具链）、参考配置与逐步复现命令

---

## 🔍 项目简介

将音频基础模型（Audio Foundation Models）迁移到**瞬态生物声学**（transient bioacoustics）场景面临严重的域差距与数据稀缺问题。主流掩码自编码器（MAE）预训练范式在该场景下存在两大结构性漏洞：

| 漏洞 | 机理 | 后果 |
|------|------|------|
| **空间捷径学习**（Spatial Shortcut） | 数据无关掩码频繁暴露连续高能量共振峰片段，物理信号的强局部谱连续性使模型可以通过邻域插值 / 边界外推重建被掩码 patch | 编码器学到的不是全局声学语义，而是浅层空间插值 |
| **跨视图信息泄露**（Cross-View Leakage） | 多视图掩码独立采样，任一焦点 patch 在至少一个视图中可见的概率为 $1-\rho_{total}^V$（V=8、ρ=75% 时约 90%） | 编码器跨视角对齐泄露的焦点特征，绕过预期的上下文推理任务 |

**EA-MCM（Energy-Anchored Multi-Context Masking）** 以声学功率为物理先验，把频谱图解耦为**高能量焦点核心**与**低能量环境上下文**：焦点核心在所有视图间被确定性锚定掩码（泄露概率严格为 0），环境上下文逐视图随机采样。仅需闭式 $O(N\log N)$ 能量排序，**无需任何辅助教师网络**，相比教师引导式掩码取得 **8× 加速**（5.1 ms/batch vs 41.0–43.8 ms/batch），同时在全部六个基准上取得领先或持平的下游性能。

### 核心结果一览（P.A.S.T. 微调准确率，n=6 runs）

| 方法 | Data-Aware | Views | Mask Time ↓ | P.A.S.T. Acc ↑ |
|------|:---:|:---:|:---:|:---:|
| Random Masking | ✗ | 1 | 1.5 ms | 93.04±0.54 |
| AttMask-Hint | ✓ (Energy) | 1 | 43.8 ms | 93.70±0.51 |
| Att-Guided Masking | ✓ (Semantic) | 1 | 41.0 ms | 94.22±0.38 |
| Naive Random Multi-Masking | ✗ | 8 | 4.5 ms | 93.37±0.52 |
| **EA-MCM（本文）** | **✓ (Physics)** | **8** | **5.1 ms** | **94.65±0.26** |

---

## 🎯 方法总览

![EA-MCM Architecture](figures/fig2_ea_mcm_architecture.png)

> **EA-MCM 整体框架。** 输入频谱图 → 声学功率度量 → 高能量锚定集 A\*（确定性锁定）+ 低能量上下文 → 动态上下文掩码逐视图采样 → 复合掩码视图送入编码器-解码器。

### 掩码范式对比

![Masking Paradigms](figures/fig1_masking_paradigms.png)

> **图 1：三类掩码范式对比。** 数据无关方法（Random / Time-Freq）暴露高能量焦点区域导致空间捷径；数据感知方法（AttMask-Hint）依赖辅助教师网络带来额外开销；EA-MCM 以物理能量先验实现"锚定核心 + 随机上下文"，兼顾质量与效率。

---

## 💡 研究动机

### 1. 空间捷径学习

将 log-Mel 频谱图 $X \in \mathbb{R}^{T \times F}$ 划分为 $N$ 个 patch。均匀随机掩码下，高能量共振峰簇 $H \subset \mathcal{P}$ 中被暴露的 patch 期望为：

$$
\mathbb{E}\big[|H \cap V_{rand}|\big] = (1-\rho_{total})|H| > 0
$$

由于物理声信号的强局部谱连续性，被掩码的高能量 patch $P_m$ 可由相邻可见 patch $P_v$（欧氏距离 $d(P_m, P_v) < \epsilon$）直接外推：

$$
\hat{P}_m \approx \mathcal{F}_{extrapolate}(P_v)
$$

网络因此通过局部插值即可最小化重建误差——学到的是**空间捷径**而非全局声学依赖。

![P.A.S.T. Spectrograms](figures/fig6_past_spectrograms.png)

> **P.A.S.T. 频谱图示例。** 生物发声表现为稀疏的高能量时频共振峰簇，被环境噪声或静默包围——能量分布高度不均匀，与连续语音 / 稳态环境声截然不同。

### 2. 跨视图信息泄露

多视图掩码框架（每样本生成 $V$ 个掩码视图）中，若各视图独立采样，焦点 patch $P_k \in A^*$ 在至少一个视图中暴露的概率为：

$$
\mathbb{P}\Big(P_k \in \bigcup_{v=1}^{V} V_{rand}^{(v)}\Big) = 1 - \rho_{total}^V \xrightarrow{V \to \infty} 1
$$

$V=8$、$\rho_{total}=75\%$ 时该概率约 **90%**；$V=16$ 时约 99%。编码器足以通过关联跨视图泄露的焦点签名绕过上下文推理任务。朴素多视图基线（Naive Random Multi-Masking, $V=8$）仅达到 93.37%，低于单视图 Random Masking 的收益预期，正是泄露效应的直接证据。

---

## 🏗️ 方法框架

EA-MCM 由三步构成，全部在**原始（未归一化）频谱图**上以闭式计算完成：

### Step 1 — 能量驱动解耦（Energy-Driven Decoupling）

对 log-Mel 输入施加声学功率算子，将 patch 映射回线性谱幅度密度：

$$
E(P_i) = \| \exp(P_i) \|_1, \quad i = 1, \dots, N \tag{3}
$$

### Step 2 — 高能量锚定集（Deterministic Anchor Lockdown）

在严格基数约束 $K_{anchor} = \lfloor \rho_{anchor} N \rfloor$ 下取累积能量最大的 patch 集作为锚定集：

$$
A^* = \arg\max_{A \subseteq \mathcal{P},\ |A| = K_{anchor}} \sum_{P_k \in A} E(P_k) \tag{4}
$$

$A^*$ 在**所有** $V$ 个视图中无条件掩码 → 跨视图目标泄露概率**由构造严格为 0**。

### Step 3 — 动态上下文掩码（Stochastic Ambient Masking）

剩余 patch 构成环境上下文 $\Omega_{context} = \mathcal{P} \setminus A^*$，逐视图均匀采样动态掩码：

$$
M_{dynamic}^{(v)} \sim \mathcal{U}\big(\Omega_{context},\ K_{dynamic}\big), \quad K_{dynamic} = \lfloor (\rho_{total} - \rho_{anchor}) N \rfloor \tag{6}
$$

复合掩码 $M_{total}^{(v)} = A^* \cup M_{dynamic}^{(v)}$。核心实现（对应 [eamcm/masking/ea_mcm.py](eamcm/masking/ea_mcm.py)）：

```python
energy  = patch_energy(raw_fbank)                # E(P_i) = ||exp(P_i)||_1        (Eq. 3)
anchor  = topk_mask(energy, k_anchor)            # A*: 高能量焦点核心             (Eq. 4)
dynamic = topk_mask(noise.masked_fill(anchor, -1), k_dynamic)   # 仅从 Ω_context 采样 (Eq. 6)
masks   = [(anchor | dynamic).clone() for _ in range(V)]        # 锚定集跨视图共享   (Eq. 7)
```

### 框架集成

Transformer 编码器仅处理可见 token，轻量解码器以 MSE 重建被掩码 patch（Eq. 8）。由于 $A^*$ 跨视图不变而背景上下文逐视图变化，该目标迫使编码器从**呼吸前兆、结构混响等物理耦合的上下文线索**反演缺失的发声核心。预训练完成后丢弃解码器，编码器 token 经全局平均池化（GAP）迁移下游。

---

## 📊 实验结果

> 所有指标均为 **6 次独立运行（n=6）的 mean ± 95% CI**。预训练数据：AudioSet → P.A.S.T.（两阶段 SSL）；统一 ViT-Base / Audio-MAE 架构，128-band log-Mel，16×16 patch（64×8 网格 = 512 tokens）。

### Table 1. 掩码范式综合评测

| Base Model / Masking Strategy | 预训练数据 | Data-Aware | Views ($V$) | Mask Time ↓ | P.A.S.T. Acc ↑ | SPC-2 Acc ↑ | ESC-50 Acc ↑ | AS-20K mAP ↑ | BirdSet cmAP ↑ | WMM Mac-F1 ↑ |
|-------------------------------|-----------|:---------:|:-----------:|:-----------:|----------------|-------------|--------------|--------------|----------------|--------------|
| **(A) 大规模基础模型基线** | | | | | | | | | | |
| AST | IN+AS | – | – | – | 87.4 | 96.2 | 89.7 | 30.6 | 35.6 | 91.0 |
| Audio-MAE | AS | – | – | – | 90.2 | 97.8 | 94.8 | 37.8 | 37.7 | 91.7 |
| BEATs (iter3) | AS | – | – | – | 91.5 | 98.3 | 95.8 | 41.8 | 38.1 | 93.5 |
| **(B) 域内 SSL 预训练（单视图）** | | | | | | | | | | |
| Random Masking | AS→P.A.S.T. | ✗ | 1 | 1.5 | 93.04±0.54 | 97.81±0.32 | 94.82±0.41 | 38.13±0.65 | 38.22±0.92 | 91.82±0.95 |
| Time-Freq Masking | AS→P.A.S.T. | ✗ | 1 | 2.4 | 92.24±0.61 | 98.21±0.24 | 94.75±0.45 | 38.40±0.72 | 38.73±0.41 | 92.50±1.21 |
| Inverse Block Masking (EAT) | AS→P.A.S.T. | ✗ | 1 | 2.1 | 93.51±0.48 | 97.91±0.31 | 95.35±0.38 | 38.94±0.58 | 38.66±0.56 | 92.47±0.89 |
| AttMask-Low | AS→P.A.S.T. | ✓ (Energy) | 1 | 43.7 | 92.16±0.59 | 97.69±0.35 | 94.31±0.47 | 38.33±0.69 | 39.11±0.76 | 92.25±0.63 |
| AttMask-Hint | AS→P.A.S.T. | ✓ (Energy) | 1 | 43.8 | 93.70±0.51 | 98.22±0.29 | 95.48±0.36 | 38.85±0.60 | 40.11±0.37 | 94.31±0.38 |
| Hard-Patch Masking | AS→P.A.S.T. | ✓ (Loss) | 1 | 35.4 | 93.86±0.44 | 98.17±0.27 | 95.41±0.33 | 39.03±0.55 | 39.53±0.45 | 93.39±1.14 |
| Att-Guided Masking | AS→P.A.S.T. | ✓ (Semantic) | 1 | 41.0 | 94.22±0.38 | 98.23±0.22 | 95.63±0.29 | 39.11±0.48 | 40.08±0.68 | 94.27±0.71 |
| **(B) 域内 SSL 预训练（多视图）** | | | | | | | | | | |
| Naive Random Multi-Masking | AS→P.A.S.T. | ✗ | 8 | 4.5 | 93.37±0.52 | 98.04±0.28 | 95.12±0.39 | 38.76±0.61 | 39.40±0.68 | 93.41±0.65 |
| **EA-MCM (Ours)** | AS→P.A.S.T. | **✓ (Physics)** | **8** | **5.1** | **94.65±0.26** | **98.61±0.19** | **95.91±0.25** | **39.14±0.10** | **40.53±0.23** | 94.13±0.47 |

**关键发现：**

- **超越数据无关基线**：EA-MCM 94.65% vs Random Masking 93.04%（**+1.61%**）vs Inverse Block 93.51%（**+1.14%**）——几何掩码留下了连续共振峰片段，让编码器得以利用局部插值捷径
- **8× 计算优势**：教师引导方法（AttMask-Hint 43.8 ms、Att-Guided 41.0 ms /batch）受制于辅助前向传播；EA-MCM 仅需 5.1 ms/batch，以闭式 $O(N\log N)$ 能量排序完全消除辅助网络依赖，同时精度更高
- **物理先验的跨域普适性**：BirdSet 上最高 40.53% cmAP（**+2.31%** over Random）；WMM 低信噪比连续谐波场景下 94.13% Macro-F1，与重型语义掩码（Att-Guided 94.27%）持平

### Table 2. 鲁棒性与正交性评测（P.A.S.T. 微调准确率 %）

> 加性噪声来自 FSD50K，混合信噪比 0 dB：Fan（宽带）、Buzz（谐波机械）、Boom（脉冲瞬态）。

| FT 范式 | 掩码策略 | No Noise | Fan | Buzz | Boom |
|---------|---------|:--------:|:---:|:----:|:----:|
| **Group 1: 标准微调** | Random | 93.0 | 83.2 | 82.5 | 75.4 |
| | Inverse Block | 93.5 | 84.6 | 84.3 | 78.2 |
| | **EA-MCM** | **94.6** | **85.8** | **84.9** | **79.1** |
| **Group 2: + SpecAugment** | Random | 93.9 | 85.6 | 86.2 | 77.9 |
| | Inverse Block | 94.6 | 87.5 | 87.1 | 82.3 |
| | **EA-MCM** | **95.4** | **88.6** | **88.3** | **83.9** |
| **Group 3: + Audio Mixup** | Random | 94.1 | 85.3 | 83.6 | 78.5 |
| | Inverse Block | 94.8 | 88.2 | 87.5 | 83.2 |
| | **EA-MCM** | **95.5** | **89.7** | **88.8** | **84.3** |

- **脉冲噪声鲁棒性**：Boom 冲击下 Random Masking 暴跌 17.6% 至 75.4%，EA-MCM 维持 **79.1%**——从上下文前兆反演发声核心的训练方式，避免网络将外部脉冲爆发与内在生物共振峰混淆
- **与标准增强正交**：配合 SpecAugment / Mixup 增益可加，峰值达 **95.5%**，验证能量锚定的结构表征与输入空间扰动相互独立

### Table 3. 锚定比例 × 视图数联合消融（P.A.S.T. Macro-F1 %）

| $\rho_{anchor}$ | V=2 | V=4 | V=8 | V=16 |
|:---:|:---:|:---:|:---:|:---:|
| $\rho_{anchor}=0.1$ | 91.43±0.83 | 91.89±0.79 | 92.22±0.53 | 92.43±0.64 |
| $\rho_{anchor}=0.2$ | 92.17±0.68 | 92.54±0.71 | 92.87±0.50 | 92.91±0.75 |
| **$\rho_{anchor}=0.4$** | 92.46±0.36 | 93.03±0.44 | **93.23±0.31** | 93.11±0.56 |
| $\rho_{anchor}=0.6$ | 92.08±0.59 | 92.47±0.26 | 92.72±0.34 | 92.58±0.44 |

![Anchor Ratio Sensitivity](figures/fig4_anchor_sensitivity.png)

> **补充材料图 1：$\rho_{anchor}$ 敏感性**（$\rho_{total}=75\%$ 固定）。倒 U 型相变：0%（坍缩为朴素多视图随机掩码）93.30% → **40% 处峰值 94.65%** → 75%（上下文池枯竭）93.80%。

- **泄露瓶颈的直接证据**：$\rho_{anchor}=0$ 时即使 $V=16$ 也仅 92.43%——理论焦点泄露概率 $1-0.75^{16} \approx 99.0\%$，重建任务退化为平凡的跨视图特征对齐
- **性能在 $\rho_{anchor}=40\%$、$V=8$ 处达到峰值**：恰好覆盖主要共振峰的同时，动态掩码保留充足的环境上下文（35% 可见 patch）；$\rho_{anchor}=60\%$ 剥夺因果上下文，降至 92.72%

![Patch × View Ablation](figures/fig5_patch_view_ablation.png)

> **补充材料图 2：patch 尺寸 × 视图数联合消融。** 对称 16×16 patch 在所有视图数下均优于拉伸 32×8 配置——各向同性 patch 保留了跨越时间与频率轴的谐波共振 / 调频扫频的二维局部结构完整性。

### 机制验证：空间捷径被系统性消除

| 实验 | 随机多视图 | EA-MCM |
|------|:---------:|:------:|
| 掩码/非掩码 patch MSE 比（冻结编码器探测） | 1.29（邻域插值造成偏斜） | **1.02（接近均匀）** |
| 反转掩码极性（暴露高能量爆发、遮挡低能量上下文） | – | 准确率降至 91.9%，最差类召回 64.2%（−3.1%） |

反转极性实验证实：**被暴露的高能量峰值本身就是捷径**；MSE 比从 1.29 均衡至 1.02 提供了消除空间捷径的定量证据。

### 少样本泛化（Fig. 3）

![Few-shot Performance](figures/fig3_few_shot.png)

- **5 shots/class** 极端稀缺下仍达 **79.2%**（**+4.7%** over Random Masking）
- **100 shots/class** 达 **94.0%**，超过 Random Masking 用**全部**标注数据训练的结果（93.04%）
- **线性探测**：Random Masking 在最稀有的 Hum 类上表征坍缩（Min-Rec 66.3%，mF1 87.4%）；EA-MCM 将 Min-Rec 提升至 **72.6%**、mF1 至 **89.7%**

### 可移植性：即插即用掩码包装器（补充材料 Table C）

| SSL Backbone | 配置 | P.A.S.T. Macro-F1 | BirdSet cmAP |
|--------------|------|:-----------------:|:------------:|
| Bird-MAE (Rauch et al. 2025) | 默认随机掩码 | 93.91 | 50.92 |
| | **+ EA-MCM** | **94.84±0.38** | **52.12±0.44** |
| iNat-MAE (Liu et al. 2026) | 默认随机掩码 | 94.35 | 52.22 |
| | **+ EA-MCM** | **95.07±0.31** | **53.35±0.38** |

EA-MCM 完全无参数、直接作用于原始频谱空间，不改变骨干网络的权重、分词器或池化机制——作为**模块化、即插即用的掩码包装器**，可无缝集成到任意专业领域 SSL 框架。

---

## 📁 P.A.S.T. 与基准数据集

### P.A.S.T.（Porcine Audio for Semantic and Transient understanding）

在活跃的商业化养殖场采集（*Sus scrofa domesticus*），环境含持续机械嗡鸣、结构敲击与通风噪声：

| 部分 | 规模 | 用途 |
|------|------|------|
| 无标注池 | 610 条连续录音 ≈ 30 h | 自监督预训练 |
| 标注核心 | 1,440 段 6 类长尾片段（平均 1.2 s） | 下游评测 |

- **6 类**：`cough`（咳嗽）/ `sneeze`（喷嚏）/ `earflap`（扇耳）/ `eat`（进食）/ `squeal`（尖叫）/ `hum`（哼鸣）
- 双专家独立标注 + 第三专家仲裁分歧
- **session 级 8:1:1 划分**：同一麦克风配置的连续录音块（session）整体归入单一 split，防止混响轮廓 / 个体声道 / 机械周期等表观声学特征跨 split 泄露
- **SNR 特性**：发声为 +12 ~ +25 dB 高信噪比瞬态事件（对比：WMM 为 0 ~ −10 dB 低信噪比连续谐波，构成对能量启发式的严格压力测试）

### 六大基准总览

| 数据集 | 域 | 任务 | 采样率 | 总时长 | 平均时长 | 划分协议 |
|--------|----|------|--------|--------|----------|----------|
| P.A.S.T. | 家猪瞬态发声 | 6 类多分类 | 16 kHz | 30 h（无标注） | 1.2 s | session 级隔离 |
| BirdSet (2025) | 鸟类（细粒度） | 多标签 cmAP | 32 kHz | 82 h | 5.0 s | 标准 train/val/test |
| WMM (2016) | 海洋哺乳动物 | 多分类 Macro-F1 | 48 kHz | 15 h | 2.5 s | 物种级不相交 |
| AS-20K (2017) | 通用环境 / 音乐 | 多标签 mAP（527 类） | 16 kHz | 57 h | 10.0 s | 类均衡子集 |
| ESC-50 (2015) | 环境声 | 50 类多分类 | 44.1 kHz | 2.7 h | 5.0 s | 5 折交叉验证 |
| SPC-2 (2018) | 人类语音瞬态 | 35 词关键词检出 | 16 kHz | 29.4 h | 1.0 s | 标准 train/val/test |

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
torchaudio >= 2.0
numpy >= 1.24
matplotlib >= 3.7   # 掩码可视化
```

### 安装

```bash
git clone https://github.com/youshou1dknowme9320-maker/EA-MCM.git
cd EA-MCM
pip install -r requirements.txt
```

### 数据准备

所有数据集通过统一的 manifest CSV（列：`filepath,label[,fold]`）接入：

```bash
# 1) P.A.S.T. 标注核心：class-per-directory 音频树 -> manifest
python tools/prepare_manifests.py folder --root data/past/clips --out data/past/manifest.csv

# 2) session 级 8:1:1 划分（防止声学轮廓跨 split 泄露）
python tools/split_past_sessions.py --manifest data/past/manifest.csv \
    --train-out data/past/train.csv --val-out data/past/val.csv --test-out data/past/test.csv

# 3) P.A.S.T. 无标注池（SSL 预训练用）
python tools/prepare_manifests.py unlabeled --root data/past/continuous --out data/past/unlabeled.csv

# 4) ESC-50（官方 metadata，保留 5-fold 划分）
python tools/prepare_manifests.py esc50 --meta ESC-50-master/meta/esc50.csv \
    --audio-root ESC-50-master/audio --out data/esc50/manifest.csv

# 5) 多标签数据集（AS-20K / BirdSet）：类别名 -> 整数 id
python tools/prepare_manifests.py multi-csv --meta as20k.csv --out data/as20k/manifest.csv \
    --labels-out data/as20k/labels.json --sep ","
```

### 自监督预训练（Table 1 Group B）

```bash
# EA-MCM（论文默认超参：rho_total=75%, rho_anchor=40%, V=8, AdamW, 300 epochs）
python tools/pretrain.py --strategy ea_mcm \
    --manifest data/past/unlabeled.csv \
    --init-checkpoint ckpt/audiomae_audioset.pth \
    --epochs 300 --batch-size 128 --num-views 8 \
    --output-dir work_dirs/pretrain_eamcm

# 任意基线策略（9 种注册策略任选）
python tools/pretrain.py --strategy random --num-views 1 \
    --manifest data/past/unlabeled.csv --output-dir work_dirs/pretrain_random
python tools/pretrain.py --strategy naive_random_multimasking --num-views 8 \
    --manifest data/past/unlabeled.csv --output-dir work_dirs/pretrain_naive_mv
python tools/pretrain.py --strategy attmask_hint --teacher-source energy \
    --manifest data/past/unlabeled.csv --output-dir work_dirs/pretrain_attmask

# 补充材料 B.2：拉伸 patch 消融（32 mel x 8 frame）
python tools/pretrain.py --strategy ea_mcm --stretched-patch ...
```

可用策略：`random` / `time_freq` / `inverse_block` / `attmask_low` / `attmask_hint` / `hard_patch` / `att_guided` / `naive_random_multimasking` / `ea_mcm`

### 下游微调（Table 1 / Table 2）

```bash
# 主协议：6 次独立运行，mean ± 95% CI
python tools/finetune.py --dataset past \
    --train-manifest data/past/train.csv --val-manifest data/past/val.csv \
    --test-manifest data/past/test.csv \
    --pretrained work_dirs/pretrain_eamcm/encoder_last.pth \
    --runs 6 --seed 42

# Table 2 正交性研究
python tools/finetune.py --dataset past ... --specaugment   # Group 2
python tools/finetune.py --dataset past ... --mixup         # Group 3

# 其他基准（dataset 可选 past/esc50/spc2/wmm/as20k/birdset）
python tools/finetune.py --dataset esc50 --train-manifest data/esc50/manifest.csv \
    --test-manifest data/esc50/manifest.csv --fold 1 \
    --pretrained work_dirs/pretrain_eamcm/encoder_last.pth --runs 6
```

### 轻量评测协议

```bash
# 线性探测（冻结编码器 + 缓存 GAP 特征，仅训练线性头）
python tools/linear_probe.py --dataset past \
    --train-manifest data/past/train.csv --test-manifest data/past/test.csv \
    --pretrained work_dirs/pretrain_eamcm/encoder_last.pth --runs 6

# 少样本（Fig. 3）：K in {5, 25, 50, 100, All}
python tools/few_shot.py --dataset past \
    --train-manifest data/past/train.csv --test-manifest data/past/test.csv \
    --pretrained work_dirs/pretrain_eamcm/encoder_last.pth \
    --shots 5,25,50,100,0 --runs 6

# 鲁棒性（Table 2 Group 1 噪声列）：FSD50K Fan/Buzz/Boom @ 0 dB SNR
python tools/robustness_eval.py --dataset past \
    --test-manifest data/past/test.csv \
    --checkpoint work_dirs/finetune/run0_best.pth \
    --noise-dir data/fsd50k/interference --noise-types clean,fan,buzz,boom
```

### 掩码可视化（Demo 图）

```bash
# 生成掩码策略对比图：视图掩码 + 跨视图曝光图（泄露可视化）+ EA-MCM 锚定集
python tools/visualize_mask.py --audio data/past/clips/session_01/clip_0007.wav \
    --strategies random naive_random_multimasking ea_mcm --output figures/mask_vis
```

多视图策略会额外渲染**跨视图曝光图**（每 patch 在 V 个视图中保持可见的比例）：朴素多视图掩码的焦点区域曝光趋近 1（泄露），EA-MCM 的锚定集曝光**严格为 0**。

---

## ♻️ 可复现性说明

- **统计协议**：全部主表指标为 6 次独立运行的 mean ± 95% CI（工具链内置 `--runs 6`，CI 半宽 = 1.96·σ/√n）
- **少样本采样**：shot 采样按 run 种子固定，固定 `--seed` 时所有预训练编码器在**完全相同**的 shot 集上评测
- **ESC-50**：5 折交叉验证，训练折为其余四折（`--fold` 指定 held-out fold）
- **session 隔离**：`tools/split_past_sessions.py` 以 session 为最小划分单元，同一 session 的片段绝不跨越 train/val/test
- **确定性**：掩码采样支持独立随机数生成器（`seed_generator`），能量排序为确定性闭式运算
- **参考配置**：[configs/pretrain_past.yaml](configs/pretrain_past.yaml) / [configs/finetune_past.yaml](configs/finetune_past.yaml) 记录论文默认超参（`tools/*.py` 为 CLI 驱动，数值一一对应）

### 仓库结构

```
EA-MCM/
├── eamcm/
│   ├── masking/            # 9 种掩码策略：EA-MCM + 8 种基线（统一接口 List[(B,N) bool]）
│   │   ├── base.py         #   基类 + 声学功率算子 patch_energy (Eq. 3)
│   │   ├── ea_mcm.py       #   EA-MCM / Naive Random Multi-Masking
│   │   ├── random.py       #   Random / Time-Freq / Inverse Block (EAT)
│   │   ├── attmask.py      #   AttMask-Low/Hint / Hard-Patch / Att-Guided
│   │   └── registry.py     #   按名构建 + 种子生成器
│   ├── models/
│   │   ├── vit.py          #   ViT 组件（支持注意力回传，供教师策略使用）
│   │   └── mae_audio.py    #   Audio-MAE ViT-Base（非对称解码器 512/4/16, norm_pix_loss）+ 分类头
│   ├── datasets/           #   P.A.S.T. / BirdSet / WMM / AS-20K / ESC-50 / SPC-2 + K-shot 数据集
│   └── utils/              #   音频前端（Kaldi fbank 25ms/10ms）、指标、优化器/调度、增强与噪声注入
├── tools/
│   ├── pretrain.py         #   SSL 预训练（9 策略 CLI，报告 ms/batch）
│   ├── finetune.py         #   下游微调（n=6 runs, mean±CI, SpecAugment/Mixup 正交性）
│   ├── linear_probe.py     #   线性探测（特征缓存）
│   ├── few_shot.py         #   少样本评测（Fig. 3）
│   ├── robustness_eval.py  #   FSD50K 加性噪声鲁棒性（Table 2）
│   ├── visualize_mask.py   #   掩码 / 曝光图可视化（demo）
│   ├── prepare_manifests.py#   manifest 构建（folder/csv/esc50/multi-csv/unlabeled）
│   ├── split_past_sessions.py  #   session 级 8:1:1 划分
│   └── common.py           #   数据集规格表、评测循环、共享工具
├── configs/                #   参考配置（论文默认超参）
├── tests/                  #   18 项单元测试（掩码预算/锚定集/泄露、前端、模型）
├── figures/                #   论文图 + 掩码可视化 demo
├── requirements.txt
└── LICENSE
```

---

## 📖 引用

```bibtex
@inproceedings{tang2026eamcm,
  title     = {Beyond Random Masking: Energy-Anchored Contextual Representation
               Learning for Transient Bioacoustics},
  author    = {Tang, Xian and Cao, Wei and Wang, Jia},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal
               Processing (ICASSP)},
  year      = {2026}
}
```

---

## 🙏 致谢

本研究得到英姿科技&华中农业大学食品健康智能研究院（IRIFH202209）与广西重点研发计划（No. 2025JBGS042）的资助。我们感谢 Audio-MAE、EAT、AttMask、Hard Patches Mining、BirdSet、Watkins Marine Mammal Database、ESC-50、Speech Commands 与 AudioSet 的作者提供的开源资源。

## 📄 License

本项目基于 [MIT License](LICENSE) 发布。
