# 2026-09-16 M1 全市场股票关联网络研究

**研究主题：Full-Market Network Construction and Structural Validation**  
**日期：2026-09-16**

---

## 1. 今日研究目标

今天的工作重点是在前期完成收益率面板、市场共同因子识别、行业标签验证以及 Raw / Market-Residual 相关性诊断的基础上，正式进入 **A 股全市场股票关联网络构建与结构验证**。

核心目标是回答：

1. 在完全可比的节点与边预算下，Raw Network 与 Market-Residual Network 是否具有实质不同的网络结构？
2. 去除市场共同因子后，网络是否呈现更明确的行业结构与经济解释？
3. 这些网络结构在时间上是否具有持续性？
4. 主要结论是否依赖于主设定 \(q=1\%\) 的 sparsification choice？

今天完成了从 **Step 1 网络设计冻结** 到 **Step 6 Sparsification Robustness** 的完整流程。

---

# 2. Step 1：Freeze Formal Network Construction Design

## 2.1 网络定义

正式比较两类网络：

- **B0 Raw Pearson Network**
- **M1 Market-Residual Pearson Network**

其中 Residual Network 基于 leave-one-out 等权市场因子进行滚动回归：

\[
r_{i,s}
=
\alpha_i
+
\beta_i F_{-i,s}
+
\varepsilon_{i,s},
\]

并使用：

\[
\rho^{Residual}_{ij}
=
Corr(\varepsilon_i,\varepsilon_j)
\]

构建关联网络。

## 2.2 公平比较原则

正式固定：

\[
\boxed{
\text{Same Date}
+
\text{Same Window}
+
\text{Same Node Universe}
+
\text{Same Comparable Pair Universe}
+
\text{Same Edge Budget}
}
\]

因此 Raw 与 Residual 网络的差异主要来自 **edge identity 与 topology 的变化**，而不是节点数或网络密度差异。

## 2.3 Sparsification 设定

使用 Global Top-\(q\)：

\[
q\in\{0.2\%,0.5\%,1\%,2\%\},
\]

主分析：

\[
q=1\%.
\]

窗口：

\[
W\in\{60,120,252\}.
\]

Step 1 的作用是把后续 Step 2–6 的比较口径全部固定下来。

---

# 3. Step 2：Construct Raw & Market-Residual Matched-Density Networks

## 3.1 完成内容

Step 2 正式构造全市场 Raw 与 Residual 网络，并生成：

- matched-density network summary；
- ranked edge master；
- common node universe；
- historical industry labels；
- market beta / \(R^2\)；
- stock mapping 与 metadata。

主分析使用：

\[
q=1\%.
\]

## 3.2 Raw 与 Residual strongest edges 明显不同

在 \(q=1\%\) 下，两类网络 edge overlap 约为：

| Window | Raw–Residual Edge Overlap |
|---|---:|
| \(W=60\) | 0.332 |
| \(W=120\) | 0.331 |
| \(W=252\) | 0.337 |

即只有约三分之一 strongest edges 同时出现在两类网络中。

因此：

\[
\boxed{
\text{Market residualization causes substantial edge rewiring.}
}
\]

## 3.3 Residual Network 的行业结构明显增强

主设定 \(q=1\%\) 下：

| Window | Raw Industry Enrichment | Residual Industry Enrichment |
|---|---:|---:|
| \(W=60\) | 3.41 | **4.83** |
| \(W=120\) | 3.97 | **5.86** |
| \(W=252\) | 4.75 | **7.07** |

且所有月份均满足：

\[
Enrichment^{Residual}
>
Enrichment^{Raw}.
\]

### Step 2 结论

\[
\boxed{
\text{去除市场共同成分后，strongest-edge 网络发生显著重构，并呈现更强的行业相关结构。}
}
\]

---

# 4. Step 3：Structural Diagnostics

Step 3 在相同 \(N\) 与 \(E\) 下比较 Raw 与 Residual 网络的 topology。

主要分析：

- connected components；
- giant component；
- isolated nodes；
- degree / strength；
- Gini；
- hub concentration；
- clustering / transitivity；
- degree assortativity；
- Raw–Residual centrality re-ranking。

---

## 4.1 Q1：Residual Network 是否更加 fragmented？

结果显示：**不是。Residual Network 反而显著更连通。**

### Giant Component Share

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | 0.674 | **0.978** |
| \(W=120\) | 0.658 | **0.952** |
| \(W=252\) | 0.669 | **0.904** |

### Isolated Node Share

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | 0.305 | **0.020** |
| \(W=120\) | 0.323 | **0.044** |
| \(W=252\) | 0.313 | **0.089** |

因此 Raw strongest edges 虽然受到明显市场共同波动影响，但这些 strongest edges 高度集中于少数股票群体；去市场后，strongest associations 反而覆盖了更广泛的股票。

---

## 4.2 Q2：哪一张网络的 connectivity 更集中？

结果非常明确：

\[
\boxed{
\text{Raw Network 的 hub concentration 明显更高。}
}
\]

### Degree Gini

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | 0.784 | **0.514** |
| \(W=120\) | 0.792 | **0.577** |
| \(W=252\) | 0.784 | **0.629** |

### Top 1% Degree Share

Raw：

\[
11.6\%,\ 12.7\%,\ 13.6\%.
\]

Residual：

\[
5.9\%,\ 6.7\%,\ 7.2\%.
\]

因此：

\[
\boxed{
\text{Residual Network 的 strongest edges 更均匀地分布于全市场，而不是被少数 hubs 支配。}
}
\]

---

## 4.3 Q3：去 Market Mode 后中心股票是否重排？

Raw–Residual Degree Rank Spearman：

\[
0.476,\quad0.505,\quad0.511.
\]

Top 1% Degree Hub overlap 仅约：

\[
6.2\%,\quad6.3\%,\quad5.8\%.
\]

说明：

\[
\boxed{
\text{Residualization 不仅 rewires edges，也重新定义了哪些股票是网络中心。}
}
\]

Residual Network 的 Degree Assortativity 也显著更高：

\[
0.452,\quad0.421,\quad0.402,
\]

而 Raw 分别约为：

\[
0.106,\quad0.010,\quad-0.051.
\]

### Step 3 结论

Residual Network 具有：

- 更大的 giant component；
- 更少的 isolated stocks；
- 更低的 hub concentration；
- 明显不同的 central stocks；
- 更强的 degree assortative organization。

因此 Raw→Residual 的变化不仅是 correlation 数值变化，而是完整的 **network topology reorganization**。

---

# 5. Step 4：Industry / Community Validation

Step 4 进一步回答：

\[
\boxed{
\text{Step 3 的 topology reorganization 是否具有经济结构？}
}
\]

分析两个层面：

1. **Industry Validation**
2. **Endogenous Community Validation**

行业标签不参与 edge selection 或 community detection，从而避免循环论证。

---

## 5.1 行业连接明显增强

Residual Network 的 industry enrichment 明显高于 Raw，并且所有月份方向一致。

Industry assortativity 也由 Raw 的约：

\[
0.12\sim0.19
\]

提高到 Residual 的：

\[
0.21\sim0.33.
\]

因此：

\[
\boxed{
\text{Residual Network 具有更强的局部同行业连接倾向。}
}
\]

---

## 5.2 Residual Network 的 modular organization 更明显

Weighted modularity：

Raw：

\[
0.34\sim0.38,
\]

Residual：

\[
0.58\sim0.64.
\]

说明去市场后的网络具有更强的 contemporaneous modular organization。

---

## 5.3 但 Residual communities 并非传统行业分类的简单复制

Community–Industry NMI：

\[
W=60:\quad0.265\rightarrow0.189,
\]

\[
W=120:\quad0.318\rightarrow0.249,
\]

\[
W=252:\quad0.366\rightarrow0.319.
\]

Raw 与 Residual community partition 之间的 NMI 约为：

\[
0.46,\quad0.50,\quad0.56.
\]

因此：

\[
\boxed{
\text{Residual Network 的行业 edge structure 更强，但其 endogenous communities 并不是传统行业分类的简单复刻。}
}
\]

### Step 4 结论

Residual Network 同时表现为：

\[
\boxed{
\text{stronger local industry assortativity}
+
\text{stronger modular organization}
}
\]

但社区结构还包含明显的跨行业信息，说明网络可能刻画了传统行业之外的经济关联。

---

# 6. Step 5：Dynamic Stability

Step 5 研究相邻月份网络的：

- edge persistence；
- node centrality stability；
- community temporal stability；
- same-industry vs cross-industry persistence。

主结果使用 common-node-adjusted stability，避免把股票池变化误认为 edge rewiring。

---

## 6.1 Q1：Residual edges 是否更加稳定？

答案是：**是。**

### Adjacent Edge Jaccard

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | 0.244 | **0.282** |
| \(W=120\) | 0.429 | **0.499** |
| \(W=252\) | 0.608 | **0.697** |

### Forward Edge Retention

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | 0.390 | **0.436** |
| \(W=120\) | 0.602 | **0.663** |
| \(W=252\) | 0.760 | **0.823** |

因此：

\[
\boxed{
\text{Residual strongest-edge network 自身具有更高的相邻期 persistence。}
}
\]

---

## 6.2 Q2：节点中心性是否稳定？

Degree rank Spearman：

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | **0.751** | 0.726 |
| \(W=120\) | 0.876 | **0.890** |
| \(W=252\) | 0.941 | **0.958** |

结论是：

- 短窗口下 Residual centrality 更动态；
- 中长窗口下 Residual hub ranking 非常稳定；
- Top-5% hub retention 同样总体支持 Residual 的持续性。

这与 Step 3 并不矛盾：

\[
\text{Raw hubs 与 Residual hubs 不同}
\]

但：

\[
\text{Residual hubs 自身可以具有较高时间持续性。}
\]

---

## 6.3 Q3：Residual community 是否更加稳定？

答案是：**总体不是。**

Community Temporal NMI：

| Window | Raw | Residual |
|---|---:|---:|
| \(W=60\) | **0.432** | 0.325 |
| \(W=120\) | **0.596** | 0.494 |
| \(W=252\) | **0.705** | 0.663 |

因此：

\[
\boxed{
\text{strong contemporaneous modularity}
\neq
\text{fixed community membership}.
}
\]

Residual communities 在每个截面具有较强结构，但 community composition 随时间会发生重新组织。

---

## 6.4 Q4：Same-industry edges 是否更持久？

答案非常明确：**是。**

### Residual Network

\[
W=60:
\quad
0.594>0.384,
\]

\[
W=120:
\quad
0.771>0.618,
\]

\[
W=252:
\quad
0.879>0.792.
\]

三个窗口的所有相邻期转换中均满足：

\[
Retention^{Same}
>
Retention^{Cross}.
\]

### Step 5 结论

今天形成了一个很重要的动态描述：

\[
\boxed{
\textbf{Stable Industry Backbone + Dynamic Cross-industry / Community Reorganization}
}
\]

即：

- 同行业关系形成更稳定的 association backbone；
- 跨行业关系更加动态；
- Residual communities 会随时间重新组合。

---

# 7. Step 6：Sparsification Robustness

Step 6 检验：

\[
q=0.2\%,0.5\%,1\%,2\%
\]

下 Step 2–5 的结论是否保持。

---

## 7.1 Raw–Residual rewiring 对 density 稳健

不同 \(q\) 下 Raw–Residual edge overlap 始终只有约：

\[
0.32\sim0.37.
\]

即使从最稀疏的 \(0.2\%\) 放宽到 \(2\%\)，仍有约：

\[
63\%-68\%
\]

的 strongest edges 不同。

因此：

\[
\boxed{
\text{Raw→Residual rewiring 不是 }q=1\%\text{ 的人为结果。}
}
\]

---

## 7.2 Industry advantage 对 density 极其稳健

全部：

\[
3\times4=12
\]

个 \(W\times q\) 组合中，所有月份都满足：

\[
Enrichment^{Residual}
>
Enrichment^{Raw}.
\]

同时发现：

\[
q\downarrow
\Rightarrow
IndustryEnrichment\uparrow.
\]

例如 \(W=252\)：

| \(q\) | Raw | Residual |
|---:|---:|---:|
| 0.2% | 7.45 | **10.89** |
| 0.5% | 5.75 | **8.58** |
| 1% | 4.75 | **7.07** |
| 2% | 3.92 | **5.70** |

说明：

\[
\boxed{
\text{越极端的 strongest associations，行业集中程度越强。}
}
\]

---

## 7.3 Topology conclusion 基本稳健

Residual Degree Gini 在所有 \(W,q\) 下平均都低于 Raw，Giant Component 平均都大于 Raw。

严格预设的 robustness criteria 中：

\[
\boxed{11/12}
\]

个 \(W\times q\) 组合全部通过。

唯一未通过的是：

\[
W=252,\quad q=0.2\%.
\]

原因是月份一致性没有达到预设 90% 门槛，但平均方向仍然与主结论一致，并没有发生结论反转。

---

## 7.4 Dynamic persistence 对 density 稳健

全部 12 个 \(W\times q\) 组合均满足：

\[
Retention^{Residual}
>
Retention^{Raw}.
\]

同时全部组合均满足：

\[
Retention^{Same}
>
Retention^{Cross}.
\]

因此 Step 5 的动态发现同样不是 \(q=1\%\) 特有结果。

---

## 7.5 Per-node Top-\(k\) 当前不能正式完成

现有 Step 2 仅保存 global Top-2% edges，无法保证恢复每只股票的真实 Top-\(k\) 邻居。

Exact recovery rate 很低：

- Top-10：Raw 约 1.5%，Residual 约 2.8%；
- Top-25：约 0.4%–0.7%；
- Top-50：接近 0.1% 或更低；
- 所有日期的 `all_dates_fully_exact=False`。

因此目前正式 robustness 应限定为：

\[
\boxed{\text{Global Top-}q\text{ robustness}}
\]

而不能声称已完成 exact per-node Top-\(k\) robustness。

---

# 8. 今日主要研究结论

经过 Step 2–6，可以把今天最重要的结果概括成以下证据链：

\[
\boxed{
\text{Market residualization}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Strong Raw–Residual edge rewiring}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Lower hub concentration + broader market connectivity}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Stronger industry assortativity + stronger modular organization}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Stable industry backbone + dynamic cross-industry/community organization}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Conclusions robust across }q=0.2\%-2\%
}
\]

因此，今天最核心的经验结论是：

> **Market-Residual Network 并不是 Raw Network 的轻微修正，而是一张在边、中心节点、行业结构和动态行为上都具有系统差异的全市场关联网络。去除市场共同波动后，strongest associations 覆盖更广泛的股票、对少数 hubs 的依赖更低、同行业结构更突出，并且同行业边具有更高的时间持续性。这些发现对较宽范围的网络 sparsification 均保持稳定。**

---

# 9. 当前方法学边界与需要保留的说明

今天也进一步明确了几个不能过度解释的部分。

### 9.1 Residual correlation 不是 conditional independence

当前主网络是：

\[
Corr(
\varepsilon_i,
\varepsilon_j
),
\]

因此它仍是 **association network**，不能解释为 Graphical Model 中的条件独立结构。

Sparse precision / nodewise Lasso / CLIME 等 conditional network 应作为后续 challenger，而不是与当前 residual correlation network 混为一谈。

### 9.2 更长窗口的稳定性更高存在机械成分

从：

\[
W=60\rightarrow120\rightarrow252
\]

稳定性普遍提高，但更长 rolling windows 的相邻样本重叠本身就更高。

因此正式结论应主要基于：

\[
\boxed{
\text{同一个 }W\text{ 下 Raw vs Residual}
}
\]

以及：

\[
\boxed{
Same\text{-industry vs Cross-industry}
}
\]

而不是直接把不同 \(W\) 的绝对值作为经济稳定性排序。

### 9.3 Community 不是 Industry 的简单替代

Residual Network 虽然具有更高 modularity 和更高 industry edge assortativity，但 Community–Industry NMI 并未提高。

因此更合理的解释是：

\[
\boxed{
\text{Residual network retains industry structure while also containing cross-industry information.}
}
\]

### 9.4 Per-node Top-\(k\) 需要重新设计 pair scan

若后续必须进行 exact Top-\(k\) robustness，应在 full pair block scan 时，为每个节点独立维护其 Top-\(k\) candidate edges。

---

# 10. 今日主要输出文件

今天建立了完整的全市场网络分析数据链。

## Step 2

- `network_construction_summary.csv`
- `matched_density_network_summary.csv`
- `edge_master/`
- `node_universe/`

## Step 3

- `structural_diagnostics_window_summary.csv`
- `raw_residual_node_comparison_window_summary.csv`
- `raw_residual_structural_difference_detail.csv`
- `node_metrics/`

## Step 4

- `industry_community_validation_window_summary.csv`
- `raw_residual_industry_community_difference_window_summary.csv`
- `raw_residual_community_partition_window_summary.csv`
- `community_membership/`

## Step 5

- `dynamic_stability_window_summary.csv`
- `raw_residual_dynamic_difference_window_summary.csv`
- `dynamic_stability_detail.csv`

## Step 6

- `sparsification_robustness_pass_table.csv`
- `density_structural_robustness_summary.csv`
- `density_raw_residual_overlap_summary.csv`
- `density_dynamic_stability_summary.csv`
- `density_same_cross_stability_summary.csv`
- `per_node_topk_exactness_summary.csv`

---

# 11. 下一步研究计划

今天已经基本完成：

\[
\boxed{
\textbf{Full-Market Network Construction + Structural Validation}
}
\]

下一阶段可以正式从“验证网络是否合理”进入：

\[
\boxed{
\textbf{Network Information Mining}
}
\]

建议依次推进：

1. 构造 node-level network factor panel；
2. 研究 Degree、Strength、Residual Centrality 等指标；
3. 区分 stable industry backbone 与 dynamic cross-industry links；
4. 构造网络 Alpha candidates；
5. 构造 network-based risk factors；
6. 检验网络特征对未来收益、波动率、下行风险和 tail risk 的解释能力；
7. 后续加入 sparse conditional network 作为方法 challenger。

---

# 12. 汇报用一句话总结

> **今天完成了 A 股全市场 Raw 与 Market-Residual 关联网络的正式构建、结构诊断、行业与社区验证、动态稳定性分析以及 sparsification robustness。结果显示，去除市场共同波动后，网络发生显著而稳定的 topology reorganization：Residual Network 的 strongest edges 更广泛分布于市场、hub concentration 更低、行业关联更强、同行业边更加持久，而且这些结论在 \(q=0.2\%-2\%\) 的广泛网络密度范围内基本保持。**
