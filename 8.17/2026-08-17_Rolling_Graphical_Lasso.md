# 2026-08-17 量化实习工作总结
## 主题：股票关联网络的动态分析——Rolling Graphical Lasso、网络持续性与行业结构演化

---

## 一、今日工作概述

今日工作在前期静态股票关联网络分析的基础上，进一步转向**动态股票条件关联网络**。核心目标是利用 Rolling Graphical Lasso 研究股票之间条件关联关系随时间的演化，并重点回答以下问题：

1. 哪些股票关系能够长期持续存在？
2. 哪些关系只在特定时期出现？
3. 哪些股票的网络中心性随时间变化最明显？
4. 哪些时间阶段网络结构变化最明显？
5. 静态重采样稳定边是否同时具有较强的时间持续性？
6. 同行业边与跨行业边在动态稳定性和网络重构中的作用是否不同？

今日主要完成了以下工作：

- 对前期 block-resampling 稳定性分析进行 block length 鲁棒性检验；
- 构建 Rolling Graphical Lasso 动态网络；
- 分析动态网络的整体拓扑变化；
- 计算 edge persistence，识别长期持续边与阶段性边；
- 分析股票节点 Degree 与 Strength 的动态变化；
- 选择代表性时间窗口绘制动态网络；
- 构造相邻网络 Jaccard similarity 与 turnover 指标；
- 比较 Static Stable 与 Dynamic Persistent 两类稳定性；
- 修复股票名称/行业映射 bug；
- 对同行业边与跨行业边进行动态分解；
- 形成“稳定行业核心 + 动态跨行业外围”的阶段性解释框架。

---

## 二、Block Length 鲁棒性检验

### 2.1 研究目的

此前的稳定性分析采用 moving-block resampling，初始 block length 设为：

\[
L=20.
\]

为了检验 Stable Graphical Lasso 网络是否依赖于这一人为设定，今日进一步固定其他参数，仅改变 block length：

\[
L\in\{10,20,40\}.
\]

固定条件包括：

- Graphical Lasso 正则化参数：\(\alpha=\alpha_{1SE}\approx0.2169\)；
- 每个 block length 的重采样次数：\(B=200\)；
- subsample ratio：0.8；
- 相同的 solver、收敛容忍度与标准化方案。

三种 block length 共进行了：

\[
3\times200=600
\]

次 Graphical Lasso 拟合。

### 2.2 主要结果

全部 600/600 次拟合均正常收敛。

三种 block length 下，满足 selection frequency 不低于 0.8 的稳定边数量均为：

\[
\boxed{41}.
\]

更重要的是，三种 block length 下得到的 41 条 stable edges 完全相同：

\[
J(10,20)=J(10,40)=J(20,40)=1.
\]

因此：

\[
\boxed{
E^{stable}_{L=10}
=
E^{stable}_{L=20}
=
E^{stable}_{L=40}.
}
\]

前期识别出的 9 条跨方法强边，在三种 block length、共 600 次重采样中均被选择，selection frequency = 1。

### 2.3 阶段性结论

当前静态 Stable GLasso 网络表现出很强的 block-length robustness。在本次实习分析中，没有必要继续大规模扩展 block length 网格。现有结果已经足以支持：41 条 stable edges 对合理范围内的 block length 设定具有很强稳健性。

---

## 三、Rolling Graphical Lasso 动态网络

### 3.1 动态网络的研究动机

此前静态 Graphical Lasso 使用整个样本期估计一个统一的条件关联网络 \(G=(V,E)\)。这种方法回答的是长期总体关系，但可能掩盖不同时间阶段之间的网络结构差异。

因此今日进一步构造：

\[
G_1,G_2,\ldots,G_K,
\]

即随时间变化的 Rolling Graphical Lasso 网络。

### 3.2 Rolling Window 设置

采用：

\[
W=252,
\]

即约一个交易年的窗口长度；步长：

\[
STEP=20,
\]

即约每月更新一次网络。

Graphical Lasso 正则化参数固定为：

\[
\alpha=\alpha_{1SE}\approx0.2169.
\]

固定 \(\alpha\) 的目的在于：使不同时间窗口之间的网络变化主要反映数据本身的时间变化，而不是每个窗口重新调参带来的额外变化。

每个 rolling window 内重新对收益率进行标准化。

### 3.3 动态网络计算结果

共得到 31 个 rolling windows，网络日期范围约为 2024-01-16 至 2026-07-14。全部 31 个窗口均正常收敛。

边数范围：

\[
40\sim57,
\]

平均边数约 50.48。

密度范围：

\[
0.381\sim0.543,
\]

平均密度约 0.481。

---

## 四、动态网络整体结构变化

### 4.1 边数量的时间变化

2024–2025 年的大多数网络包含约 50–57 条边。2025 年末开始，网络明显变稀疏。

例如：

- 2025-10-16：54 edges；
- 2025-11-13：44 edges。

一次减少 10 条边。

后续部分窗口进一步下降到 40 条边。因此，从边数角度看，2025 年末是当前样本中最明显的网络稀疏化阶段之一。

### 4.2 平均条件关联强度

网络边数下降的同时，平均绝对 GLasso partial correlation 并未同步下降。

例如从 54 条边下降到 44 条边时：

\[
Mean|\rho|:
0.1153\rightarrow0.1311.
\]

说明该阶段更接近“弱关系退出，而剩余关系平均更强”，而不是所有股票关系整体同步变弱。

### 4.3 同行业边比例变化

同行业边比例大致在 30.4%–46.3% 之间变化。后期网络变稀疏时，同行业边比例明显上升。该现象在后续行业分解分析中得到进一步解释。

---

## 五、Edge Persistence：长期持续边与阶段性边

### 5.1 Persistence 定义

对每条股票边定义：

\[
Persistence_{ij}
=
\frac{1}{K}
\sum_{k=1}^{K}
I\{(i,j)\in E_k\},
\]

其中 \(K=31\)。

因此：

- \(Persistence=1\)：31 个窗口中全部存在；
- \(Persistence\ge0.8\)：在绝大多数窗口中存在；
- 较低 persistence：更可能具有阶段性或外围特征。

### 5.2 长期持续边

当前结果中：

- 23 条边满足 \(Persistence=1\)；
- 40 条边满足 \(Persistence\ge0.8\)。

前期识别出的 9 条跨方法强边全部满足：

\[
\boxed{Persistence=1}.
\]

说明这些关系不仅在静态分析和 resampling 中稳定，也在整个 rolling 时间序列中始终存在。

### 5.3 阶段性边

部分边只在特定连续时期内出现，例如：

- 五粮液–立讯精密；
- 伊利股份–赣锋锂业；
- 农业银行–赣锋锂业；
- 立讯精密–中国平安；
- 贵州茅台–海康威视等。

这类边通常 \(Persistence<0.5\)，且平均条件关联强度相对较弱。

当前结果显示：

\[
\boxed{
\text{动态变化主要集中在外围弱/中等关系，而非最核心的强关系。}
}
\]

因此动态网络更接近：

\[
\boxed{
\text{Stable Core + Dynamic Periphery}.
}
\]

---

## 六、股票中心性的动态变化

今日主要使用 Degree 与 Strength 研究股票网络角色随时间的变化，其中：

\[
Strength_i(t)
=
\sum_j |\rho^{GL}_{ij,t}|.
\]

### 6.1 Degree 变化较大的股票

Degree 时间波动最明显的股票包括：

1. 京东方 A；
2. 赣锋锂业；
3. 泸州老窖；
4. 海康威视；
5. 中国平安。

其中京东方 A 的 Degree 在 3–11 之间变化，范围最大，可视为 regime-dependent central node。

### 6.2 Strength 变化较大的股票

Strength 波动较明显的股票包括：

- 赣锋锂业；
- 海康威视；
- 京东方 A；
- 中信证券；
- 韦尔股份。

Degree 与 Strength 不完全一致，说明一只股票连接数量多，并不代表其所有条件关联都很强。

### 6.3 稳定核心节点

中信证券平均 Degree 约 9.10，为样本中最高水平之一，但波动范围较小，为 8–11。因此中信证券更接近“长期高连接、相对稳定的核心节点”。

---

## 七、代表性动态网络绘制

为了直观比较网络结构变化，今日重点比较：

\[
Window~22
\rightarrow
Window~23
\rightarrow
Window~26.
\]

对应结果为：

| Window | Network Date | Edges | Density |
|---|---|---:|---:|
| 22 | 2025-10-16 | 54 | 0.514 |
| 23 | 2025-11-13 | 44 | 0.419 |
| 26 | 2026-02-09 | 40 | 0.381 |

绘图中采用：

- 相同节点位置；
- 节点颜色表示行业；
- 节点大小表示 Degree；
- 边宽表示绝对 GLasso partial correlation；
- 深色边表示多个代表窗口均存在的关系；
- 浅色边表示时间变化边。

这样可以避免不同 spring layout 导致视觉上的伪变化。

---

## 八、相邻网络变化率：Jaccard 与 Turnover

### 8.1 定义

对相邻窗口 \(E_{t-1},E_t\)，定义：

\[
J_t
=
\frac{|E_{t-1}\cap E_t|}
{|E_{t-1}\cup E_t|},
\]

进一步定义：

\[
Turnover_t
=
1-J_t.
\]

同时记录：

\[
Lost_t
=
|E_{t-1}\setminus E_t|,
\]

\[
Gained_t
=
|E_t\setminus E_{t-1}|.
\]

### 8.2 为什么不能只看边数变化

若前后网络边数相同，仍可能发生大量边替换。因此：

\[
\boxed{
\Delta|E_t|
\neq
\text{网络结构变化程度}.
}
\]

### 8.3 Turnover 最大的几个时期

| 排名 | 日期区间 | Edges | Lost | Gained | Turnover |
|---:|---|---:|---:|---:|---:|
| 1 | 2024-01-16 → 2024-02-21 | 52 → 51 | 6 | 5 | 0.1930 |
| 2 | 2026-04-15 → 2026-05-18 | 45 → 42 | 6 | 3 | 0.1875 |
| 3 | 2024-08-15 → 2024-09-12 | 53 → 54 | 5 | 6 | 0.1864 |
| 4 | 2025-10-16 → 2025-11-13 | 54 → 44 | 10 | 0 | 0.1852 |
| 5 | 2024-09-12 → 2024-10-21 | 54 → 55 | 5 | 6 | 0.1833 |

### 8.4 三类动态变化机制

1. **Network Rewiring**：Lost 与 Gained 接近，边数变化不大，但具体关系发生替换；
2. **Network Sparsification**：Lost 明显大于 Gained，网络净稀疏化；
3. **Sparsification + Rewiring**：既有净删边，也有部分新关系进入。

---

## 九、Static Stable vs Dynamic Persistent

### 9.1 两种稳定性的区别

Static Stable 研究的是样本扰动下的稳定性；Dynamic Persistent 研究的是时间维度上的持续性。两者不是同一个概念。

### 9.2 核心结果

- Static Stable：41 条边；
- Dynamic Persistent（\(Persistence\ge0.8\)）：40 条边；
- 两者共同边：37 条；
- Jaccard similarity：0.8409；
- Static Stable 中同时满足 Dynamic Persistent 的比例：
  \[
  \frac{37}{41}=90.24\%;
  \]
- Dynamic Persistent 中得到 Static Stable 支持的比例：
  \[
  \frac{37}{40}=92.50\%.
  \]

说明：

\[
\boxed{
\text{Sample Robustness 与 Temporal Persistence 高度一致。}
}
\]

### 9.3 Always Persistent 与 Static Stable

23 条满足 \(Persistence=1\) 的长期持续边全部属于 Static Stable：

\[
\boxed{
E_{\mathrm{AlwaysPersistent}}
\subseteq
E_{\mathrm{StaticStable}}.
}
\]

因此形成：

\[
41
\rightarrow
37
\rightarrow
23
\]

的层级：

\[
\boxed{
\text{Static Stable}
\supset
\text{Stable + Persistent}
\supset
\text{Stable + Always Persistent}.
}
\]

### 9.4 Static-only 与 Dynamic-only

仅有少数边出现两种稳定性定义不一致：

- Static Stable only：4 条；
- Dynamic Persistent only：3 条。

Static-only 边的 persistence 仍约为 0.742–0.774，仅略低于 0.8 工作阈值；Dynamic-only 边则表现为“时间上经常出现，但对重采样扰动较敏感”。

---

## 十、股票名称/行业映射 Bug 修复

在 Static Stable 与 Dynamic Persistent 合并分析中发现股票名称和行业出现错位。

问题原因是：对无向边执行 `canonicalize_pair()` 后，stock_1 和 stock_2 的顺序发生变化，但 name_1/name_2 和 industry_1/industry_2 没有同步交换。

修复方案：

1. 在 pair canonicalization 之前建立：
   \[
   \text{code}\rightarrow(\text{name},\text{industry})
   \]
   映射；
2. canonicalize 股票代码；
3. 再根据新的 stock_1、stock_2 重新生成名称与行业；
4. 后续不再使用旧的 name_1/name_2、industry_1/industry_2。

核心数量结果不受该 bug 影响，但行业相关分析必须使用修复后的映射。

---

## 十一、同行业边与跨行业边的动态分解

### 11.1 同行业比例增加主要由 Cross 减少造成

31 个 rolling windows 中，同行业边数量基本维持在：

\[
17\sim19,
\]

而跨行业边变化范围更大：

\[
22\sim39.
\]

例如：

- 2025-10-16：Same = 19，Cross = 35；
- 2026-06-15：Same = 19，Cross = 22。

同行业边数量没有增加：

\[
19\rightarrow19,
\]

但跨行业边：

\[
35\rightarrow22.
\]

与此同时 Same-industry ratio：

\[
35.19\%
\rightarrow
46.34\%.
\]

因此：

\[
\boxed{
\text{同行业比例的上升主要由跨行业边数量下降驱动，而不是同行业边数量明显增加。}
}
\]

### 11.2 同行业边比跨行业边长期更加稳定

Persistence 汇总结果：

| 指标 | Same Industry | Cross Industry |
|---|---:|---:|
| 候选股票对 | 19 | 86 |
| Mean Persistence | **0.968** | **0.373** |
| Median Persistence | **1.000** | **0.210** |
| Persistence ≥ 0.8 | 18 | 22 |
| Persistence ≥ 0.8 比例 | **94.74%** | **25.58%** |
| Persistence = 1 | 15 | 8 |
| Persistence = 1 比例 | **78.95%** | **9.30%** |

同行业边成为高持续边的比例约为跨行业边的 3.7 倍。

此外，同行业边被选择时的平均绝对 partial correlation 也明显高于跨行业边。因此：

\[
\boxed{
\text{同行业关系不仅更稳定，而且平均条件关联也更强。}
}
\]

### 11.3 动态变化主要发生在跨行业关系

30 次相邻窗口变化中：

- 总退出 85 次，其中 Same = 5，Cross = 80；
- 总新增 73 次，其中 Same = 5，Cross = 68；
- 总 edge-change events = 158，其中 Same = 10，Cross = 148。

因此：

\[
\boxed{
\frac{148}{158}=93.67\%
}
\]

的边状态变化来自跨行业关系。

进一步按候选股票对数 × 30 次相邻窗口标准化：

- 同行业状态变化率约 1.75%；
- 跨行业约 5.74%。

因此单个跨行业股票对发生状态变化的频率约为同行业的：

\[
\boxed{3.27\text{ 倍}}.
\]

所以“动态变化主要发生在跨行业关系”并不仅仅是因为跨行业候选边更多。

---

## 十二、今日形成的核心认识

综合 Static Stability、Dynamic Persistence、Turnover 与 Industry Decomposition，当前结果支持以下动态股票网络结构：

\[
\boxed{
G_t
\approx
G_{\mathrm{stable\ industry\ core}}
+
G_{\mathrm{dynamic\ cross-industry\ periphery},t}.
}
\]

即：

\[
\boxed{
\textbf{稳定行业核心 + 动态跨行业外围}
}
\]

具体表现为：

1. 最核心的强条件关联在静态、重采样与动态分析中均高度稳定；
2. 行业内股票关系具有更高 Persistence；
3. 同行业边数量随时间相对稳定；
4. 跨行业边具有更低 Persistence；
5. 网络 Turnover 的绝大部分来自跨行业边进入和退出；
6. 后期同行业边比例提高主要来自跨行业边减少，而非同行业边大量增加；
7. 动态网络整体并非完全重构，而是在较稳定核心骨架周围发生外围调整。

---

## 十三、解释时需要保留的限定

### 13.1 Rolling windows 并非独立

当前：

\[
W=252,\qquad STEP=20.
\]

相邻窗口重叠：

\[
232/252\approx92.1\%.
\]

因此 Persistence、Turnover 等应理解为描述性动态网络指标，不能直接视为独立样本概率或普通独立观测。

### 13.2 Graphical Lasso 网络不是因果网络

当前边代表 contemporaneous conditional association，不能直接解释为因果关系、风险传染方向或“谁影响谁”。

### 13.3 Static Stable 与 Dynamic Persistent 含义不同

前者研究 sample perturbation robustness，后者研究 temporal persistence。两者高度一致是当前数据中的重要经验结果，而不是定义上的必然关系。

---

## 十四、今日主要输出文件

### Block-length Robustness

- `block_length_edge_stability.csv`
- `block_length_robustness_summary.csv`
- `block_length_pairwise_jaccard.csv`
- `block_length_robust_edges.csv`
- `strong_common_edge_block_robustness.csv`
- `block_length_run_diagnostics.csv`

### Rolling Graphical Lasso

- `rolling_glasso_network_summary.csv`
- `rolling_glasso_node_metrics.csv`
- `rolling_glasso_edge_history.csv`
- `rolling_glasso_diagnostics.csv`

### Representative Networks

- `rolling_glasso_representative_network_comparison.png`
- `representative_window_network_comparison.csv`
- `representative_window_edge_changes.csv`

### Adjacent Network Turnover

- `rolling_glasso_adjacent_network_turnover.csv`
- `rolling_glasso_adjacent_edge_changes.csv`
- `rolling_glasso_top_network_changes.csv`

### Static Stable vs Dynamic Persistent

- `static_stable_vs_dynamic_persistent_all_pairs.csv`
- `static_stable_vs_dynamic_persistent_summary.csv`
- `static_stable_dynamic_persistent_common_edges.csv`
- `static_stable_but_not_dynamic_persistent.csv`
- `dynamic_persistent_but_not_static_stable.csv`
- `static_stable_always_persistent_edges.csv`

### Industry Dynamic Decomposition

- `rolling_glasso_industry_edge_decomposition.csv`
- `rolling_glasso_edge_persistence_by_industry.csv`
- `rolling_glasso_industry_persistence_summary.csv`
- `rolling_glasso_industry_edge_turnover.csv`

---

## 十五、今日工作总结

今日工作将前期静态股票条件关联网络进一步扩展到动态场景。

整个分析链条由：

\[
\boxed{
\text{Static GLasso}
\rightarrow
\text{Static Stability}
\rightarrow
\text{Block-length Robustness}
\rightarrow
\text{Rolling GLasso}
\rightarrow
\text{Edge Persistence}
\rightarrow
\text{Node Dynamics}
\rightarrow
\text{Network Turnover}
\rightarrow
\text{Static/Dynamic Stability Comparison}
\rightarrow
\text{Industry Dynamic Decomposition}.
}
\]

当前阶段最重要的结论是：

> 股票条件关联网络并非在不同时间窗口中完全重新生成，而是表现出一个相对稳定的核心结构。行业内部关系具有非常高的时间持续性，而网络的大部分动态调整主要发生在跨行业连接。后期同行业边比例上升主要来自跨行业关系的收缩，而非同行业关系数量明显增加。因此，当前样本下的股票动态条件关联网络可以描述为“稳定行业核心 + 动态跨行业外围”。

---

## 十六、下一步可继续研究的方向

1. **Window-size sensitivity**
   \[
   W\in\{126,252,504\}
   \]
   比较不同时间尺度下动态网络的稳定性与变化速度；

2. **Edge lifecycle**
   对每条边进一步计算 first appearance、last appearance、longest consecutive run、number of episodes；

3. **中心性动态稳定性**
   构造：
   \[
   CV_i^{Degree}
   =
   \frac{SD(Degree_i)}
   {Mean(Degree_i)}
   \]
   以及 Strength 的对应指标；

4. **更正式的动态精度矩阵模型**
   进一步了解 Fused Graphical Lasso、Time-Varying Graphical Lasso、dynamic precision matrix estimation；

5. **方向性网络**
   如研究“谁影响谁”，可进一步考虑 VAR / Granger causality 等方法。

---

**日期：2026-08-17**  
**研究主题：股票关联网络动态分析与稳定结构识别**
