# M1 A股全市场股票关联网络研究

**日期：2026-09-17**  
**主题：Network Feature Engineering & Factor Panel Construction**  
**研究主线：从“验证网络结构”进一步推进到“构造股票级网络因子并完成初步去冗余”**

---

## 1. 今日研究目标

在前期已经完成全市场 Raw Pearson Network 与 Market-Residual Network 的构造、结构诊断、行业验证、动态稳定性与稀疏化稳健性分析的基础上，今天的主要任务是将网络层面的结论进一步转化为**股票层面的可研究因子**。

今日核心目标包括：

1. 冻结 Network Factor Design，避免后续因未来收益结果反向修改因子定义；
2. 将 Raw / Residual Network 转换为统一的股票级基础网络特征面板；
3. 将 Residual Network 拆分为行业内部与跨行业连接；
4. 构造股票级动态网络特征；
5. 构造 Community / Bridge 特征；
6. 对候选因子完成 pre-outcome QA、相关性分析与初步去冗余；
7. 最终形成一套可进入后续 Alpha / Risk Validation 的精简 Network Factor Library。

今日整体流程为：

\[
\boxed{
\text{Frozen Design}
\rightarrow
\text{Base Node Panel}
\rightarrow
\text{Industry Decomposition}
\rightarrow
\text{Dynamic Features}
\rightarrow
\text{Community / Bridge Features}
\rightarrow
\text{Factor QA \& Screening}
}
\]

---

# 2. Step 1：Freeze Factor Design

## 2.1 主要任务

首先冻结第5天的网络因子研究设计，明确后续所有特征均基于相同的研究口径。

主网络设定为：

- **Primary Network**：`M1_RESIDUAL_POSITIVE`
- **Benchmark Network**：`B0_RAW_POSITIVE`
- 滚动窗口：
  \[
  W\in\{60,120,252\}
  \]
- 主稀疏度：
  \[
  q=1\%
  \]
- 稳健性稀疏度：
  \[
  q\in\{0.2\%,0.5\%,1\%,2\%\}
  \]
- 主行业分类：`industry_id1`
- 更细行业分类 `industry_id2` 仅作为 robustness
- 分析频率：月末网络日期
- 动态特征仅使用 \(t\) 与 \(t-1\)
- 后续未来收益/风险标签必须严格从分析日之后开始

冻结后的设计哈希为：

```text
b2f66e9ac2c999e7ceeef0ed453b3bf500fa727eb9913cccfac0b77b8a9caa8e
```

共形成：

\[
\boxed{402}
\]

个 Date × Window 网络任务。

## 2.2 主要输出

输出目录：

```text
D:\M1_StockNetwork\output\M1_day4\01_step1_factor_design
```

主要文件包括：

- `factor_design_config.json`
- `factor_analysis_dates.csv`
- `network_factor_dictionary.csv`
- `future_target_dictionary.csv`
- `factor_input_inventory.csv`
- `factor_input_inventory_summary.csv`
- `factor_design_summary.md`

## 2.3 Step 1 结论

通过 Step 1，后续所有因子构造均基于**预先冻结、且尚未使用未来收益/风险结果**的研究设计，从源头上减少了因结果驱动的研究者自由度。

---

# 3. Step 2：Build Base Node Feature Panel

## 3.1 研究目的

将 Day 4 的 Raw Network 与 Residual Network 从网络层面转换为统一的股票级特征面板。

股票级主键：

\[
\boxed{
(analysis\_date,\ window,\ master\_index)
}
\]

主要构造：

- `raw_degree`
- `raw_strength`
- `raw_degree_percentile`
- `raw_strength_percentile`
- `residual_degree`
- `residual_strength`
- `residual_degree_percentile`
- `residual_strength_percentile`
- `delta_degree_percentile`
- `delta_strength_percentile`
- isolate / giant-component 状态

其中：

\[
\Delta DegreePct_{i,t}
=
DegreePct^{Residual}_{i,t}
-
DegreePct^{Raw}_{i,t}.
\]

该指标用于衡量股票在去除 Market Mode 后的网络地位重排。

---

## 3.2 数据质量

Step 2 共运行：

\[
402
\]

个 Date × Window 任务，全部通过 QA：

\[
\boxed{402/402}
\]

关键检查包括：

- Node count 一致；
- Panel key 唯一；
- Raw / Residual Degree handshake：
  \[
  \sum_i d_i = 2E;
  \]
- Degree / Strength 非负；
- Degree percentile 可严格重构；
- Step 3 percentile 到当前 factor panel 的 lineage 一致；
- Strength percentile 的 float32 重排名差异仅作为 diagnostic，不再误判为 QA failure。

---

## 3.3 核心结果

由于 Raw 与 Residual 使用相同的 Node Universe 和 Edge Budget：

\[
\bar d^{Raw}
=
\bar d^{Residual}.
\]

但股票之间的连接分配发生了明显变化。

平均零 Degree 比例：

| Window | Raw | Residual |
|---|---:|---:|
| 60 | 30.49% | 2.02% |
| 120 | 32.33% | 4.36% |
| 252 | 31.33% | 8.95% |

说明 Raw Network 呈现：

\[
\boxed{
\text{大量外围股票 + 少数超级 hubs}
}
\]

而 Residual Network 的 strongest associations 更广泛地分布在市场中。

典型股票的 Raw → Residual Degree 排名绝对变化约为：

| Window | Median \(|\Delta DegreePct|\) |
|---|---:|
| 60 | 17.32% |
| 120 | 16.53% |
| 252 | 16.07% |

即：

\[
\boxed{
\text{典型股票去市场后网络排名移动约16\%-17\% percentile points}
}
\]

---

## 3.4 Step 2 结论

Step 2 表明，Market residualization 并不是简单改变平均连接数量，而是显著重新分配了 strongest network links。  
最值得后续继续研究的基础变量包括：

\[
\boxed{
ResidualDegreePct,\;
ResidualStrengthPct,\;
\Delta DegreePct,\;
\Delta StrengthPct
}
\]

---

# 4. Step 3：Industry Backbone Decomposition

## 4.1 研究目的

将 Residual Network 的股票级连接进一步拆分为：

\[
\boxed{
Degree_i^{Residual}
=
Degree_i^{Same}
+
Degree_i^{Cross}
+
Degree_i^{Unknown}
}
\]

以及：

\[
\boxed{
Strength_i^{Residual}
=
Strength_i^{Same}
+
Strength_i^{Cross}
+
Strength_i^{Unknown}.
}
\]

其中：

- Same：同行业连接；
- Cross：跨行业连接；
- Unknown：至少一端缺少有效行业标签。

本次正式分析使用 `industry_id1`。

---

## 4.2 QA

402 个 Date × Window 任务全部通过：

\[
\boxed{402/402}
\]

并且：

- 行业标签覆盖率：
  \[
  \boxed{100\%}
  \]
- Unknown Industry Edge Share：
  \[
  \boxed{0}
  \]
- Degree 分解最大误差：
  \[
  \boxed{0}
  \]
- Strength 最大重构误差约：
  \[
  1.53\times10^{-5}
  \]

---

## 4.3 核心结果

Residual strongest edges 中：

| Window | Same-industry | Cross-industry |
|---|---:|---:|
| 60 | 25.20% | 74.80% |
| 120 | 30.41% | 69.59% |
| 252 | 36.40% | 63.60% |

从绝对数量看，跨行业连接更多，但这并不意味着行业结构弱，因为 Day 4 已经发现 Same-industry edges 相对于随机行业基准存在明显 enrichment。

更重要的是，随着窗口变长：

\[
SameIndustryShare:
25.2\%
\rightarrow
30.4\%
\rightarrow
36.4\%.
\]

而：

\[
CrossIndustryShare:
74.8\%
\rightarrow
69.6\%
\rightarrow
63.6\%.
\]

因此长期网络中，行业内部骨架更加突出。

同时，Cross-industry Degree 的横截面分布非常厚尾，说明少量股票具有明显的跨行业 connector 属性。

---

## 4.4 Step 3 结论

Residual Network 可以有效分解为：

\[
\boxed{
\text{Industry Backbone}
+
\text{Cross-industry Connections}
}
\]

并且股票之间存在明显角色异质性：

- 一部分股票更像 **Industry Core**；
- 一部分股票则表现为 **Cross-industry Connector**。

最值得保留的股票级变量包括：

\[
SameIndustryDegreePct,
\quad
CrossIndustryDegreePct,
\quad
CrossIndustryDegreeRatio.
\]

---

# 5. Step 4：Dynamic Network Features

## 5.1 研究目的

将静态网络角色进一步扩展到时间变化维度。

主要构造：

\[
\Delta ResidualDegreePct_{i,t},
\]

\[
\Delta CrossIndustryDegreePct_{i,t},
\]

以及股票级邻居集合动态：

\[
Jaccard_{i,t}
=
\frac{
|N_i(t)\cap N_i(t-1)|
}{
|N_i(t)\cup N_i(t-1)|
},
\]

\[
Retention_{i,t}
=
\frac{
|N_i(t)\cap N_i(t-1)|
}{
|N_i(t-1)|
}.
\]

同时计算：

- New Links
- Lost Links
- Retained Links
- Neighbor Turnover

主动态指标均采用 **common-node-conditioned** 比较，以尽量排除股票进入/退出样本导致的伪动态。

---

## 5.2 QA

402 个任务全部通过：

\[
\boxed{402/402}
\]

有效相邻期转换数：

- \(W=60\)：137
- \(W=120\)：134
- \(W=252\)：128

平均可比较股票比例：

| Window | Comparable Share |
|---|---:|
| 60 | 97.94% |
| 120 | 98.42% |
| 252 | 98.63% |

说明股票池变化对动态指标的影响较小。

---

## 5.3 邻居稳定性结果

平均 Neighbor Jaccard：

| W | Mean Jaccard |
|---|---:|
| 60 | 0.219 |
| 120 | 0.402 |
| 252 | 0.588 |

平均 Neighbor Retention：

| W | Mean Retention |
|---|---:|
| 60 | 0.360 |
| 120 | 0.571 |
| 252 | 0.741 |

说明：

\[
\boxed{
\text{短期网络邻居重构较强，长期网络关系更加持续}
}
\]

但较长窗口具有更高样本重叠，因此跨窗口差异不能完全解释为经济机制。

---

## 5.4 Edge Rewiring

平均每只股票新增与消失邻居数：

| W | New | Lost |
|---|---:|---:|
| 60 | 21.77 | 21.61 |
| 120 | 13.08 | 12.84 |
| 252 | 6.82 | 6.55 |

因此：

\[
NewLinks \approx LostLinks.
\]

说明网络动态主要体现为：

\[
\boxed{
\text{Edge Rewiring}
}
\]

而不是所有股票连接数单方向增加或减少。

---

## 5.5 Cross-industry Role 更动态

平均绝对排名变化：

| W | \(|\Delta ResidualDegreePct|\) | \(|\Delta CrossDegreePct|\) |
|---|---:|---:|
| 60 | 0.155 | 0.170 |
| 120 | 0.092 | 0.106 |
| 252 | 0.053 | 0.064 |

三个窗口均有：

\[
\boxed{
|\Delta CrossIndustryDegreePct|
>
|\Delta ResidualDegreePct|
}
\]

说明跨行业网络角色比整体网络中心性更动态。

---

## 5.6 Step 4 结论

股票网络动态主要表现为：

\[
\boxed{
\text{Edge Rewiring + Cross-industry Role Reorganization}
}
\]

这一步进一步支持：

\[
\boxed{
\textbf{Stable Industry Backbone
+
Dynamic Cross-industry Relationships}
}
\]

值得继续保留的动态变量包括：

\[
\Delta ResidualDegreePct,
\quad
\Delta CrossIndustryDegreePct,
\quad
NeighborJaccard,
\quad
NeighborRetention.
\]

---

# 6. Step 5：Community / Bridge Features

## 6.1 研究目的

结合 Day 4 已完成的 Residual Louvain community detection，把股票的网络角色继续拆分为：

\[
\boxed{
\text{Within-community Connectivity}
+
\text{Outside-community Connectivity}
+
\text{Bridge Role}
}
\]

Community 与 Industry 的区别在于：

- Industry 是外部给定经济分类；
- Community 是由 Residual Network 自身识别的内生结构。

主要指标包括：

- `within_community_degree_percentile`
- `outside_community_degree_percentile`
- `outside_community_degree_ratio`
- `participation_coefficient`
- `participation_coefficient_percentile`
- `weighted_participation_coefficient`
- `external_community_count`

Participation Coefficient：

\[
P_i
=
1-
\sum_c
\left(
\frac{k_{ic}}{k_i}
\right)^2.
\]

它用于判断股票的连接是否分散到多个不同 communities。

---

## 6.2 QA

402 个 Date × Window 任务全部通过：

\[
\boxed{402/402}
\]

并且：

- Active node community coverage：
  \[
  100\%
  \]
- Unknown community edges：
  \[
  0
  \]
- Degree reconstruction error：
  \[
  0
  \]
- Strength reconstruction error：
  约 \(1.53\times10^{-5}\)

Community membership source inventory 也确认了每个任务的 community 文件 lineage 一致。

---

## 6.3 Community Internal Structure

平均 Within-community Edge Share：

| W | Within | Outside |
|---|---:|---:|
| 60 | 74.16% | 25.84% |
| 120 | 77.55% | 22.45% |
| 252 | 80.92% | 19.08% |

因此 Residual Network 具有很强的 endogenous community organization。

这一结果和 Step 3 的 Industry decomposition 不同：同行业边仅约 25%–36%，而同 community 边约 74%–81%。

因此：

\[
\boxed{
\text{Network Community 并非传统 Industry 的简单复制}
}
\]

---

## 6.4 Bridge Stocks

平均 Participation Coefficient：

\[
0.410,\quad
0.348,\quad
0.265
\]

对应 \(W=60,120,252\)。

Participation 的横截面 P95 约：

\[
0.719,\quad
0.690,\quad
0.640.
\]

说明虽然大部分 strongest residual links 位于自身 community 内，但仍存在一批股票同时连接多个不同 communities，形成明显的：

\[
\boxed{
\text{Cross-community Bridge Stocks}
}
\]

平均 External Community Count：

\[
2.75,\quad2.23,\quad1.63.
\]

部分上尾股票可同时连接约 4–6 个不同外部 communities。

---

## 6.5 Step 5 结论

Residual Network 同时具有：

\[
\boxed{
\text{Strong Community Internal Structure}
}
\]

和：

\[
\boxed{
\text{A Small Set of Cross-community Bridges}
}
\]

因此 Step 5 进一步把股票区分为：

- Community Core；
- Cross-community Bridge；
- Broad Structural Hub；
- Peripheral Stock。

---

# 7. Step 6：Factor QA & Initial Screening

## 7.1 研究目的

Step 6 不再继续增加网络因子，而是对 Step 2–5 的候选变量做统一的：

\[
\boxed{
Coverage
+
Variation
+
Temporal Stability
+
Redundancy
+
Interpretability
}
\]

检查。

特别强调：

\[
\boxed{
\text{本阶段不使用未来收益或未来风险结果}
}
\]

因此属于 pre-outcome screening。

---

## 7.2 候选因子筛选结果

初始候选变量：

\[
\boxed{29}
\]

最终筛选状态：

\[
\boxed{
29
=
9\ KEEP\_CORE
+
7\ KEEP\_AUXILIARY
+
11\ REVIEW\_REDUNDANCY
+
1\ REVIEW\_QA
+
1\ DROP
}
\]

---

## 7.3 最终 9 个核心候选因子

| 维度 | 因子 | 经济含义 |
|---|---|---|
| Centrality | `residual_degree_percentile` | 整体 Residual Network 中心性 |
| Market removal | `delta_degree_percentile` | 去市场后的中心性重排 |
| Industry | `cross_industry_degree_percentile` | 跨行业连接中心性 |
| Industry composition | `cross_industry_degree_ratio` | 网络连接中的跨行业占比 |
| Dynamic | `delta_residual_degree_percentile_1m` | 整体网络地位短期变化 |
| Dynamic industry | `delta_cross_industry_degree_percentile_1m` | 跨行业 connector 地位变化 |
| Neighbor dynamics | `neighbor_jaccard_1m` | 邻居集合重构程度 |
| Neighbor persistence | `neighbor_retention_1m` | 原网络关系持续性 |
| Community | `outside_community_degree_percentile` | 跨内生 community 连接中心性 |

---

## 7.4 Degree 与 Strength 高度冗余

发现大量 Degree / Strength 对的横截面排序几乎完全一致，例如：

\[
\rho_s(
ResidualDegreePct,
ResidualStrengthPct
)
\approx0.999,
\]

\[
\rho_s(
CrossIndustryDegreePct,
CrossIndustryStrengthPct
)
\approx0.999,
\]

\[
\rho_s(
\Delta DegreePct,
\Delta StrengthPct
)
\approx0.998.
\]

因此正式研究优先保留：

\[
\boxed{\text{Degree-based Factors}}
\]

而将 Strength 版本用于 robustness。

---

## 7.5 Community Bridge 指标也存在明显冗余

普通 Participation 与 Weighted Participation：

\[
|\rho_s|\approx0.998.
\]

Participation 与 Outside-community Degree Ratio：

\[
|\rho_s|\approx0.985.
\]

因此不同 community bridge 指标虽然定义不同，但股票横截面排序高度相似，不需要全部进入后续核心模型。

---

## 7.6 Level 与 Change 的时间特征明显不同

`residual_degree_percentile` 相邻期 temporal Spearman：

\[
0.743,\quad0.899,\quad0.966.
\]

`cross_industry_degree_percentile`：

\[
0.697,\quad0.871,\quad0.948.
\]

说明：

\[
\boxed{
\text{静态网络角色具有明显 persistence}
}
\]

而：

\[
\Delta ResidualDegreePct
\]

temporal Spearman 约：

\[
-0.107,\quad-0.011,\quad0.012.
\]

\[
\Delta CrossIndustryDegreePct
\]

约：

\[
-0.143,\quad-0.063,\quad-0.042.
\]

因此：

\[
\boxed{
\text{网络角色的变化本身并不持续}
}
\]

说明：

\[
\boxed{
Level \neq Change
}
\]

两类变量确实提供不同维度的信息。

---

## 7.7 QA 待检查项

`cross_industry_strength_ratio` 被标记为：

```text
REVIEW_QA
```

原因是出现了一些理论范围外的观测。

考虑到此前 Strength 存在 float32 保存精度问题，需要进一步检查：

\[
\max(
CrossIndustryStrengthRatio-1
)
\]

以及：

\[
\min(
CrossIndustryStrengthRatio
).
\]

但该指标与 `cross_industry_degree_ratio` 同时存在高度冗余，因此目前不影响核心因子集合。

---

# 8. 今日整体研究结论

经过 Step 1–6，今天已经完成了从“全市场网络结构”到“股票级网络因子库”的完整转换。

研究结果可以概括为以下四层结构：

### 8.1 Network Centrality

Market residualization 会显著重排股票网络地位：

\[
\boxed{
\text{典型股票 Raw→Residual 排名变化约16\%-17\%}
}
\]

---

### 8.2 Industry Structure

Residual Network 中同时存在：

\[
\boxed{
\text{Stable Industry Backbone}
}
\]

和大量：

\[
\boxed{
\text{Cross-industry Connections}
}
\]

并且少量股票具有明显的 Cross-industry Connector 特征。

---

### 8.3 Dynamic Structure

股票级网络关系存在明显：

\[
\boxed{
\text{Edge Rewiring}
}
\]

且：

\[
\boxed{
\text{Cross-industry Role 比整体 Centrality 更动态}
}
\]

---

### 8.4 Community Structure

Residual Network 具有明显的 endogenous community organization：

\[
74\%-81\%
\]

的 strongest residual edges 位于同一 community 内。

同时，一小部分股票充当：

\[
\boxed{
\text{Cross-community Bridges}
}
\]

连接多个不同的网络群组。

---

# 9. 今日最重要的综合认识

今天的结果进一步支持一个较为清晰的网络结构图景：

\[
\boxed{
\textbf{
Stable Local / Industry / Community Backbone
+
Dynamic Cross-group Reorganization
}
}
\]

股票的网络角色至少可以从四个维度描述：

\[
\boxed{
\text{Centrality}
+
\text{Composition}
+
\text{Dynamics}
+
\text{Bridge Role}
}
\]

最终通过 Step 6 的 pre-outcome screening，将 29 个候选变量压缩到 9 个主要核心候选因子，为后续正式预测研究提供了较为精简、经济含义清晰且去冗余后的 Network Factor Library。

---

# 10. 下一阶段建议

下一步建议正式冻结 Step 6 的 reduced factor set，然后进入：

## 10.1 Alpha Validation

构造严格的未来收益：

\[
R_{i,t\rightarrow t+1}
\]

并进行：

- Monthly Rank IC；
- ICIR；
- 五分位 / 十分位 portfolio sort；
- Top-minus-Bottom return；
- 行业中性化；
- Size neutralization；
- Turnover / coverage 检查；
- 不同 \(W\) 的稳健性比较。

---

## 10.2 Risk Validation

构造未来风险标签，例如：

\[
FutureVolatility,
\]

\[
DownsideVolatility,
\]

\[
MaximumDrawdown,
\]

\[
IdiosyncraticRisk.
\]

重点检验：

- Network centrality 是否对应未来风险集中；
- Emerging Cross-industry Connector 是否对应风险传播；
- Low Neighbor Jaccard 是否对应未来高波动 / 高不确定性；
- Community Bridge 是否与跨板块风险传导有关。

---

## 10.3 研究纪律

后续建议保持以下顺序：

\[
\boxed{
\text{Freeze Factor Set}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Construct Future Labels}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Alpha / Risk Validation}
}
\]

不再根据未来收益结果随意调整今天的因子定义，以降低数据挖掘和研究者自由度。

---

# 11. 今日主要产出

今日最终形成：

1. 一套冻结的 Network Factor Design；
2. 股票级 Base Node Feature Panel；
3. Industry Backbone / Cross-industry Factor Panel；
4. Dynamic Network Feature Panel；
5. Community / Bridge Feature Panel；
6. Factor QA / Correlation / Temporal Stability / Redundancy 结果；
7. 一套精简后的 9 个 pre-outcome core network factors。

最终实现：

\[
\boxed{
\text{Full-market Network}
\rightarrow
\text{Stock-level Network Features}
\rightarrow
\text{Reduced Network Factor Library}
}
\]

为下一阶段进入正式的 **Alpha Mining 与 Risk Factor Validation** 奠定了数据和方法基础。
