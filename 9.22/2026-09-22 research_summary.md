# M1 A股全市场股票关联网络：Market-Regime Conditional Validation / Economic Relevance

**日期：2026-09-23**  
**项目：M1 — A股全市场股票关联网络研究**  
**研究主题：Network State、Conditional Information 与 Implementation Capacity 的统一检验**

---

## 1. 今日研究目标

今天完成了 **Day 9 Step 1–6** 的完整研究流程，核心目标是回答：

> **股票网络特征所包含的未来收益/风险信息，是否随市场网络状态发生变化？这些状态变化是否同时影响策略实施容量？**

今日研究主线为：

\[
\boxed{
\text{Network State}
\rightarrow
\text{Conditional Alpha / Risk Information}
\rightarrow
\text{Implementation Capacity}
}
\]

整个流程坚持以下原则：

- 所有 regime 均采用 **PIT（point-in-time）** 构造，不使用未来信息；
- 不基于 Day 9 结果重新选择因子、窗口或方向；
- 不重新定义 Day 6/7 的 Alpha 或 Risk outcome；
- 不重新求解 Day 8 的 Capacity；
- 不因结果不符合预期而修改 regime score；
- Native sample 与 Common-Date sample 同时保留；
- 结果均解释为 **conditional association**，不作因果解释。

---

# 2. Step 1 — Market / Network State Panel

## 2.1 目标

构建每个调仓日、每个网络窗口 \(W\) 下的市场状态与网络状态面板，为后续 PIT regime classification 提供输入。

输出目录：

```text
D:\M1_StockNetwork\output\M1_day8\
01_stage1_market_network_state
```

核心输出：

```text
market_network_state_panel.parquet
market_network_state_panel.csv
daily_market_state.csv
market_network_state_qa.csv
day9_step1_metadata.json
```

---

## 2.2 市场状态变量

市场状态严格使用截至当前调仓日的信息。

基于全市场等权收益构造：

\[
R^{mkt}_{t,20},
\quad
R^{mkt}_{t,60},
\]

\[
Vol^{mkt}_{t,20},
\quad
Vol^{mkt}_{t,60},
\]

\[
Dispersion_{t,20},
\quad
Dispersion_{t,60},
\]

\[
Breadth_{t,20},
\quad
Breadth_{t,60}.
\]

具体变量：

- `market_return_20d`
- `market_return_60d`
- `market_vol_20d_ann`
- `market_vol_60d_ann`
- `market_dispersion_20d`
- `market_dispersion_60d`
- `market_breadth_20d`
- `market_breadth_60d`

所有 trailing window 均以当前市场日结束，不含未来数据。

---

## 2.3 网络状态变量

网络状态基于 Day 4 已冻结的 matched-density 网络结果，不重新计算网络。

主密度固定：

\[
q=0.01.
\]

核心状态变量：

- `market_r2_mean`
- `residual_edge_threshold`
- `residual_cross_industry_edge_share`
- `residual_industry_enrichment`
- `raw_residual_overlap_fraction`
- `residual_positive_pair_share`

并计算一个调仓期变化量及其绝对值，包括：

\[
\Delta R^2,
\quad
\Delta Threshold,
\quad
\Delta CrossIndustryShare,
\quad
\Delta Overlap.
\]

---

## 2.4 QA 结果

Step 1：

\[
\boxed{\text{PASS + FROZEN}}
\]

主要 QA：

- panel rows = **402**
- unique analysis dates = **138**
- windows = **3**
- duplicate rows = **0**
- core network missing = **0**
- market state missing = **0**
- primary density \(q=0.01\) 全部满足
- future information used = **False**

三个网络窗口有效行数：

- W60：138
- W120：135
- W252：129

由于不同 \(W\) 下 residual threshold 等变量水平存在机械差异，因此后续 regime classification 必须在每个窗口内分别标准化。

---

# 3. Step 2 — PIT Regime Classification

## 3.1 目标

基于 Step 1 状态变量构造：

1. **Market Stress**
2. **Network Stress**
3. **Network Reconfiguration**

并使用严格历史信息进行 LOW / MEDIUM / HIGH 分类。

输出目录：

```text
D:\M1_StockNetwork\output\M1_day8\
02_stage2_pit_regime_classification
```

---

## 3.2 Market Stress

Market Stress 定义为：

\[
MarketStress_t
=
\frac14
\left[
Z(Vol_t)
+
Z(Dispersion_t)
-
Z(Return_t)
-
Z(Breadth_t)
\right].
\]

其中当前值的标准化只使用严格之前的历史：

\[
Z_t
=
\frac{X_t-\bar X_{1:t-1}}
{s_{1:t-1}}.
\]

---

## 3.3 Network Stress

Network Stress：

\[
NetworkStress_{t,W}
=
\frac14
\left[
Z(R^2)
+
Z(Threshold)
+
Z(CrossIndustryShare)
-
Z(Overlap)
\right].
\]

所有标准化均在每个 \(W\) 内单独进行。

---

## 3.4 Network Reconfiguration

定义：

\[
Reconfiguration_{t,W}
=
\frac14
\sum_k
Z(|\Delta N_{k,t,W}|),
\]

其中包括：

- \(|\Delta market\_r2\_mean|\)
- \(|\Delta residual\_edge\_threshold|\)
- \(|\Delta residual\_cross\_industry\_edge\_share|\)
- \(|\Delta raw\_residual\_overlap\_fraction|\)

---

## 3.5 分类方法

固定：

```text
MIN_Z_HISTORY = 24
MIN_SCORE_HISTORY = 12
LOW_Q = 1/3
HIGH_Q = 2/3
```

分类阈值同样只使用严格之前的历史分布，不使用全样本 `qcut`。

---

## 3.6 QA 与样本结构

Step 2：

\[
\boxed{\text{PASS + FROZEN}}
\]

主要 QA：

- input/output rows = **402**
- duplicate rows = **0**
- invalid label = **0**
- future/full-sample threshold used = **False**
- market regime classified rows = **306**
- network regime classified rows = **294**
- reconfiguration regime classified rows = **291**
- `all_formal_qa_pass = True`

Market regime 每个窗口：

- LOW = 34
- MEDIUM = 35
- HIGH = 33

不同网络窗口的分类数量不完全均衡，是 expanding-history PIT quantile 的正常结果，不进行事后强制平衡。

---

# 4. Step 3 — Conditional Alpha Validation

## 4.1 目标

检验冻结的网络因子与未来收益关系是否随 regime 改变。

不重新估计股票层面的 Alpha，而是复用 Day 7 已冻结的：

- monthly controlled rank IC
- monthly controlled Q5−Q1 spread

主 outcome：

\[
future\_excess\_return.
\]

分析维度：

\[
9\ factors
\times
3\ windows
\times
3\ methods.
\]

主控制规格：

\[
\boxed{\texttt{FULL\_CHARACTERISTIC\_NEUTRAL}}
\]

主 regime contrast：

\[
\boxed{
HIGH-LOW
}
\]

---

## 4.2 回归定义

对每个 factor-window-method：

\[
y_t
=
\beta_LD_{L,t}
+
\beta_MD_{M,t}
+
\beta_HD_{H,t}
+
u_t,
\]

其中：

\[
HIGH-LOW
=
\beta_H-\beta_L.
\]

使用 Newey–West HAC：

\[
L=3.
\]

同时保留：

- `NATIVE`
- `COMMON_DATE`

两个样本口径。

---

## 4.3 QA

Step 3 主要 QA：

- frozen factors = **9**
- IC groups = **81**
- spread groups = **81**
- duplicate keys = **0**
- result rows = **3888**
- Native rows = **1944**
- Common-date rows = **1944**
- factor sign flipped = **False**
- regime reestimated = **False**
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 4.4 Alpha 主要结果

### Network Stress

FULL controls + Native：

- HIGH−LOW 为负：**24/27**
- 为正：3/27
- nominal \(|t|\ge1.96\)：**0/27**
- 平均 HIGH−LOW：

\[
\boxed{-0.00716}
\]

说明方向上高 Network Stress 下很多 Alpha IC 更负，但整体差异推断证据较弱。

---

### Network Reconfiguration

FULL controls + Native：

- HIGH−LOW 为负：**21/27**
- nominal \(|t|\ge1.96\)：**10/27**
- 平均：

\[
\boxed{-0.01056}
\]

Common-date 下也约有：

\[
10/27
\]

个 nominally large contrasts。

因此：

\[
\boxed{
\text{Alpha relation 的状态异质性更多体现在 Network Reconfiguration。}
}
\]

但仍需注意多重检验问题。

---

# 5. Step 4 — Regime-Conditional Idiosyncratic-Risk Validation

## 5.1 目标

专门检验：

\[
\boxed{
NetworkStress/Reconfiguration
\times
future\_idio\_vol\_annualized
}
\]

是否存在稳定的条件异质性。

不重新估计 risk model，只复用 Day 7 冻结的 monthly controlled IVOL rank IC。

---

## 5.2 QA

Step 4：

- target = `future_idio_vol_annualized`
- frozen factors = **9**
- expected groups = **81**
- actual groups = **81**
- result rows = **1296**
- HIGH−LOW rows = **324**
- duplicate keys = **0**
- minimum regime component \(n\) = **18**
- small-sample result count = **0**
- regime reestimated = **False**
- risk target changed = **False**
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 5.3 `delta_degree_percentile × IVOL`

FULL controls + Native，Network Stress：

| Window | LOW IC | MEDIUM IC | HIGH IC | HIGH−LOW | HAC t | BH q |
|---|---:|---:|---:|---:|---:|---:|
| W60 | -0.0023 | 0.0128 | 0.0138 | +0.0161 | 1.54 | 0.319 |
| W120 | 0.0047 | 0.0329 | 0.0328 | **+0.0280** | **2.85** | **0.0597** |
| W252 | 0.0446 | 0.0511 | 0.0629 | +0.0183 | 1.52 | 0.319 |

三个窗口：

\[
\boxed{
IC^{IVOL}_{HIGH}
-
IC^{IVOL}_{LOW}
>0
}
\]

方向一致。

其中 W120 最明显。

但 Common-Date 下 W120：

\[
HIGH-LOW=0.0256,
\quad
t=2.54,
\quad
q_{BH}=0.149.
\]

因此更准确的解释是：

> `delta_degree_percentile` 与未来 IVOL 的正向关系在高 Network Stress 中倾向更强，但这种状态差异并未在所有窗口和多重检验调整后 uniformly 成立。

---

## 5.4 Reconfiguration 结果

对于 `delta_degree_percentile`，FULL controls + Native：

\[
W60:-0.0030,
\]

\[
W120:-0.0026,
\]

\[
W252:-0.0132.
\]

三个 HIGH−LOW 均为负，但均无强 adjusted evidence。

因此：

\[
\boxed{
\text{没有证据表明高 Network Reconfiguration 会系统增强 delta-degree 的 IVOL relation。}
}
\]

---

## 5.5 更重要的 state-robustness

`delta_degree_percentile` 在绝大多数 regime 中本身仍保持正 IVOL IC。

这说明：

\[
\boxed{
\text{delta-degree 与未来 IVOL 的关系具有较强的 state robustness。}
}
\]

它并不是只在极端 network state 中才存在。

---

## 5.6 其他显著异质性

一个较明显的结果来自：

\[
\texttt{cross\_industry\_degree\_ratio},
\quad W252.
\]

Network Stress：

\[
HIGH-LOW
=
-0.02495,
\]

\[
t=-3.50,
\]

\[
q_{BH}=0.0124.
\]

该结果在 Native 和 Common-Date 中一致，表明部分具体网络特征的 IVOL relation 确实存在明显状态变化。

---

# 6. Step 5 — Regime-Conditional Capacity Validation

## 6.1 目标

严格复用 Day 8 冻结的逐调仓 pair-level capacity：

\[
A^{cap}_{t,f,W},
\]

不重新求解 capacity。

主容量定义仍固定为：

\[
\boxed{
ADV20
+
5\%\ participation
+
1\text{-day execution}
+
95\%\ mean executable
}
\]

分析：

\[
A^{cap}_{t,f,W}
\mid
NetworkStress
\]

与：

\[
A^{cap}_{t,f,W}
\mid
Reconfiguration.
\]

---

## 6.2 QA

Step 5：

- expected specifications = **27**
- actual specifications = **27**
- capacity rows = **3556**
- duplicate keys = **0**
- summary rows = **324**
- HIGH−LOW contrast rows = **108**
- small-sample contrasts = **0**
- capacity reestimated = **False**
- capacity definition changed = **False**
- ADV definition changed = **False**
- participation threshold changed = **False**
- regime reestimated = **False**
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 6.3 Network Stress 与 Capacity

### Native sample

27 个规格平均：

\[
LOW:
10.22\text{亿元},
\]

\[
MEDIUM:
16.26\text{亿元},
\]

\[
HIGH:
16.75\text{亿元}.
\]

平均 HIGH−LOW：

\[
\boxed{+6.53\text{亿元}}.
\]

并且：

\[
\boxed{27/27}
\]

个规格 HIGH−LOW 均为正。

BH：

\[
q<0.10:
24/27.
\]

---

### Common-Date

\[
LOW:
10.29\text{亿元},
\]

\[
HIGH:
17.82\text{亿元}.
\]

平均 HIGH−LOW：

\[
\boxed{+7.52\text{亿元}}.
\]

并且：

\[
\boxed{27/27}
\]

均为正，且全部达到：

\[
q<0.05.
\]

因此：

\[
\boxed{
\text{High Network Stress 与更高的典型 rebalance-level capacity 相关。}
}
\]

注意：这是条件相关，不可解释为 Network Stress “提高”流动性。

---

## 6.4 Reconfiguration 与 Capacity

Native：

\[
HIGH-LOW
=
+0.29\text{亿元}.
\]

其中：

\[
11/27>0,
\qquad
16/27<0,
\]

且：

\[
0/27
\]

达到 \(q<0.10\)。

Common-Date：

\[
HIGH-LOW
=
+0.19\text{亿元},
\]

同样没有系统性 evidence。

因此：

\[
\boxed{
\text{Network Reconfiguration 与 Capacity 没有稳定的系统关系。}
}
\]

---

## 6.5 Capacity 解释注意事项

Step 5 使用的是：

\[
E(A_t^{cap}\mid Regime)
\]

或：

\[
Median(A_t^{cap}\mid Regime),
\]

而不是重新计算：

\[
A^{cap}_{Regime}
=
\sup
\left\{
A:
E[G_t(A)\mid Regime]\ge0.95
\right\}.
\]

因此“16.75亿元”应解释为：

> High Network Stress 条件下逐调仓容量的平均值。

不能称为：

> High Network Stress 下策略的正式容量上限。

---

# 7. Step 6 — Information–Implementation Summary

## 7.1 目标

将 Step 3–5 统一到：

\[
factor
\times
window
\times
regime
\]

框架。

核心变量：

\[
\Delta IC^\alpha
=
IC^\alpha_{HIGH}
-
IC^\alpha_{LOW},
\]

\[
\Delta IC^{IVOL}
=
IC^{IVOL}_{HIGH}
-
IC^{IVOL}_{LOW},
\]

\[
\Delta A^{cap}
=
A^{cap}_{HIGH}
-
A^{cap}_{LOW}.
\]

只使用：

\[
\boxed{
FULL\_CHARACTERISTIC\_NEUTRAL
}
\]

作为主结果。

---

## 7.2 QA

Step 6：

- Step 3 QA = PASS
- Step 4 QA = PASS
- Step 5 QA = PASS
- regime-level rows = **324**
- expected = **324**
- HIGH−LOW rows = **108**
- expected = **108**
- missing regime-level values = **0**
- missing core HIGH−LOW values = **0**
- new Alpha estimation = **False**
- new Risk estimation = **False**
- Capacity reestimated = **False**
- regime reestimated = **False**
- factor/window selected from summary = **False**
- future information used = **False**
- `all_formal_qa_pass = True`

---

# 8. Step 6 综合结果

## 8.1 Native sample

| Regime | Mean Alpha H−L | Alpha H−L < 0 | Mean IVOL H−L | IVOL H−L > 0 | Mean Capacity H−L |
|---|---:|---:|---:|---:|---:|
| Network Stress | **-0.00716** | 24/27 | **+0.00255** | 16/27 | **+6.53亿元** |
| Reconfiguration | **-0.01056** | 21/27 | **-0.00087** | 12/27 | **+0.29亿元** |

---

## 8.2 Common-Date

| Regime | Mean Alpha H−L | Alpha H−L < 0 | Mean IVOL H−L | IVOL H−L > 0 | Mean Capacity H−L |
|---|---:|---:|---:|---:|---:|
| Network Stress | **-0.00793** | 24/27 | **+0.00303** | 16/27 | **+7.52亿元** |
| Reconfiguration | **-0.01126** | 23/27 | **-0.00069** | 12/27 | **+0.19亿元** |

---

# 9. Information–Capacity Trade-off 是否成立？

原先可能存在的研究假设为：

\[
RiskInformation\uparrow
\quad\Rightarrow\quad
Capacity\downarrow.
\]

即：

\[
\Delta IC^{IVOL}>0,
\qquad
\Delta A^{cap}<0.
\]

但 Network Stress 下：

\[
\boxed{
27/27
}
\]

个 Capacity HIGH−LOW 都是正的。

其中 Native：

\[
16/27
\]

属于：

\[
RISK\_UP\_CAPACITY\_UP,
\]

另外：

\[
11/27
\]

属于：

\[
RISK\_DOWN\_CAPACITY\_UP.
\]

因此：

\[
\boxed{
RISK\_UP\_CAPACITY\_DOWN=0/27.
}
\]

Common-Date 完全一致。

所以：

\[
\boxed{
\text{当前结果不支持简单的 adverse information–capacity trade-off。}
}
\]

---

## 9.1 Reconfiguration 四象限

Reconfiguration 下：

- `RISK_UP_CAPACITY_UP` = 8
- `RISK_UP_CAPACITY_DOWN` = 4
- `RISK_DOWN_CAPACITY_UP` = 3
- `RISK_DOWN_CAPACITY_DOWN` = 12

真正满足：

\[
Risk\uparrow,\ Capacity\downarrow
\]

的只有：

\[
4/27.
\]

并且 Reconfiguration 的 capacity contrasts 本身没有 BH-adjusted evidence。

因此不能据此建立“网络重排导致信息—容量权衡”的主线。

---

# 10. `delta_degree_percentile` 的综合解释

此前重点关注的 `delta_degree_percentile` 在 Network Stress 下表现得尤其清楚。

例如 W120：

\[
\Delta IC^{IVOL}
=
+0.02805,
\]

\[
t=2.85,
\qquad
q=0.0597.
\]

同时：

\[
\Delta Capacity
=
+7.38\text{亿元},
\]

且 capacity evidence 较强。

因此该规格属于：

\[
\boxed{
RISK\_UP\_CAPACITY\_UP.
}
\]

所以并没有出现：

\[
\text{风险信息增强}
\Rightarrow
\text{更难实施}.
\]

而是：

\[
\boxed{
\text{高 Network Stress 下，IVOL relation 倾向更强，同时典型 capacity 也更高。}
}
\]

---

# 11. Native 与 Common-Date 稳健性

Step 6 中两个 sample scope 的 HIGH−LOW 结果高度一致。

### Network Stress

\[
Corr(
\Delta IC^{IVOL}_{Native},
\Delta IC^{IVOL}_{Common}
)
\approx0.993,
\]

方向一致：

\[
27/27.
\]

Capacity：

\[
Corr\approx0.957,
\]

方向一致：

\[
27/27.
\]

### Reconfiguration

IVOL：

\[
Corr\approx0.969.
\]

Capacity：

\[
Corr\approx0.999.
\]

因此主要结果并不是由不同网络窗口的样本起点差异所驱动。

---

# 12. 今日最重要的研究发现

今天的 Step 1–6 表明，**Network Stress 与 Network Reconfiguration 是两个性质不同的状态维度**。

可以概括为：

\[
\boxed{
\text{Network Reconfiguration}
\longleftrightarrow
\text{更明显的 Alpha relation 异质性}
}
\]

而：

\[
\boxed{
\text{Network Stress}
\longleftrightarrow
\text{部分 IVOL information 异质性}
}
\]

并且：

\[
\boxed{
\text{Network Stress}
\longleftrightarrow
\text{非常明显的 rebalance-level Capacity 异质性}.
}
\]

因此不应简单把二者统称为同一个 “network instability” 指标。

---

# 13. 当前最合理的 Day 9 研究结论

英文可表述为：

> **Network states are associated with distinct forms of economic heterogeneity. Short-run network reconfiguration is more closely associated with variation in conditional return relations, whereas network-stress levels exhibit clearer heterogeneity in future idiosyncratic-risk information and especially in rebalance-level implementation capacity. Importantly, the information and implementation dimensions do not exhibit a simple adverse trade-off: under high network stress, rebalance-level capacity is generally higher rather than lower, while the direction of conditional risk information remains factor dependent.**

中文可概括为：

> **网络状态对应不同形式的经济异质性。短期网络重排更多表现为收益关系的条件变化，而网络压力水平更多体现为未来特质风险信息及实施容量的状态差异。信息与实施能力之间并不存在简单的负向权衡：高 Network Stress 状态下逐调仓容量总体反而更高，而风险信息的变化方向依赖具体网络特征。**

---

# 14. 研究纪律与限制

今日结果仍需保留以下限制：

1. **所有结论均为条件相关关系，不作因果解释。**
2. Network Stress 与 Capacity 的强正相关可能部分受到 calendar-time / secular-liquidity composition 影响。
3. HIGH Network Stress 下 capacity 更高不能解释成 “Network Stress 提高流动性”。
4. Reconfiguration 下 Alpha 的 nominal significance 仍处于多重检验环境中。
5. `delta_degree_percentile` 的 IVOL relation 更适合解释为具有较强的 state robustness，而不是只在 high-stress state 有效。
6. Capacity conditional statistics 是 frozen rebalance-level capacity 的分组统计，不是重新求得的 regime-specific strategy capacity。
7. 不应根据 Day 9 结果重新选择 factor、window、sign 或 regime definition。

---

# 15. 今日工作完成状态

计算层面：

\[
\boxed{
\text{Day 9 Step 1--6 computationally complete}
}
\]

各步骤：

- Step 1 — Market / Network State Panel：**PASS + FROZEN**
- Step 2 — PIT Regime Classification：**PASS + FROZEN**
- Step 3 — Conditional Alpha Validation：**PASS**
- Step 4 — Conditional IVOL Validation：**PASS**
- Step 5 — Regime-Conditional Capacity Validation：**PASS**
- Step 6 — Information–Implementation Summary：**PASS**

如果此前已经人工确认 Step 3–5 所读取的物理路径均对应正式冻结版本，则可以进一步记录：

\[
\boxed{
\textbf{Day 9 = PASS + FROZEN}
}
\]

若物理路径 provenance 尚未最终确认，则更严谨的状态为：

\[
\boxed{
\textbf{Day 9 computationally complete, pending final provenance confirmation}
}
\]

---

# 16. 后续建议

Day 9 已经形成完整闭环，不建议继续增加新的 regime 回归或 outcome。

下一阶段建议围绕以下主线组织：

\[
\text{Network Structure}
\rightarrow
\text{Forward Risk Information}
\rightarrow
\text{State Dependence}
\rightarrow
\text{Implementation Capacity}.
\]

其中目前最稳健的总体结论仍然是：

> **股票在关联网络中的结构位置与网络重排包含传统股票特征之外的前瞻性风险信息；这种风险信息具有一定市场状态异质性，但并不存在一个简单的“信息越强、实施容量越差”的统一权衡。**

