# M1 全市场股票关联网络研究
**日期：2026-09-18**  
**主题：Forward Validation、Portfolio Sort、Risk Validation 与 Neutralization**

---

## 1. 今日研究目标

今天的工作围绕 **M1 全市场 A 股股票关联网络研究的 Day 6** 展开。核心目标是：

1. 在 **不使用未来信息重新定义因子** 的前提下，冻结 Day 5 已筛选出的网络因子；
2. 构造严格的未来收益与未来风险标签；
3. 检验网络因子的未来收益预测能力；
4. 通过组合排序检验经济意义；
5. 检验网络因子是否同时包含未来风险信息；
6. 控制行业与规模后，判断网络信号是否仍然存在；
7. 对 9 个冻结因子进行初步功能分类。

今天形成了一条完整的 forward-validation 研究链条：

\[
\boxed{
\text{Factor Freeze}
\rightarrow
\text{Forward Labels}
\rightarrow
\text{Rank IC}
\rightarrow
\text{Portfolio Sort}
\rightarrow
\text{Risk Validation}
\rightarrow
\text{Neutralization}
}
\]

---

# 2. Day 6 Step 1：Freeze Reduced Factor Set

## 2.1 目标

在查看任何未来收益或风险结果之前，先冻结用于 Day 6 的核心网络因子，避免后续 outcome-driven factor selection。

冻结后的核心因子共 9 个：

1. `residual_degree_percentile`
2. `delta_degree_percentile`
3. `cross_industry_degree_percentile`
4. `cross_industry_degree_ratio`
5. `delta_residual_degree_percentile_1m`
6. `delta_cross_industry_degree_percentile_1m`
7. `neighbor_jaccard_1m`
8. `neighbor_retention_1m`
9. `outside_community_degree_percentile`

Day 6 Step 1 输出目录：

```text
D:\M1_StockNetwork\output\M1_day5
\01_step1_freeze_reduced_factor_set
```

核心设计 hash：

```text
4827056a4ed1f87f79d1db92444063ee8217824b7c176c4f32cf4f74cfd34fdd
```

## 2.2 研究纪律

Step 1 明确冻结以下原则：

- 后续不根据未来收益重新选择因子；
- 不根据未来收益翻转因子方向；
- 不根据未来结果挑选“最佳窗口”；
- \(W=60,120,252\) 三个窗口均保留；
- Step 3–6 只能对冻结因子进行验证，不能反向修改因子定义。

---

# 3. Day 6 Step 2：Forward Return & Risk Labels

## 3.1 初始问题

最初 Step 2 使用 Day 1 的：

```text
return_network
```

作为未来日收益来源。

运行后发现：

- 一些 2015 年区间 future-return coverage 明显偏低；
- 例如：
  \[
  2015\text{-}06\text{-}30
  \rightarrow
  2015\text{-}07\text{-}31
  \]
  在 \(W=60\) 下平均 coverage 仅约：
  \[
  83.2\%.
  \]

进一步诊断发现，低 coverage 并不是随机缺失造成，而主要来自：

1. 停牌日；
2. 复牌首日；
3. `return_network` 的定义本身只允许“连续开市日对”。

---

## 3.2 Day 1 收益定义重新检查

今天重点回查了 Day 1 return panel 的构造逻辑，确认：

### `return_network`

定义为：

\[
\log
\left(
\frac{P^{adj}_{t}}
{P^{adj}_{t-1}}
\right)
\]

但仅当：

- 当前交易日开市；
- 前一个市场交易日也开市；
- 两日连续。

因此：

- 停牌日：`return_network = NA`
- 复牌首日：`return_network = NA`

这是 Day 1 有意设计，并不是 bug。

### `return_last_trade_simple`

Day 1 同时已经保存：

\[
\boxed{
return\_last\_trade\_simple
=
\frac{
P^{adj}_{t}
}{
P^{adj}_{\text{previous actual trade}}
}
-1
}
\]

它会跨过停牌期，直接保留复牌日从上一实际交易价到当前价格的累计变化。

---

## 3.3 Step 2 最终修正

最终将 Day 6 的 holding-period return 定义为：

\[
\boxed{
r^{hold}_{i,t}
=
\begin{cases}
0,
& isOpen_{i,t}=0,\\[4pt]
return\_last\_trade\_simple_{i,t},
& isOpen_{i,t}=1 \text{ 且该值可得},\\[4pt]
NA,
& \text{otherwise}.
\end{cases}
}
\]

解释：

- 明确停牌日：
  \[
  r=0
  \]
  表示持有者当日没有新的可交易价格更新；
- 正常交易日：
  使用 `return_last_trade_simple`；
- 复牌日：
  自动包含整个停牌期间的累计价格跳跃；
- IPO/首个实际交易日等无上一实际交易价情况：
  仍保持 NA，不人为设为 0。

---

## 3.4 修正后的 market benchmark

为避免股票收益与市场基准使用不同口径，市场收益也改为基于同一 holding-return convention 的：

\[
\boxed{
\text{LOO Equal-Weight Holding Return}
}
\]

即每日先在全市场 PIT 股票上构造 holding return，再对个股 \(i\) 使用：

\[
r^{mkt,-i}_t
=
\frac{
\sum_{j\neq i}
r^{hold}_{j,t}
}{
N_t-1
}.
\]

---

## 3.5 Step 2 最终结果

修正后：

- 402/402 个 Date × Window 任务全部通过 QA；
- 复牌股票日总数：
  \[
  17,919
  \]
- 成功恢复复牌收益：
  \[
  17,919
  \]
- unresolved reopen missing：
  \[
  0
  \]
- unresolved open missing：
  \[
  0
  \]

停牌日以 0 收益进入 holding path 的 stock-days 共：

\[
71,671
+
62,363
+
58,216
=
192,250.
\]

三个窗口平均 coverage：

| Window | Mean Return Coverage |
|---:|---:|
| 60 | 99.9994% |
| 120 | 99.9988% |
| 252 | 99.9934% |

原 2015 年低 coverage 问题基本消失。

### Step 2 结论

\[
\boxed{
\text{原低 coverage 主要由 suspension/reopen return convention 引起，}
}
\]

而不是随机数据缺失。

Step 2 最终形成两个清晰的收益概念：

### Network Return

用于 Day 2–5 网络估计：

\[
return\_network
=
\text{consecutive-open log return}.
\]

### Holding Return

用于 Day 6 forward validation：

\[
r^{hold}
=
0 \text{ on suspension}
+
return\_last\_trade\_simple
\text{ on active days}.
\]

---

# 4. Day 6 Step 3：Initial Alpha Screening — Rank IC

## 4.1 方法

对每个：

\[
analysis\_date
\times
W
\times
Factor
\]

计算月度横截面 Spearman Rank IC：

\[
\boxed{
IC_{t,W,F}
=
Corr_{\mathrm{Spearman},i}
\left(
F_{i,t,W},
R^{future}_{i,t}
\right)
}
\]

主 Alpha 标签：

```text
future_excess_return
```

Robustness：

```text
future_total_return
```

强调：

- 不做 pooled correlation；
- 每月单独计算横截面 IC；
- 不根据 IC 结果翻转符号；
- 不根据 IC 选择最佳窗口。

---

## 4.2 Step 3 主要结果

最突出因子：

\[
\boxed{
delta\_degree\_percentile
}
\]

其未来 excess-return Rank IC：

\[
W60:
-0.0508,
\]

\[
W120:
-0.0461,
\]

\[
W252:
-0.0389.
\]

HAC t-stat 约：

\[
-6.46,\quad
-5.12,\quad
-3.91.
\]

说明：

\[
\boxed{
deltaDegree\uparrow
\Rightarrow
FutureReturn\downarrow.
}
\]

这里的 `delta_degree_percentile` 表示：

\[
\boxed{
Residual\ Degree\ Percentile
-
Raw\ Degree\ Percentile
}
\]

因此它刻画的不是单纯 residual-network centrality，而是：

\[
\boxed{
\text{Raw network 到 residual network 的相对中心性重排}
}
\]

---

## 4.3 其他因子

### `residual_degree_percentile`

IC 基本为 0：

\[
0.0016,\ -0.0010,\ 0.0023.
\]

说明简单 residual centrality level 本身 Alpha 信息很弱。

### `delta_cross_industry_degree_percentile_1m`

短窗口有小幅正向 IC：

\[
0.0103,\quad
0.0088,\quad
0.0025.
\]

### `neighbor_jaccard_1m`

随着窗口增大而增强：

\[
0.0009
\rightarrow
0.0052
\rightarrow
0.0149.
\]

### `neighbor_retention_1m`

同样在较长窗口更明显：

\[
0.0051
\rightarrow
0.0083
\rightarrow
0.0092.
\]

---

## 4.4 Step 3 结论

总体结果表明：

\[
\boxed{
\text{Static Centrality Level}
\text{ 的 Alpha 信息有限}
}
\]

而：

\[
\boxed{
\text{Relative Network Rewiring}
+
\text{Network Dynamics}
}
\]

可能包含更多未来收益排序信息。

---

# 5. Day 6 Step 4：Portfolio Sort — Economic Significance

## 5.1 目的

Step 3 的 Rank IC 只是统计排序关系。

Step 4 进一步检验：

\[
\boxed{
\text{这种排序是否转化为具有经济幅度的组合收益差异？}
}
\]

每月按当期因子值构造：

\[
Q1,\ldots,Q5.
\]

统一定义：

\[
\boxed{
Spread
=
Q5-Q1.
}
\]

不根据因子方向自动改成 \(Q1-Q5\)。

---

## 5.2 `delta_degree_percentile` 的组合结果

### W = 60

未来 excess-return 五组约为：

\[
Q1=+47.6\text{ bp},
\]

\[
Q2=-2.9\text{ bp},
\]

\[
Q3=-19.1\text{ bp},
\]

\[
Q4=-46.3\text{ bp},
\]

\[
Q5=-77.0\text{ bp}.
\]

因此：

\[
Q5-Q1
=
-124.6\text{ bp/month}.
\]

对应：

\[
-14.96\%
\]

算术年化 spread。

### W = 120

\[
Q5-Q1
=
-114.9\text{ bp/month}.
\]

### W = 252

\[
Q5-Q1
=
-96.0\text{ bp/month}.
\]

三个窗口全部稳定为负。

---

## 5.3 时间稳定性

`delta_degree_percentile` 年度平均 spread 为负的年份数：

\[
W60:
10/12,
\]

\[
W120:
11/12,
\]

\[
W252:
10/11.
\]

删除 2015 年后，平均 spread 仍约为：

\[
-102.2,\quad
-106.8,\quad
-96.0
\text{ bp/month}.
\]

说明其结果不是单一极端市场阶段推动。

---

## 5.4 Rank IC 与 Portfolio Spread 的一致性

逐月 Rank IC 与 Q5−Q1 spread 的相关性约：

\[
0.937,\quad
0.955,\quad
0.959.
\]

说明：

\[
\boxed{
\text{Rank IC 与组合排序的经济结果高度一致。}
}
\]

---

## 5.5 Ties 问题

Step 4 还发现，部分离散网络因子存在大量 ties。

尤其：

```text
outside_community_degree_percentile
```

在 \(W=252\) 下只有约 20 个月能成功形成 5 个 genuine quintiles。

因此对于离散度较高的网络变量：

\[
\boxed{
Rank IC 结果比 quintile spread 更可靠。
}
\]

---

# 6. Day 6 Step 5：Risk Validation

## 6.1 目的

检验网络因子是否同时预测：

1. `future_realized_vol_annualized`
2. `future_downside_vol_annualized`
3. `future_max_drawdown`
4. `future_idio_vol_annualized`

即：

\[
\boxed{
F_{i,t}
\rightarrow
FutureRisk_{i,t}.
}
\]

---

## 6.2 `delta_degree_percentile` 的风险结果

该因子不仅对应更低未来收益，还对应更高未来风险。

### Total Vol

Risk IC：

\[
0.0272,\quad
0.0294,\quad
0.0296.
\]

### Downside Vol

\[
0.0401,\quad
0.0371,\quad
0.0328.
\]

### MDD

\[
0.0469,\quad
0.0470,\quad
0.0451.
\]

### IVOL

最强：

\[
\boxed{
0.1228,\quad
0.1328,\quad
0.1424.
}
\]

对应 Q5−Q1 IVOL 约：

\[
+5.87,\quad
+5.96,\quad
+5.73
\]

个百分点。

因此：

\[
\boxed{
deltaDegree\uparrow
\Rightarrow
\begin{cases}
FutureReturn\downarrow,\\
FutureIVOL\uparrow,\\
FutureMDD\uparrow,\\
FutureDownsideRisk\uparrow.
\end{cases}
}
\]

---

## 6.3 年度稳定性

`delta_degree_percentile` 对 IVOL 的年度平均 Risk IC：

- \(W60\)：12/12 年为正；
- \(W120\)：12/12 年为正；
- \(W252\)：11/11 年为正。

其 Q5−Q1 IVOL spread 也几乎全部年份保持正向。

---

## 6.4 其他主要风险结果

### `residual_degree_percentile`

未来风险显著负相关：

\[
ResidualDegree\uparrow
\Rightarrow
FutureRisk\downarrow.
\]

但 Raw Alpha 基本为 0。

因此更像：

\[
\boxed{
Risk/Stable Network Characteristic
}
\]

而不是主要 Alpha signal。

### `cross_industry_degree_percentile`

同样表现为：

\[
CrossIndustryDegree\uparrow
\Rightarrow
FutureRisk\downarrow.
\]

### Neighbor stability

`neighbor_jaccard_1m` 和 `neighbor_retention_1m` 都表现为：

\[
NeighborPersistence\uparrow
\Rightarrow
FutureRisk\downarrow.
\]

### `outside_community_degree_percentile`

风险关系非常强：

\[
IC_{TotalVol}
\approx
-0.17\sim-0.19,
\]

\[
IC_{IVOL}
\approx
-0.13\sim-0.17.
\]

---

## 6.5 Step 5 总体结构

36 个：

\[
9\ Factors
\times
4\ RiskMetrics
\]

组合中：

\[
\boxed{
36/36
}
\]

在 \(W=60,120,252\) 下平均 Risk IC 保持同号。

同时：

\[
\boxed{
delta\_degree\_percentile
}
\]

是唯一在所有风险标签和窗口下均为正 Risk IC 的因子。

其余多数因子表现为负向风险关系。

---

# 7. Day 6 Step 6：Neutralization & Initial Factor Classification

## 7.1 目标

控制网络因子的：

\[
Industry
\]

以及：

\[
Size
\]

暴露，判断 Step 3–5 的结果是否只是行业或大小盘效应。

构造三种版本：

\[
RAW,
\]

\[
INDUSTRY\_NEUTRAL,
\]

\[
INDUSTRY\_SIZE\_NEUTRAL.
\]

---

## 7.2 Industry Neutralization

定义：

\[
F^{IND}_{i,t}
=
F_{i,t}
-
\overline F_{g(i,t),t}.
\]

等价于行业固定效应回归残差。

---

## 7.3 Industry + Size Neutralization

模型：

\[
F_{i,t}
=
\alpha_{g(i,t)}
+
\beta_t\log(MV_{i,t})
+
\varepsilon_{i,t}.
\]

最终使用：

\[
\boxed{
F^{IND+SIZE}_{i,t}
=
\widehat\varepsilon_{i,t}.
}
\]

行业和市值全部来自 analysis date 的 PIT 信息。

---

## 7.4 Neutralization QA

数值结果非常稳定：

行业中性化后：

\[
\max
\left|
\overline{
F^{IND}
}_{g}
\right|
\approx
1.57\times10^{-16}.
\]

Industry + Size 后：

\[
\max
\left|
\overline{
F^{IND+SIZE}
}_{g}
\right|
\approx
7.05\times10^{-16}.
\]

残差与 log size 最大相关绝对值：

\[
4.34\times10^{-16}.
\]

说明中性化在数值精度范围内完成。

---

# 8. Step 6 最重要结果：`delta_degree_percentile` 仍然稳健

## 8.1 Alpha

Industry + Size neutral 后：

\[
W60:
-0.0400,
\]

\[
W120:
-0.0349,
\]

\[
W252:
-0.0323.
\]

HAC t-stat：

\[
-11.41,\quad
-9.33,\quad
-7.23.
\]

说明：

\[
\boxed{
\Delta Degree
\text{ 的负向收益关系不能由 Industry + Size 简单解释。}
}
\]

---

## 8.2 Portfolio Spread

中性化后：

\[
W60:
-78.8\text{ bp/month},
\]

\[
W120:
-70.5\text{ bp/month},
\]

\[
W252:
-68.2\text{ bp/month}.
\]

Raw 时平均约：

\[
-111.9\text{ bp/month}.
\]

Industry + Size 后平均约：

\[
-72.5\text{ bp/month}.
\]

仍保留约：

\[
65\%
\]

的经济幅度。

---

## 8.3 Risk

更重要的是，风险关系在中性化后反而增强。

Raw 四种风险平均 IC：

\[
+0.061.
\]

Industry neutral：

\[
+0.095.
\]

Industry + Size neutral：

\[
\boxed{
+0.122.
}
\]

其中 IVOL：

\[
0.1360,\quad
0.1467,\quad
0.1549.
\]

因此：

\[
\boxed{
\Delta Degree
\text{ 的高未来风险关系并不是 size/industry exposure 造成，}
}
\]

反而可能被原始 size exposure 部分掩盖。

---

# 9. Step 6 对其他因子的更新认识

## 9.1 `residual_degree_percentile`

Raw Alpha：

\[
\approx0.
\]

但 Industry + Size neutral 后：

\[
0.0111,\quad
0.0125,\quad
0.0146.
\]

同时 Risk IC 仍明显负向。

说明：

\[
\boxed{
ResidualDegree
\text{ 更像 risk-dominant favorable characteristic，}
}
\]

而其 Alpha 信息被规模暴露部分掩盖。

---

## 9.2 `cross_industry_degree_percentile`

Industry + Size neutral 后 Alpha IC：

\[
0.0109,\quad
0.0123,\quad
0.0136.
\]

同时未来风险持续下降。

可初步理解为：

\[
\boxed{
\text{Risk/diversification-like network characteristic}
}
\]

并伴随小幅正 Alpha。

---

## 9.3 Neighbor Stability

### `neighbor_jaccard_1m`

Industry + Size neutral Alpha IC：

\[
0.0101,\quad
0.0121,\quad
0.0159.
\]

Spread：

\[
+24.0,\quad
+21.0,\quad
+35.0
\text{ bp/month}.
\]

风险 IC 显著为负。

### `neighbor_retention_1m`

Alpha IC：

\[
0.0130,\quad
0.0139,\quad
0.0155.
\]

Spread：

\[
+31.6,\quad
+30.1,\quad
+28.6
\text{ bp/month}.
\]

表现为：

\[
\boxed{
\text{higher future return + lower future risk}.
}
\]

---

## 9.4 `outside_community_degree_percentile`

Industry + Size neutral 后：

\[
MeanAlphaIC
\approx
0.0191.
\]

风险关系仍然非常强：

\[
MeanRiskIC
\approx
-0.1172.
\]

其中：

\[
MeanIVOLIC
\approx
-0.1360.
\]

因此它当前更适合作为：

\[
\boxed{
Risk-dominant structural factor
}
\]

而不是仅根据 Raw Portfolio Sort 判断其 Alpha。

---

## 9.5 `cross_industry_degree_ratio`

中性化后：

\[
AlphaIC
\approx0,
\]

Risk IC 也很弱且方向不够稳定。

因此：

\[
\boxed{
cross\_industry\_degree\_ratio
}
\]

目前更适合作为：

\[
\boxed{
Structural / Auxiliary Characteristic
}
\]

而不是核心 Alpha/Risk 因子。

---

# 10. 当前 9 个网络因子的初步功能分类

基于 Step 3–6 的联合结果，今天形成如下初步分类：

| 因子 | 当前建议角色 |
|---|---|
| `delta_degree_percentile` | **核心 Mixed / Adverse Alpha-Risk Factor** |
| `neighbor_retention_1m` | **Favorable Alpha + Stability/Risk Factor** |
| `neighbor_jaccard_1m` | Stability/Risk + Moderate Alpha |
| `outside_community_degree_percentile` | **Risk-dominant Structural Factor** |
| `residual_degree_percentile` | Risk-dominant；Size-neutral 后 Alpha 出现 |
| `cross_industry_degree_percentile` | Risk / Diversification-like；Size-neutral 后 Alpha 增强 |
| `delta_cross_industry_degree_percentile_1m` | Weak Dynamic Alpha/Risk |
| `delta_residual_degree_percentile_1m` | Weak Dynamic Alpha/Risk |
| `cross_industry_degree_ratio` | **Structural / Unstable Auxiliary** |

这里的分类仅是：

\[
\boxed{
\text{Descriptive Initial Classification}
}
\]

不会修改：

- frozen factor set；
- factor sign；
- network window；
- 后续研究设计。

---

# 11. 今天最重要的研究发现

## 11.1 Relative Network Rewiring 比 Static Centrality 更重要

今天最清晰的结果不是：

\[
ResidualDegree
\]

本身，而是：

\[
\boxed{
\Delta Degree
=
Degree^{Residual}
-
Degree^{Raw}.
}
\]

它反映：

\[
\boxed{
\text{市场共同成分被剔除后，股票在网络中的相对中心性如何重新排序。}
}
\]

这一重排变量与未来收益和未来风险都具有非常稳定的关系。

---

## 11.2 核心模式

目前最重要的实证关系可以概括为：

\[
\boxed{
\Delta Degree\uparrow
\Rightarrow
\begin{cases}
FutureExcessReturn\downarrow,\\
FutureIVOL\uparrow,\\
FutureMDD\uparrow,\\
FutureDownsideVol\uparrow.
\end{cases}
}
\]

并且这一关系：

- 跨 \(W=60,120,252\) 稳定；
- 跨多数年份稳定；
- Portfolio Sort 与 Rank IC 一致；
- Industry neutral 后仍存在；
- Industry + Size neutral 后仍存在；
- 风险关系在中性化后反而增强。

因此：

\[
\boxed{
\text{该结构不能简单归因于行业或市值暴露。}
}
\]

---

## 11.3 Neighbor Stability 可能对应更稳定的未来状态

`neighbor_jaccard_1m` 与 `neighbor_retention_1m` 表现为：

\[
\boxed{
\text{邻居关系越稳定}
\Rightarrow
\text{未来风险越低}
}
\]

并在控制 Industry + Size 后出现更清晰的正 Alpha。

这为此前 Day 5 中“稳定行业核心 + 动态跨行业外围”的网络结构观察提供了一个新的 forward-validation 视角。

---

## 11.4 Size 是非常重要的混杂来源

很多网络因子的 Alpha 在：

\[
RAW
\rightarrow
INDUSTRY\_NEUTRAL
\]

变化不大，

但在：

\[
INDUSTRY\_NEUTRAL
\rightarrow
INDUSTRY\_SIZE\_NEUTRAL
\]

后明显增强或改变。

例如：

\[
ResidualDegree:
0.001
\rightarrow
0.0045
\rightarrow
0.0127.
\]

以及：

\[
OutsideCommunityDegree:
0.0107
\rightarrow
0.0113
\rightarrow
0.0191.
\]

说明：

\[
\boxed{
Size Exposure
}
\]

是后续研究中必须重点控制的变量。

---

# 12. 今日解决的主要技术问题

今天并不仅仅完成了 Step 1–6，还解决了多个会直接影响研究有效性的技术问题。

## 12.1 修正 Future Return Convention

识别出：

```text
return_network
```

并不适合作为 holding-period forward-return source。

最终改为：

```text
return_last_trade_simple
+
suspension return = 0
```

成功解决 2015 年 coverage 异常。

---

## 12.2 区分 Network Return 与 Holding Return

形成明确的两套定义：

### 网络估计

\[
\boxed{
return\_network
}
\]

服务于同步横截面网络估计。

### Forward validation

\[
\boxed{
holding\_return
}
\]

服务于真实持有路径收益和风险标签。

两种定义不再混用。

---

## 12.3 严格保持 PIT 和 no-future-leakage

整个 Day 6：

- 当前 factor universe 不由未来数据决定；
- neutralization 只使用当期行业和市值；
- portfolio breakpoint 只使用当期 factor；
- future return availability 不参与当期分组；
- 不根据未来 outcome 改变因子定义。

---

## 12.4 对 Ties 的识别

发现一些离散网络因子在 quintile sort 中存在大量 ties。

因此后续解释明确区分：

\[
\boxed{
RankIC
}
\]

与：

\[
\boxed{
PortfolioSort
}
\]

的有效样本覆盖差异。

对于高度离散因子，Rank IC 的结果应优先于当前 quintile spread。

---

# 13. 当前研究链条的完整性

截至今天，M1 的 Day 6 已形成完整的 forward-validation workflow：

### Step 1
冻结网络因子。

### Step 2
构造严格未来收益和风险标签。

### Step 3
检验横截面 Rank IC。

### Step 4
检验组合收益经济幅度。

### Step 5
检验未来风险关系。

### Step 6
行业/规模中性化，并进行初步功能分类。

因此：

\[
\boxed{
\text{Day 6 已基本完成网络因子第一次正式 forward validation 闭环。}
}
\]

---

# 14. 下一步建议

后续不建议继续机械增加网络指标，而应围绕当前核心发现做更严格验证。

## 14.1 Out-of-Sample / Subsample Validation

建议进一步分：

- early / late sample；
- bull / bear；
- high-vol / low-vol；
- crisis / normal periods。

检验：

\[
\Delta Degree
\]

是否在不同市场状态下仍保持：

\[
LowReturn + HighRisk.
\]

---

## 14.2 Traditional Characteristic Controls

Step 6 目前只控制：

\[
Industry + Size.
\]

下一阶段可以考虑：

- Momentum；
- Short-term reversal；
- Turnover / liquidity；
- Volatility；
- Beta；
- Book-to-market；
- Value / quality proxies。

从而判断：

\[
\Delta Degree
\]

是否仍包含独立信息。

---

## 14.3 Turnover & Transaction Cost

目前 Step 4 是 gross spread。

后续需要计算：

\[
PortfolioTurnover
\]

以及：

\[
NetReturn
=
GrossReturn
-
TransactionCost.
\]

特别是对于动态因子：

\[
delta\_*\_1m
\]

和 neighbor-dynamics 因子，这一步很重要。

---

## 14.4 更正式的 Asset-Pricing Test

后续可考虑：

\[
Fama\text{-}MacBeth
\]

或其他横截面回归框架，检验网络变量在传统 controls 后是否仍具有增量解释力。

---

# 15. 今日工作简要汇报版

今天完成了 M1 全市场股票关联网络研究的 Day 6 全部六个步骤。首先冻结了 9 个核心网络因子，随后构造严格的未来收益和风险标签。在 forward-label 构造中发现原 `return_network` 不适合持有期收益，因为停牌和复牌首日会被有意设为缺失；通过回查 Day 1 代码，最终采用 `return_last_trade_simple` 并将明确停牌日设为 0 收益，成功将三个窗口的平均 future-return coverage 提升到 99.99% 左右，并完整恢复全部 17,919 个复牌收益。

随后进行 Rank IC 和 Portfolio Sort。最核心的结果来自 `delta_degree_percentile`：其在 \(W=60,120,252\) 下未来收益 Rank IC 分别约为 -0.051、-0.046 和 -0.039，对应 Q5−Q1 月均 excess-return spread 约为 -125、-115 和 -96 bp，且跨年份保持较稳定的负向关系。

风险验证进一步发现，`delta_degree_percentile` 同时预测更高未来风险，特别是未来 IVOL，其 Risk IC 约为 0.123–0.142，Q5−Q1 IVOL 差异约为 5.7–6.0 个百分点。因此该变量呈现出“低未来收益 + 高未来风险”的 adverse return-risk pattern。

最后进行 Industry 和 Size neutralization。中性化通过严格数值 QA。`delta_degree_percentile` 在 Industry + Size neutral 后仍保留约 -0.032 至 -0.040 的 Alpha IC，以及约 -68 至 -79 bp/month 的组合 spread；同时其风险关系反而增强，四类风险平均 IC 从 Raw 的约 0.061 上升至约 0.122。结果说明，该核心关系不能由行业和市值暴露简单解释。

综合 Step 1–6，目前最重要的研究结论是：**Raw-to-Residual network reordering 所携带的信息明显不同于 static network centrality，并且可能是比单纯网络中心性更有价值的前瞻性股票特征。**

