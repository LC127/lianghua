# M1 A股全市场股票关联网络：Joint Incremental and Out-of-Sample Risk Forecasting Validation
**日期：2026-09-29**  
**主题：Joint Incremental and Out-of-Sample Risk Forecasting Validation**  
**中文主题：网络风险信息的联合增量性与严格样本外预测检验**

---

## 1. 今日研究目标

Day 11 的核心目标是把此前 Day 6–10 中已经得到的“网络特征与未来个股特质波动率（future idiosyncratic volatility）存在稳定关系”的证据，进一步推进到严格的样本外预测层面。

今天要回答的核心问题是：

\[
\boxed{
\text{冻结的股票网络特征能否在传统股票特征之外，}
}
\]

\[
\boxed{
\text{稳定提高 future idiosyncratic volatility 的严格 PIT 样本外预测？}
}
\]

研究过程中坚持以下原则：

- 不新增网络因子；
- 不根据结果重新选择窗口；
- 不根据 Step 2 的相关性或 VIF 删除因子；
- 不利用 OOS 结果重新调整主模型；
- 预测日期 \(t\) 的训练集只能使用当时已经完全实现的历史 forward labels；
- Step 4–6 不重新定义 target；
- PRIMARY 模型始终保留全部 9 个 frozen network factors。

Day 11 共完成 6 个步骤：

1. **Frozen Prediction Panel & PIT Audit**
2. **Frozen Network-Factor Redundancy Diagnostic**
3. **Strict Expanding-Window OOS Risk Forecasting**
4. **Incremental OOS Risk Metrics**
5. **OOS Temporal and Block-Incremental Validation**
6. **Day 11 Joint OOS Validation Summary**

最终状态：

\[
\boxed{\textbf{Day 11 = PASS + FROZEN}}
\]

---

# 2. Step 1 — Frozen Prediction Panel & PIT Audit

## 2.1 研究目的

Step 1 的目标是构建 Day 11 后续所有 OOS 分析唯一允许使用的预测面板，并严格检查 point-in-time（PIT）约束，避免未来信息泄露。

最终 prediction panel 的逻辑结构为：

\[
\boxed{
(i,t,W,C_{i,t},F_{i,t,W},Y^{future}_{i,t})
}
\]

其中：

- \(i\)：股票；
- \(t\)：调仓/分析日期；
- \(W\in\{60,120,252\}\)：网络估计窗口；
- \(C_{i,t}\)：传统股票特征；
- \(F_{i,t,W}\)：9 个冻结网络特征；
- \(Y^{future}_{i,t}\)：future idiosyncratic volatility。

---

## 2.2 数据源冻结

### 网络特征源

使用 Day 5 冻结的网络特征面板：

```text
D:\M1_StockNetwork\output\M1_day5\
05_step5_community_features\
community_bridge_feature_panel.parquet
```

### Future-IVOL 标签源

使用 Day 6 当前正式的 consolidated forward-label panel：

```text
D:\M1_StockNetwork\output\M1_day6\
02_step2_forward_labels\
forward_label_panel.parquet
```

排除了：

- `archive_before_holding_return_v2_*`
- `panel_parts\...`

避免混入旧标签版本或重新拼接分片。

### 传统控制变量源

使用 Day 7：

```text
D:\M1_StockNetwork\output\M1_day7\
02_stage2_traditional_characteristic_controls\
traditional_characteristic_panel.parquet
```

传统特征固定为：

```text
industry_id1
log_market_value
momentum_120_20
reversal_20
volatility_60
log_turnover_20
```

其中 `industry_id1` 在预测模型中作为行业固定效应处理，其余 5 个变量作为连续传统特征。

---

## 2.3 PIT 样本构造原则

预测面板始终以 Day 5 frozen factor universe 为左表：

\[
\boxed{
\mathcal U_t
=
\text{当前时点可获得的 frozen factor universe}
}
\]

而不是：

\[
\{i:Y^{future}_{i,t}\text{ 可观测}\}.
\]

因此 future label 是否缺失不会决定时点 \(t\) 的股票 universe。

对于预测日期 \(T\)，历史 observation \(s\) 只有满足：

\[
analysis\_date_s<T,
\]

并且：

\[
\boxed{
label\_end_s\le T
}
\]

时才允许进入训练样本。

这一规则比简单使用：

\[
analysis\_date_s<T
\]

更加严格，因为 forward-IVOL 必须已经完全实现。

---

## 2.4 Step 1 QA

最终 prediction panel：

\[
\boxed{1,549,521}
\]

条 stock-date-window observations。

关键 QA：

```text
factor_universe_preserved = True
duplicate_stock_date_window_count = 0

frozen_factor_count = 9
window_count = 3

target_used_to_define_current_universe = False

label_end_date_mismatch_count = 0
target_with_missing_label_end_count = 0
nonforward_target_interval_count = 0

training_calendar_leakage_count = 0

network_features_reestimated = False
future_ivol_reestimated = False
future_information_used_as_predictor = False
```

Future-IVOL 可用 observations：

\[
1,518,496
\]

缺失：

\[
31,025.
\]

其中绝大多数缺失来自样本末端尚未实现的未来标签，因此不构成 PIT 问题。

### Step 1 结论

\[
\boxed{\textbf{Step 1 = PASS + FROZEN}}
\]

---

# 3. Step 2 — Frozen Network-Factor Redundancy Diagnostic

## 3.1 研究目的

Day 10 已发现多个网络变量都对未来 IVOL 具有稳定关系，因此 Step 2 进一步回答：

\[
\boxed{
\text{9 个 frozen network factors 是否实际上高度重复？}
}
\]

本步骤不进行变量筛选，仅进行描述性冗余诊断。

主要指标：

1. 当期横截面 Spearman correlation；
2. Variance Inflation Factor（VIF）；
3. correlation-matrix condition number；
4. PCA variance share；
5. participation-ratio effective rank。

---

## 3.2 九个冻结网络因子

```text
1. residual_degree_percentile
2. delta_degree_percentile
3. cross_industry_degree_percentile
4. cross_industry_degree_ratio
5. delta_residual_degree_percentile_1m
6. delta_cross_industry_degree_percentile_1m
7. neighbor_jaccard_1m
8. neighbor_retention_1m
9. outside_community_degree_percentile
```

QA：

- 总 date-window cells：402；
- 有效 cells：399；
- common valid dates：128；
- 每个窗口均完整得到 36 个 factor pairs；
- future target 未参与；
- 没有因子删除、合并或结果驱动选择。

---

## 3.3 总体冗余结构

| Window | Native Effective Rank | Common-Date Effective Rank | PC1 Share | PC1–PC3 Cumulative | PCs for 80% | PCs for 90% |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 3.751 | 3.781 | 43.7% | 76.8% | 4 | 5 |
| 120 | 4.030 | 4.027 | 40.9% | 75.8% | 4 | 5 |
| 252 | 4.429 | 4.429 | 36.4% | 73.5% | 4 | 5 |

因此：

\[
\boxed{
9\text{ 个网络变量具有明显但不完全的冗余}
}
\]

更准确地说：

\[
\boxed{
\text{participation effective rank}\approx3.8-4.4
}
\]

而 80% 与 90% 的横截面 variation 大约需要 4 个和 5 个主成分解释。

所以该网络因子系统是：

\[
\boxed{
\text{moderately low-dimensional, but not one-dimensional}
}
\]

---

## 3.4 典型高冗余关系

### Degree-level

```text
residual_degree_percentile
cross_industry_degree_percentile
```

median Spearman 约为：

\[
0.917,\ 0.901,\ 0.873
\]

对应 W60/W120/W252。

### Short-run degree change

```text
delta_residual_degree_percentile_1m
delta_cross_industry_degree_percentile_1m
```

median Spearman 约为：

\[
0.923,\ 0.888,\ 0.831.
\]

### Neighbor stability

```text
neighbor_jaccard_1m
neighbor_retention_1m
```

median Spearman 约为：

\[
0.901,\ 0.825,\ 0.724.
\]

---

## 3.5 Delta degree 的独立性

`delta_degree_percentile` 的 median VIF 仅约：

\[
1.44,\quad1.40,\quad1.30.
\]

说明该变量并不是其他 8 个 frozen factors 的简单线性复制。

相反，部分 degree-level variables 的 median VIF 超过 10，说明它们存在较明显联合冗余。

### Step 2 结论

\[
\boxed{\textbf{Step 2 = PASS + FROZEN}}
\]

并明确：

\[
\boxed{
\text{冗余诊断只用于解释联合模型，不用于删因子。}
}
\]

---

# 4. Step 3 — Strict Expanding-Window OOS Risk Forecasting

## 4.1 模型设计

### Traditional Baseline

\[
Y^{IVOL}_{i,t+1}
=
\alpha
+
IndustryFE_{i,t}
+
\gamma^\top C_{i,t}
+
\varepsilon_{i,t+1}.
\]

### Network-Augmented Model

\[
Y^{IVOL}_{i,t+1}
=
\alpha
+
IndustryFE_{i,t}
+
\gamma^\top C_{i,t}
+
\beta^\top F^{Network}_{i,t,W}
+
\varepsilon_{i,t+1}.
\]

其中：

\[
F^{Network}
\]

一次性包含全部 9 个 frozen factors。

---

## 4.2 OOS 规则

对每个 forecast date \(t\)，只有历史样本满足：

\[
analysis\_date_s<t
\]

且：

\[
label\_end_s\le t
\]

时才进入 expanding training sample。

同时要求至少：

\[
\boxed{24}
\]

个已经实现的历史预测期。

训练使用 date-balanced weighting，避免上市公司数量随时间增长导致后期月份在 pooled regression 中被机械赋予更高权重。

---

## 4.3 OOS 预测规模

共生成：

\[
\boxed{1,286,838}
\]

条严格 OOS stock-level predictions。

预测日期数：

| Window | Forecast dates | First forecast |
|---:|---:|---|
| 60 | 111 | 2017-07-31 |
| 120 | 110 | 2017-08-31 |
| 252 | 104 | 2018-02-28 |

Step 3 只生成：

```text
baseline_score
network_score
```

并不评价预测效果，从而避免根据 OOS 表现修改模型。

---

## 4.4 数值稳定性说明

早期 expanding period 的部分 design matrices 存在 rank deficiency，因此主模型使用 Moore–Penrose pseudoinverse 生成 minimum-norm OLS solution。

这并未造成：

- missing prediction；
- duplicate prediction；
- PIT leakage。

后续 Step 4 专门使用 FULL_RANK 子样本检查该问题是否驱动主结果。

### Step 3 结论

\[
\boxed{\textbf{Step 3 = PASS + FROZEN}}
\]

---

# 5. Step 4 — Incremental OOS Risk Metrics

## 5.1 核心评价指标

核心指标定义为：

\[
IC^{OOS}_{B,t}
=
Corr_{rank}
(
\widehat Y^{B}_{i,t},
Y^{future}_{i,t}
),
\]

\[
IC^{OOS}_{N,t}
=
Corr_{rank}
(
\widehat Y^{N}_{i,t},
Y^{future}_{i,t}
),
\]

以及：

\[
\boxed{
\Delta IC_t^{OOS}
=
IC^{OOS}_{N,t}
-
IC^{OOS}_{B,t}.
}
\]

另外报告：

- rank-scale MSE；
- rank-scale MAE；
- relative OOS rank-\(R^2\)；
- predicted top-vs-bottom realized-risk spread。

两个评价口径：

- **PRIMARY**：所有 future-IVOL 已实现的 OOS forecasts；
- **FULL_RANK**：baseline 与 network 两个 design matrices 均满秩的数值稳健性子样本。

---

## 5.2 PRIMARY 结果

| W | Baseline IC | Network IC | \(\Delta IC\) | Positive Months | Positive Years | Relative OOS \(R^2\) |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.4902 | 0.5039 | **+0.0137** | **84.4%** | **90.0%** | **+2.7%** |
| 120 | 0.4916 | 0.5036 | **+0.0119** | **77.8%** | **100.0%** | **+2.3%** |
| 252 | 0.4923 | 0.5054 | **+0.0130** | **82.4%** | **100.0%** | **+2.6%** |

三个窗口均满足：

\[
\boxed{
IC^{OOS}_{Network}>IC^{OOS}_{Baseline}.
}
\]

平均增量约：

\[
\boxed{
0.012-0.014.
}
\]

---

## 5.3 FULL_RANK Robustness

| W | \(\Delta IC\) | Positive Months | Positive Years | Relative OOS \(R^2\) |
|---:|---:|---:|---:|---:|
| 60 | **+0.0100** | 83.5% | 85.7% | +2.0% |
| 120 | **+0.0078** | 74.7% | 100.0% | +1.6% |
| 252 | **+0.0104** | 79.7% | 100.0% | +2.1% |

因此：

\[
\boxed{
\text{主结果并非由早期 rank-deficient OLS 数值问题驱动。}
}
\]

需要注意，FULL_RANK 同时改变了 calendar-period composition，因此它应解释为 numerical robustness，而不是对 rank deficiency 的因果净化。

---

## 5.4 其他指标方向一致

PRIMARY 下，Network model 相对 Baseline：

- rank-MSE 更低；
- rank-MAE 更低；
- relative OOS \(R^2>0\)；
- top-minus-bottom realized-risk spread 更大。

因此：

\[
\boxed{
\text{OOS improvement 并不是单一 IC 指标的特殊现象。}
}
\]

### Step 4 结论

\[
\boxed{\textbf{Step 4 = PASS + FROZEN}}
\]

核心研究升级：

\[
\boxed{
\text{network-risk association}
\rightarrow
\text{strict OOS incremental predictive value}.
}
\]

---

# 6. Step 5 — OOS Temporal and Block-Incremental Validation

## 6.1 Temporal Stability

PRIMARY：

| W | Years | Mean Annual \(\Delta IC\) | Positive-Year Share | Worst Year |
|---:|---:|---:|---:|---|
| 60 | 10 | 0.0132 | **90.0%** | 2021 |
| 120 | 10 | 0.0118 | **100.0%** | 2021 |
| 252 | 9 | 0.0129 | **100.0%** | 2021 |

FULL_RANK：

| W | Years | Mean Annual \(\Delta IC\) | Positive-Year Share |
|---:|---:|---:|---:|
| 60 | 7 | 0.0097 | 85.7% |
| 120 | 7 | 0.0077 | 100.0% |
| 252 | 7 | 0.0103 | 100.0% |

三个窗口共同最弱年份均为：

\[
\boxed{2021}.
\]

但仅 W60 略为负：

\[
-0.0003.
\]

W120、W252 仍为正。

因此更准确的结论是：

> 2021 年 network incremental predictive value 明显减弱，但没有在多个 horizon 上系统性反转。

---

## 6.2 预先冻结的四个信息块

### LEVEL_POSITION

```text
residual_degree_percentile
cross_industry_degree_percentile
outside_community_degree_percentile
```

### CROSS_INDUSTRY_COMPOSITION

```text
cross_industry_degree_ratio
```

### RECONFIGURATION

```text
delta_degree_percentile
delta_residual_degree_percentile_1m
delta_cross_industry_degree_percentile_1m
```

### NEIGHBOR_STABILITY

```text
neighbor_jaccard_1m
neighbor_retention_1m
```

这些 block 根据变量构造含义预先定义，不使用 Step 2 VIF 或 Step 4 OOS performance 进行数据驱动分组。

---

## 6.3 LOBO 定义

采用 Leave-One-Block-Out：

\[
Contribution_g
=
IC_{AllNetwork}
-
IC_{Without\ block\ g}.
\]

如果：

\[
Contribution_g>0,
\]

表示在其他 network blocks 均已存在的条件下，该 block 仍提供条件 OOS 增量。

注意：

\[
\boxed{
\sum_g Contribution_g
\neq
IC_{AllNetwork}-IC_{Baseline}
}
\]

因此不能把 marginal contribution 当作 additive decomposition 或贡献百分比。

---

## 6.4 PRIMARY Block 结果

### W60

| Block | Marginal IC | Positive Months | Positive Years |
|---|---:|---:|---:|
| LEVEL_POSITION | **+0.0057** | 75.2% | 90.0% |
| RECONFIGURATION | **+0.0050** | 66.1% | 70.0% |
| NEIGHBOR_STABILITY | **+0.0029** | 78.9% | 100.0% |
| CROSS_INDUSTRY_COMPOSITION | ~0 | 54.1% | 70.0% |

### W120

| Block | Marginal IC | Positive Months | Positive Years |
|---|---:|---:|---:|
| RECONFIGURATION | **+0.0072** | 69.4% | 80.0% |
| LEVEL_POSITION | **+0.0057** | 75.0% | 80.0% |
| NEIGHBOR_STABILITY | **+0.0011** | 69.4% | 100.0% |
| CROSS_INDUSTRY_COMPOSITION | ~0 | 50.9% | 60.0% |

### W252

| Block | Marginal IC | Positive Months | Positive Years |
|---|---:|---:|---:|
| RECONFIGURATION | **+0.0109** | 81.4% | 100.0% |
| LEVEL_POSITION | **+0.0048** | 75.5% | 66.7% |
| NEIGHBOR_STABILITY | ~0 | 52.9% | 44.4% |
| CROSS_INDUSTRY_COMPOSITION | ~0 | 51.0% | 66.7% |

---

## 6.5 Block-Level 解释

### Network Level / Position

跨三个窗口持续提供正向条件贡献：

\[
0.0057,\quad0.0057,\quad0.0048.
\]

说明静态网络位置是相对稳定的 OOS 风险信息维度。

### Network Reconfiguration

条件增量随网络窗口增大：

\[
0.0050
\rightarrow
0.0072
\rightarrow
0.0109.
\]

说明在较长网络窗口下，近期网络结构变化相对于长期结构的位置尤其具有风险预测价值。

### Neighbor Stability

主要贡献集中于 W60/W120：

\[
0.0029,\quad0.0011,\quad\approx0.
\]

因此更像短中 horizon 信息。

### Cross-Industry Composition

三个窗口 marginal IC 均接近：

\[
0.
\]

说明在其他网络信息已经存在时，`cross_industry_degree_ratio` 的额外 future-IVOL OOS prediction contribution 很有限。

该结果不能解释为该变量“没有信息”，而应解释为：

\[
\boxed{
\text{conditional incremental predictive value is limited}.
}
\]

---

## 6.6 Step 5 QA

最关键的一项：

\[
\max |
Step5\ reproduced\ Step4
-
Step4\ original
|
=
9.95\times10^{-17}.
\]

同时：

```text
forecast_sample_mismatch_count = 0
training_period_mismatch_count = 0
training_row_mismatch_count = 0
```

说明 Step 5 完整复现了 Step 4 的 primary evaluation structure。

### Step 5 结论

\[
\boxed{\textbf{Step 5 = PASS + FROZEN}}
\]

---

# 7. Step 6 — Day 11 Joint OOS Validation Summary

## 7.1 研究目的

Step 6 不再进行任何新估计，只把 Step 2、Step 4、Step 5 冻结结果整合为统一 evidence chain：

\[
\boxed{
Redundancy
\rightarrow
Strict\ OOS
\rightarrow
Incremental\ Gain
\rightarrow
Temporal\ Stability
\rightarrow
Structural\ Source
}
\]

---

## 7.2 跨步骤一致性 QA

最终：

```text
step1_formal_qa_pass = True
step2_formal_qa_pass = True
step3_formal_qa_pass = True
step4_formal_qa_pass = True
step5_formal_qa_pass = True

joint_summary_row_count = 6
expected_joint_summary_row_count = 6

block_summary_row_count = 24
expected_block_summary_row_count = 24

joint_duplicate_scope_window_count = 0
block_duplicate_scope_window_block_count = 0

step4_step5_positive_year_share_max_abs_error = 0.0
step4_step5_total_delta_ic_max_abs_error = 0.0

new_model_fitted = False
new_factor_constructed = False
factor_selected_from_results = False
factor_removed_from_primary_model = False
window_selected_from_results = False
hyperparameter_tuned = False
future_target_used_for_model_selection = False

step6_is_summary_only = True
all_formal_qa_pass = True
```

因此：

\[
\boxed{\textbf{Step 6 = PASS + FROZEN}}
\]

---

# 8. Day 11 最终证据链

今天最终形成的完整研究逻辑为：

\[
\boxed{
\text{Frozen PIT Network Features}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{9 factors contain substantial but incomplete redundancy}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{roughly 4 effective cross-sectional dimensions}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Strict expanding-window PIT OOS forecasting}
}
\]

\[
\Downarrow
\]

\[
\boxed{
IC^{OOS}_{Network}
>
IC^{OOS}_{Traditional}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\Delta IC^{OOS}
\approx0.012-0.014
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{78\%-84\% of forecast months show positive incremental IC}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{90\%-100\% of PRIMARY calendar years are positive}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{FULL-RANK numerical robustness remains positive}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{incremental information is concentrated mainly in}
}
\]

\[
\boxed{
\text{Network Level/Position + Network Reconfiguration}.
}
\]

---

# 9. 今日主要研究结论

今天最重要的研究升级是：

> **此前 Day 7–10 得到的 network–future-IVOL 关系，不仅是控制传统股票特征后的统计关联，也不仅具有跨年份稳定性；在严格 point-in-time expanding-window 的样本外预测框架下，冻结的网络特征能够稳定提高未来个股特质波动率的预测表现。**

具体而言：

1. **网络变量不是 9 个完全独立维度。**  
   participation effective rank 约为 3.8–4.4，但 PC1 只能解释约 36%–44%，因此网络信息具有低维结构，同时明显不是单一共同因子。

2. **全部 9 个 frozen network factors 提供稳定的 OOS 增量。**  
   PRIMARY 三个窗口平均 \(\Delta IC\) 均约为 0.012–0.014。

3. **增量不是单一指标的结果。**  
   Network model 同时表现出更低的 rank-MSE、rank-MAE，正的 relative OOS rank-\(R^2\)，以及更高的 top-bottom realized-risk separation。

4. **结果具有明显 temporal robustness。**  
   PRIMARY W120 和 W252 的正增量年份比例达到 100%，W60 为 90%。

5. **FULL_RANK robustness 仍然成立。**  
   因此 Step 3 早期 design-matrix rank deficiency 不是主结果的唯一来源。

6. **OOS 增量具有明确结构来源。**  
   Level/Position 与 Reconfiguration 是最主要的条件预测信息来源；Neighbor Stability 更偏短中 horizon；Cross-Industry Composition 在其他网络变量已存在时的额外贡献接近零。

---

# 10. 与 Day 6–10 的衔接

目前 M1 关于 future idiosyncratic risk 的证据链已经形成：

### Day 6–7

\[
\boxed{
\text{Network characteristics}
\rightarrow
\text{future-IVOL controlled association}
}
\]

### Day 9

\[
\boxed{
\text{Risk relation is not confined to one stress regime}
}
\]

### Day 10

\[
\boxed{
\text{Network-risk relation is temporally persistent}
}
\]

### Day 11

\[
\boxed{
\text{Network information improves strict OOS risk forecasting}
}
\]

因此研究主线已经从：

\[
\text{association}
\]

推进到：

\[
\boxed{
\text{incremental out-of-sample predictive information}.
}
\]

---

# 11. 解释边界与研究纪律

后续报告中需要保持以下表述纪律。

## 11.1 不把 LOBO 当作贡献百分比分解

例如 W252 的 Reconfiguration marginal IC 很大，但不能写成：

> Reconfiguration explains XX% of the total network effect.

因为：

\[
\sum_g Contribution_g
\neq
\Delta IC_{All}.
\]

LOBO 是 conditional contribution，不是 Shapley decomposition。

## 11.2 不根据 Step 5 删除因子

尽管 `CROSS_INDUSTRY_COMPOSITION` 的条件增量接近 0，PRIMARY 模型仍然保持全部 9 个 frozen factors。

否则会形成：

\[
\boxed{
\text{post-OOS model selection}.
}
\]

## 11.3 不把 FULL_RANK 与 PRIMARY 的差异解释为 rank deficiency 的因果效应

FULL_RANK 样本从较晚年份开始，同时改变了 calendar composition，因此只能解释为 numerical robustness。

## 11.4 不把统计预测关系直接解释为因果机制

目前结果支持：

\[
\boxed{
\text{predictive information}
}
\]

而不是：

\[
\boxed{
\text{causal transmission mechanism}.
}
\]

---

# 12. Day 11 最终状态

Step 1–6 均完成并通过相应 QA：

\[
\boxed{
\textbf{Day 11 = PASS + FROZEN}
}
\]

今天不再增加新的 Step 7、因子组合或预测模型。

下一阶段更合理的工作是：

\[
\boxed{
\textbf{Day 6--11 Risk Evidence Chain Integration}
}
\]

即将：

- controlled association；
- state dependence；
- temporal persistence；
- strict OOS improvement；
- structural block contribution；

整合成 M1 的完整风险研究主线，并进入阶段性汇报/论文式结果整理。

---

# 13. 一句话总结

> **Day 11 在严格 PIT expanding-window 样本外框架下验证了 A 股全市场股票关联网络的增量风险预测价值：9 个冻结网络特征在传统股票特征之外能够稳定改善未来个股特质波动率预测，该改善跨窗口、跨年份且在 full-rank 样本中保持，并主要与网络位置和网络重构信息有关。**
