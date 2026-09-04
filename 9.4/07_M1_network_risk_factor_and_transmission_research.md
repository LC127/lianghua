# M1 Stage 7：网络风险因子、风险传导与尾部风险研究

## 1. Stage 7 的定位

Stage 7 的目标是：

\[
oxed{
	ext{把 Stage 5 已构建完成的 A 股全市场股票网络转化为可解释的风险指标，研究网络结构是否能够刻画或预警未来市场风险、个股风险及风险传导。}
}
\]

Stage 7 是 M1 中正式回答：

\[
oxed{
	ext{“股票网络是否包含传统风险模型之外的风险信息？”}
}
\]

的阶段。

整体流程为：

\[
	ext{Stage 1：股票池定义}
ightarrow
	ext{Stage 2：数据质量验收}
ightarrow
	ext{Stage 3：市场结构诊断}
ightarrow
	ext{Stage 4：网络选型}
ightarrow
	ext{Stage 5：全市场正式构网}
ightarrow
	ext{Stage 6：Alpha 因子研究}
ightarrow
oxed{	ext{Stage 7：Risk 因子与风险传导研究}}
\]

Stage 7 同时承担 M1 与 M4 的接口作用：

\[
oxed{
	ext{M1：网络风险结构}
\longrightarrow
	ext{M4：尾部风险预警与应对}
}
\]

---

## 2. Stage 7 的核心输入

至少需要：

- `main_network_edges.parquet`
- `benchmark_network_edges.parquet`
- `network_node_features.parquet`
- `network_level_summary.csv`
- `network_universe_by_date.parquet`

以及：

- 日收益；
- 指数收益；
- 波动率；
- 成交额、换手率、市值；
- Barra 风格暴露；
- 行业；
- ST / 停牌状态；
- 交易日历；
- 若有：Lead-Lag / Tail auxiliary network。

所有风险指标必须严格使用时点 \(t\) 已经可观测的信息构造，未来风险标签只用于之后的检验。

---

## 3. Stage 7 的研究对象分三层

### 3.1 市场级 Network Risk

研究整个 A 股网络当前是否更加拥挤、集中、脆弱或异常。

候选指标：

- Network Density；
- Mean Absolute Edge Weight；
- Cross-industry Connectivity；
- Centralization；
- Community Concentration；
- Transition Intensity；
- Structural Novelty。

### 3.2 个股级 Network Risk

研究哪些股票在网络中承担更高的传染、桥接或集中风险。

候选指标：

- Degree / Strength；
- Betweenness；
- Cross-industry Degree；
- Bridge Score；
- Node Driver Score；
- Neighbor Distress Exposure；
- Tail Neighbor Exposure。

### 3.3 传导级 Network Risk

研究风险通过哪些边、行业和节点传播。

包括：

- 同行业传播；
- 跨行业传播；
- 核心节点向外围传播；
- Lead-Lag spillover；
- Tail co-movement；
- Regime boundary rewiring。

---

## 4. Step 7.1：市场级 Network Risk Factor Library

### 4.1 Network Density

\[
Density_t=rac{|E_t|}{p_t(p_t-1)/2}.
\]

如果主网络采用固定密度，则 Density 本身不再具有风险信息，应改用 MeanAbsWeight、Centralization、CrossStrength 等指标。

### 4.2 Mean Absolute Weight

\[
MeanAbsWeight_t=rac{1}{|E_t|}\sum_{(i,j)\in E_t}|w_{ij,t}|.
\]

若网络密度固定，该指标通常比 Density 更有信息。

### 4.3 Same / Cross Industry Connectivity

\[
SameEdgeRatio_t=rac{|E_t^{same}|}{|E_t|},
\qquad
CrossEdgeRatio_t=rac{|E_t^{cross}|}{|E_t|}.
\]

强度版本：

\[
CrossStrengthShare_t=
rac{\sum_{(i,j)\in E_t^{cross}}|w_{ij,t}|}
{\sum_{(i,j)\in E_t}|w_{ij,t}|}.
\]

### 4.4 Network Centralization / Concentration

令：

\[
s_{i,t}=
rac{Strength_{i,t}}
{\sum_j Strength_{j,t}},
\]

定义：

\[
HHI_t=\sum_i s_{i,t}^2.
\]

并报告：

- Top-5 node share；
- Top-10 node share；
- entropy。

### 4.5 Community Concentration

如 Stage 5 已有 community label，可计算：

- community count；
- largest-community share；
- modularity；
- inter-community edge share。

---

## 5. Step 7.2：动态网络风险指标

### 5.1 Transition Intensity

设网络权重矩阵为 \(R_t\)，定义：

\[
TI_t=\|R_t-R_{t-1}\|_F.
\]

可分解为：

\[
TI_t^2=TI_{same,t}^2+TI_{cross,t}^2.
\]

也可分解为：

\[
TI_t^2=TI_{support,t}^2+TI_{persistent,t}^2.
\]

分别表示边支持集变化和存续边权重重定价。

### 5.2 Edge Turnover / Rewiring

\[
GrossChange_t=Lost_t+Gained_t,
\]

\[
NetChange_t=Gained_t-Lost_t,
\]

\[
Rewiring_t=GrossChange_t-|NetChange_t|.
\]

相邻网络 Jaccard：

\[
Jaccard_t=
rac{|E_t\cap E_{t-1}|}
{|E_t\cup E_{t-1}|}.
\]

### 5.3 Structural Novelty

历史质心：

\[
ar R_{t-1}
=
rac{1}{t-1}\sum_{s<t}R_s.
\]

定义：

\[
Novelty_t^{centroid}
=
\|R_t-ar R_{t-1}\|_F.
\]

最近历史状态距离：

\[
Novelty_t^{nearest}
=
\min_{s<t}\|R_t-R_s\|_F.
\]

支持集 novelty：

\[
Novelty_t^{support}
=
1-\max_{s<t}J(E_t,E_s).
\]

Transition 回答“和上一期相比变了多少”，Novelty 回答“和整个历史相比有多罕见”。

---

## 6. Step 7.3：个股级 Network Risk

### 6.1 Node Strength Risk

\[
Strength_{i,t}
=
\sum_j|w_{ij,t}|.
\]

### 6.2 Cross-industry Bridge Risk

\[
CrossStrength_{i,t}
=
\sum_{j:Industry_j
eq Industry_i}|w_{ij,t}|,
\]

\[
BridgeShare_{i,t}
=
rac{CrossStrength_{i,t}}{Strength_{i,t}}.
\]

### 6.3 Betweenness / Brokerage Risk

如计算可扩展，可使用 Betweenness。对于 5000+ 节点应优先采用稀疏图和近似算法，而不是每日精确全量计算。

### 6.4 Neighbor Distress Exposure

\[
NDE_{i,t}
=
rac{
\sum_j |w_{ij,t}|RiskSignal_{j,t}
}{
\sum_j|w_{ij,t}|
}.
\]

RiskSignal 可采用：

- realized volatility；
- drawdown；
- negative return；
- tail indicator；
- ST/distress status。

### 6.5 Node Transition Driver

\[
D_{i,t}
=
\sum_{j
eq i}(\Delta w_{ij,t})^2.
\]

节点贡献份额：

\[
DriverShare_{i,t}
=
rac{D_{i,t}}
{2\sum_{a<b}(\Delta w_{ab,t})^2}.
\]

用于识别“谁在推动网络结构变化”。

---

## 7. Step 7.4：定义未来风险标签

Stage 7 的关键是：

\[
NetworkRisk_t
ightarrow
FutureRisk_{t+h}.
\]

### 7.1 Future Realized Volatility

\[
RV_{t,h}
=
\sqrt{
\sum_{s=t+1}^{t+h}r_{m,s}^2
}.
\]

可基于：

- 全市场等权；
- 全市场市值加权；
- CSI300；
- CSI500；
- CSI1000。

### 7.2 Future Drawdown

定义未来 \(h\) 日最大回撤：

\[
MDD_{t,h}.
\]

### 7.3 Future Tail Loss

例如：

\[
TailLoss_{t,h}
=
-\min_{1\le k\le h}R_{t,t+k}.
\]

或定义未来是否进入极端下尾。

### 7.4 Future Cross-sectional Dispersion

用于衡量未来横截面风险分化。

### 7.5 Individual Stock Future Risk

个股层面可使用：

- future volatility；
- future drawdown；
- future crash indicator；
- downside semivariance；
- idiosyncratic volatility。

---

## 8. Step 7.5：市场级前瞻检验

对市场级指标 \(Z_t\)，先做：

\[
FutureRisk_{t+h}
=
lpha+eta Z_t+u_{t+h}.
\]

再加入传统控制变量：

\[
FutureRisk_{t+h}
=
lpha+eta Z_t+\gamma'X_t+u_{t+h}.
\]

控制变量可包括：

- 当前波动率；
- 市场收益；
- 换手率；
- 流动性；
- aggregate Barra risk；
- valuation。

研究 \(eta\) 在加入传统风险控制后是否仍有稳定信息。

---

## 9. Step 7.6：个股横截面风险检验

对个股 Network Risk：

\[
FutureVol_{i,t+h}
=
lpha_t
+
eta_t NRisk_{i,t}
+
\gamma_t'X_{i,t}
+
u_{i,t+h}.
\]

可使用：

- Fama-MacBeth；
- panel regression；
- grouped portfolio analysis。

控制：

- Size；
- Beta；
- Volatility；
- Liquidity；
- Industry；
- Barra exposures。

---

## 10. Step 7.7：Risk Sorting

按 Network Risk 将股票分为：

\[
Q1,\ldots,Q5.
\]

比较未来：

- volatility；
- drawdown；
- tail loss；
- crash probability。

这不是 Alpha 回测，而是风险排序能力评价。

---

## 11. Step 7.8：Tail Risk 与 M4 联动

若保留 Tail Network，可构造：

\[
TailCentrality_{i,t},
\quad
TailDensity_t,
\quad
TailCrossIndustryRatio_t.
\]

研究：

\[
TailNetwork_t
ightarrow
FutureCrashRisk.
\]

可形成早期预警指标库，但具体减仓、对冲、风险预算等操作规则应留给 M4。

---

## 12. Step 7.9：风险传导路径分析

### 12.1 Industry-pair Decomposition

\[
TI_g=
\sqrt{
\sum_{(i,j)\in g}
(\Delta w_{ij})^2
}.
\]

识别主要行业对。

### 12.2 Node Driver Decomposition

\[
D_i=\sum_j(\Delta w_{ij})^2.
\]

识别核心结构驱动股票。

### 12.3 Support vs Persistent

区分：

- Lost；
- Gained；
- Persistent-strengthening；
- Persistent-weakening。

回答风险传播主要来自“新边形成/旧边消失”，还是“既有关系迅速增强”。

---

## 13. Step 7.10：Regime Analysis

只有当网络风险指标呈现明显阶段性时再进行 Regime 分析。

可使用：

- change-point detection；
- dynamic programming segmentation；
- HMM（扩展）；
- clustering of network states。

重要原则：

滚动窗口高度重叠时，不能把每个 rolling state 当成独立样本做普通显著性检验。

---

## 14. Step 7.11：Main Network vs Benchmark

比较：

\[
RiskFactor_t^{main}
\]

和：

\[
RiskFactor_t^{benchmark}.
\]

可比较：

- Future volatility \(R^2\)；
- rank correlation；
- tail-event AUC；
- OOS prediction error；
- event detection ability。

核心问题：

> 主网络是否比简单 Raw Correlation 网络提供新增风险信息？

---

## 15. Step 7.12：与 Barra 风险模型比较

市场级可做：

\[
FutureRisk_t
=
lpha
+
eta NetworkRisk_t
+
\gamma BarraRisk_t
+
u_t.
\]

个股级可做：

\[
FutureVol_{i,t+h}
=
lpha_t
+
eta NRisk_{i,t}
+
\gamma'B_{i,t}
+
u_{i,t+h}.
\]

如果控制 Barra 后网络风险仍稳定有效，才说明其具有更强的新增价值。

---

## 16. Step 7.13：样本外风险验证

必须至少采用：

- Development Period；
- OOS Evaluation Period；

或 walk-forward。

风险因子定义、阈值、Regime 参数不得根据 OOS 表现反复调整。

---

## 17. Step 7.14：风险预警能力评价

若目标包含早期预警，可定义：

\[
Y_{t,h}
=
I\{FutureRisk_{t,h}>	au\}.
\]

评价：

- ROC-AUC；
- PR-AUC；
- hit rate；
- false alarm rate；
- lead time；
- calibration。

不能只展示“某次危机前指标上升”。

---

## 18. Stage 7 中必须避免的解释错误

### 18.1 Correlation 不等于 causality

网络关联不应自动解释为因果传播。

### 18.2 高中心性不等于风险源头

高中心性仅表示网络位置重要。

### 18.3 高 Transition 不等于危机

\[
TI_t
\]

只表示网络变化大。

### 18.4 高 Novelty 不等于危险

Novelty 只表示历史罕见。

是否具有风险预警意义必须通过未来风险检验。

---

## 19. Stage 7 建议输出文件

```text
stage7_network_risk/
├── risk_research_protocol.md
├── market_network_risk_factors.csv
├── stock_network_risk_factors.parquet
├── transition_intensity.csv
├── structural_novelty.csv
├── edge_turnover_rewiring.csv
├── industry_pair_risk_decomposition.csv
├── node_risk_driver_scores.parquet
├── future_market_risk_labels.csv
├── future_stock_risk_labels.parquet
├── market_risk_regression_summary.csv
├── stock_risk_fama_macbeth_summary.csv
├── risk_sorting_summary.csv
├── tail_event_prediction_summary.csv
├── regime_summary.csv
├── main_vs_benchmark_risk.csv
├── barra_incremental_risk_test.csv
├── oos_risk_summary.csv
├── robustness_summary.csv
├── risk_factor_scorecard.csv
└── validated_risk_factor_list.csv
```

---

## 20. 建议输出图

至少包括：

1. `network_risk_factors_over_time.png`
2. `transition_intensity_over_time.png`
3. `structural_novelty_over_time.png`
4. `future_volatility_vs_network_risk.png`
5. `risk_sorting_future_volatility.png`
6. `tail_event_early_warning.png`
7. `industry_pair_risk_contribution.png`
8. `top_node_risk_drivers.png`
9. `regime_risk_comparison.png`
10. `main_vs_benchmark_risk.png`

---

## 21. Stage 7 常见问题及回答

### Q1. Network Density 越高就代表风险越大吗？

不一定。若网络固定密度，Density 甚至没有风险含义。此时应看 MeanAbsWeight、Centralization、CrossStrength、Transition、Novelty 等。

### Q2. 高中心性股票一定是风险源吗？

不一定。高中心性只能说明网络位置重要。若要讨论方向性传播，需要 Lead-Lag、Granger 或其他 temporal evidence。

### Q3. Transition Intensity 高是否等于危机？

不等于。TI 只表示相邻网络变化大，必须进一步检验其对 FutureRisk 的前瞻关系。

### Q4. Novelty 与 Transition 有什么区别？

Transition：

\[
TI_t=d(R_t,R_{t-1})
\]

回答“和上一期相比变了多少”。

Novelty：

\[
N_t=d(R_t,\mathcal H_{t-1})
\]

回答“和整个历史相比有多罕见”。

### Q5. Same-industry 和 Cross-industry 哪个更重要？

不能预设。Same 更可能反映行业共同风险，Cross 更可能反映系统性传染和跨行业风险扩散，必须分别检验。

### Q6. 为什么要控制 Barra？

为了判断 Network Risk 是否只是 Size、Beta、Volatility、Liquidity、Industry 等传统风险暴露的重新表达。

### Q7. 为什么不能只分析危机事件？

只选危机分析容易产生 event selection bias。应先做全样本前瞻检验，再用典型事件解释机制。

### Q8. Regime Detection 是不是必须？

不是。只有在网络风险指标呈现明显阶段性时才值得正式建模。

### Q9. Tail Network 是否必须做？

不一定。如果 Stage 3/4 已显示尾部共振明显、且需要与 M4 联动，则值得保留；否则可作为扩展。

### Q10. Stage 7 是否要做因果推断？

第一版不建议。更合适的表述是 risk association、prediction、transmission pattern。若要声称因果传播，需要更严格识别设计。

### Q11. Stage 7 和 M4 有什么区别？

Stage 7 研究网络风险指标是否有效；M4 将经过验证的指标整合进尾部风险预警和风险应对机制。

### Q12. Stage 7 是否需要和 Stage 6 Alpha 联动？

建议。要检查某个 Alpha 是否同时承担了较高网络风险，避免把风险补偿误解释为纯 Alpha。

### Q13. 是否需要 OOS？

必须。否则很容易根据历史危机反复调阈值、Novelty 定义或 Regime 参数。

### Q14. 什么时候可以说“网络风险因子有效”？

至少需要：

1. 与未来风险存在稳定关系；
2. 多年份有效；
3. 控制传统风险指标后仍有信息；
4. OOS 仍有效；
5. Main Network 相比 Benchmark 有新增或互补信息；
6. 对窗口、密度、股票池较稳健；
7. 经济解释合理；
8. 不是由单一危机事件驱动。

---

## 22. 风险因子分级

### Level A：Descriptive Network Risk Metric

仅描述当前网络。

### Level B：Candidate Risk Factor

与未来风险存在一定关系。

### Level C：Validated Network Risk Factor

满足：

- OOS；
- 控制传统风险指标；
- 稳健性；
- benchmark comparison；
- 经济解释。

只有 Level C 才适合进入 M4 风险预警体系。

---

## 23. 风险因子评分表

建议：

`risk_factor_scorecard.csv`

| Factor | Future Vol | Future Drawdown | Tail Event | OOS | Barra Incremental | Benchmark Gain | Robustness | Interpretation | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| MeanAbsWeight | | | | | | | | | |
| CrossStrength | | | | | | | | | |
| Centralization | | | | | | | | | |
| Transition Intensity | | | | | | | | | |
| Structural Novelty | | | | | | | | | |
| Tail Density | | | | | | | | | |

Decision：

- Reject；
- Descriptive；
- Candidate；
- Validated。

---

## 24. Stage 7 完成判定标准

- [ ] 市场级 Network Risk factor library 已冻结；
- [ ] 个股级 Network Risk factor library 已冻结；
- [ ] 已计算 Transition / Rewiring；
- [ ] 已计算 Structural Novelty；
- [ ] 已构造未来市场风险标签；
- [ ] 已构造未来个股风险标签；
- [ ] 已完成市场级前瞻检验；
- [ ] 已完成个股级风险排序/回归；
- [ ] 已控制 Barra / traditional risk；
- [ ] 已完成 Main vs Benchmark；
- [ ] 已完成 OOS；
- [ ] 已完成尾部风险初步检验；
- [ ] 已完成行业/节点风险传导分解；
- [ ] 如适用，已完成 Regime 分析；
- [ ] 已形成 risk factor scorecard；
- [ ] 已明确哪些指标仅描述网络，哪些可进入 M4。

---

## 25. Stage 7 最终汇报模板

> **Stage 7 Network Risk Research Conclusion**
>
> 1. **Main Network**：________  
> 2. **Risk Factors Tested**：________  
> 3. **Best Future Volatility Predictor**：________  
> 4. **Best Drawdown Predictor**：________  
> 5. **Best Tail-Risk Indicator**：________  
> 6. **Barra-Incremental Information**：________  
> 7. **Main vs Benchmark Improvement**：________  
> 8. **OOS Performance**：________  
> 9. **Main Industry Risk Channels**：________  
> 10. **Main Node Risk Drivers**：________  
> 11. **Regime / Transition Finding**：________  
> 12. **Validated Network Risk Factors**：________  
> 13. **Rejected Factors and Reasons**：________  
> 14. **Recommended Inputs for M4**：________

---

## 26. Stage 7 的一句话目标

\[
oxed{
	extbf{
把 A 股全市场网络结构转化为市场级、个股级和传导级风险指标，并通过未来波动、回撤、尾部损失、Barra 增量信息、样本外与稳健性检验，识别真正具有风险预警和风险传导价值的网络因子。
}
\]
