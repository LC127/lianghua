# M1 Stage 4：候选股票关联网络比较、验证与正式选型

## 1. Stage 4 的定位

Stage 4 的目标是：

\[
\boxed{
\text{基于 Stage 3 已识别的 A 股全市场数据结构，对若干候选网络进行统一构建、验证和比较，最终选出进入正式全市场研究的主网络与辅助网络。}
}
\]

Stage 4 是整个 M1 中真正完成“网络方法选型”的阶段。

前后关系为：

\[
\text{Stage 1：股票池定义}
\rightarrow
\text{Stage 2：数据质量验收}
\rightarrow
\text{Stage 3：市场结构诊断}
\rightarrow
\boxed{\text{Stage 4：候选网络比较与选型}}
\rightarrow
\text{Stage 5：全市场网络构建}
\rightarrow
\text{Stage 6：Alpha 因子挖掘}
\rightarrow
\text{Stage 7：Risk 因子挖掘}
\]

Stage 4 不应以“模型越复杂越好”为原则，而应以：

\[
\boxed{
\text{统计合理性}
+
\text{计算可扩展性}
+
\text{网络稳定性}
+
\text{经济解释}
+
\text{下游 Alpha/Risk 可用性}
}
\]

作为统一评价标准。

---

# 2. Stage 4 的核心输入

Stage 4 至少需要 Stage 3 输出的以下结果：

1. 不同窗口下的 \(p_t/W\)；
2. 原始收益相关结构；
3. Same vs Cross Industry 结构；
4. PCA / eigenvalue 共同因子强度；
5. Raw vs Barra residual 相关结构；
6. 流动性和市值异质性；
7. 时间变化特征；
8. 非同步交易与 lag 结构；
9. 尾部共振初筛；
10. `network_candidate_evidence_matrix.csv`。

Stage 4 必须基于这些数据证据决定候选网络，而不是直接沿用 15 股票阶段的 Rolling Graphical Lasso。

---

# 3. Stage 4 建议保留的候选网络

具体候选应根据 Stage 3 结果调整，但第一版建议至少考虑以下四类。

## Candidate A：Raw Correlation Network

定义：

\[
w_{ij,t}^{raw}
=
Corr(r_i,r_j).
\]

定位：

\[
\boxed{\text{基础市场共振 benchmark}}
\]

### 优点

- 解释简单；
- 计算方便；
- 全市场可扩展；
- 便于和既有股票网络文献比较。

### 缺点

- 受到市场、行业、风格共同因子影响；
- 容易形成高密度网络；
- 不代表条件依赖。

---

## Candidate B：Barra-Residual Correlation Network

设：

\[
r_{i,t}
=
\alpha_i+
\beta_i^\top F_t+
\varepsilon_{i,t},
\]

定义：

\[
w_{ij,t}^{resid}
=
Corr(\varepsilon_i,\varepsilon_j).
\]

定位：

\[
\boxed{\text{去除传统共同风险后的局部关联主候选}}
\]

### 优点

- 能减少市场、行业、风格共同驱动；
- 更容易发现传统风险模型之外的局部关系；
- 适合 Network Alpha 与异质风险研究。

### 缺点

- 依赖 Barra 数据质量；
- residualization 方法会影响网络；
- 如果过度残差化，可能同时删除真实经济共振。

---

## Candidate C：Block / Hierarchical Sparse Precision Network

若 Stage 3 表明行业或社区 block structure 明显，可对 block 内估计稀疏精度矩阵。

例如：

\[
\hat\Omega_g
=
\arg\min_{\Omega_g\succ0}
\left[
\mathrm{tr}(S_g\Omega_g)
-\log\det(\Omega_g)
+\lambda_g\|\Omega_g\|_{1,off}
\right].
\]

边权：

\[
\rho_{ij|-ij}
=
-\frac{\hat\Omega_{ij}}
{\sqrt{\hat\Omega_{ii}\hat\Omega_{jj}}}.
\]

定位：

\[
\boxed{\text{条件关联 benchmark}}
\]

### 优点

- 能去除部分间接相关；
- 条件关联解释清晰；
- 保留此前 Graphical Lasso 研究积累。

### 缺点

- 全市场直接估计可能计算和稳定性不足；
- block 划分会影响结果；
- 正则化参数敏感；
- 不应默认全维 Rolling GLasso 为最终主方法。

---

## Candidate D：Lead-Lag / Directed Network

基础形式可从：

\[
w_{ij,t}^{lag}
=
Corr(r_{i,t-1},r_{j,t})
\]

开始，也可以进一步使用 regularized VAR、Granger 或其他方向性估计。

定位：

\[
\boxed{\text{信息扩散 / 邻居动量 / Spillover Alpha 候选}}
\]

### 优点

- 具有方向性；
- 更贴近信息传播；
- 对 Alpha 因子特别有价值。

### 缺点

- 容易受到非同步交易影响；
- 低流动性股票可能产生伪 lead-lag；
- 高维 VAR 计算成本更高。

---

## Candidate E：Tail-Dependence Network（Risk 扩展）

仅当 Stage 3 显示明显厚尾与极端共振时进入。

例如基于：

\[
P(r_i\le q_{i,\alpha},r_j\le q_{j,\alpha})
\]

或其他尾部依赖指标构网。

定位：

\[
\boxed{\text{尾部风险与风险传染网络}}
\]

不建议第一轮作为 Alpha 主网络，但可作为 Risk 专用辅助网络。

---

# 4. Stage 4 第一原则：统一比较口径

候选网络必须在尽可能一致的条件下比较。

至少统一：

- 股票池；
- 日期范围；
- 滚动窗口；
- rebalance frequency；
- 缺失处理；
- 同一组行业标签；
- 同一流动性规则；
- 输出稀疏度控制；
- 相同评价期。

否则不同网络结果不可公平比较。

---

# 5. 网络稀疏化必须公平

这是 Stage 4 的关键问题之一。

不同方法天然具有不同密度：

- Pearson 几乎是 fully connected；
- GLasso 天然 sparse；
- Lead-lag 可能很密；
- Tail network 取决于阈值。

因此不能直接比较原始网络。

建议至少采用一种统一密度方案：

## 方案 A：固定 Density

令所有候选网络每期具有相同边密度：

\[
Density_t = d.
\]

例如通过保留绝对权重最大的前 \(K\) 条边。

## 方案 B：每个节点 Top-k

每只股票保留：

\[
k
\]

个最强邻居。

## 方案 C：多密度稳健性

比较：

\[
d\in\{0.5\%,1\%,2\%,5\%\}
\]

或合理区间。

Stage 4 不建议仅使用单一任意阈值。

---

# 6. Stage 4 必须完成的五类评价

---

## 6.1 计算可扩展性

对每个候选网络记录：

- 单窗口运行时间；
- 内存占用；
- 是否支持 5000+ 股票；
- 是否支持 rolling；
- 是否支持增量更新；
- 是否需要大规模参数调优。

输出：

`network_computational_benchmark.csv`

建议字段：

| candidate | p | W | runtime_sec | peak_memory | success | remarks |
|---|---:|---:|---:|---:|---|---|

### 核心问题

> 某网络理论上合理，但如果每个窗口都无法稳定运行，是否还能作为 M1 主网络？

答案：不应。

M1 是全市场长期研究，计算可扩展性本身就是选型标准。

---

## 6.2 网络稳定性

候选网络至少比较：

### Edge Persistence

\[
Persistence_{ij}
=
\frac{1}{T}\sum_t I\{(i,j)\in E_t\}.
\]

### 相邻网络 Jaccard

\[
J_t
=
\frac{|E_t\cap E_{t-1}|}
{|E_t\cup E_{t-1}|}.
\]

### Weight Stability

对共同边比较：

\[
Corr(w_{ij,t},w_{ij,t-1}).
\]

### Window Sensitivity

比较：

\[
W=60,120,252.
\]

### Bootstrap / Subsample Stability

若计算允许，可对代表性样本进行 bootstrap 或 subsample。

输出：

- `network_stability_summary.csv`
- `edge_persistence_summary.csv`
- `window_sensitivity_network.csv`

---

## 6.3 经济解释与结构合理性

至少检查：

### Same vs Cross

\[
SameEdgeRatio_t
=
\frac{|E_t^{same}|}{|E_t|}.
\]

### 行业集中度

### 社区与行业一致性

可使用：

- NMI；
- ARI；
- purity；

作为辅助指标。

### 节点中心性与市值/流动性的关系

例如：

\[
Corr(Degree_i,MarketCap_i),
\]

\[
Corr(PageRank_i,Turnover_i).
\]

需要回答：

> 网络中心性是否仅仅是 Size 或 Liquidity 的另一种表达？

输出：

- `network_economic_structure_summary.csv`
- `centrality_exposure_summary.csv`

---

## 6.4 Alpha Utility：只做“初筛”，不做正式完整因子研究

Stage 4 可以做少量 downstream screening，因为最终选型不能完全脱离任务目标。

但此处只做：

\[
\boxed{\text{快速、统一、低自由度的 Alpha screening}}
\]

例如每个候选网络统一构造 2–3 个简单指标：

1. Weighted Degree；
2. Neighbor Momentum；
3. Network Residual / Peer Signal。

计算：

\[
IC_t
=
Spearman(Factor_{i,t},r_{i,t+h})
\]

例如：

\[
h=5,20.
\]

比较：

- Mean IC；
- ICIR；
- 正 IC 比例。

### 注意

Stage 4 的 Alpha 结果只用于：

\[
\boxed{\text{网络方法筛选}}
\]

不能替代 Stage 6 的正式 Alpha 因子研究。

---

## 6.5 Risk Utility：只做基础筛选

统一比较简单网络级指标：

- Density；
- Average absolute weight；
- Cross-industry edge ratio；
- concentration；
- simple network change measure。

然后考察：

\[
NetworkRisk_t
\rightarrow
FutureVolatility_{t+h}
\]

或：

\[
FutureDrawdown_{t+h}.
\]

同样只作为候选网络是否具有风险研究价值的初筛证据。

---

# 7. Stage 4 需要建立统一评分表

建议建立：

`network_selection_scorecard.csv`

示例：

| Candidate | Scalability | Stability | Interpretability | Alpha Utility | Risk Utility | Final Role |
|---|---:|---:|---:|---:|---:|---|
| Raw Corr | ... | ... | ... | ... | ... | Benchmark |
| Residual Corr | ... | ... | ... | ... | ... | Main / Auxiliary |
| Block Precision | ... | ... | ... | ... | ... | Conditional Benchmark |
| Lead-Lag | ... | ... | ... | ... | ... | Alpha Auxiliary |
| Tail | ... | ... | ... | ... | ... | Risk Auxiliary |

不建议机械计算一个“总分”决定模型。

更合理的是：

\[
\boxed{
\text{硬约束}
+
\text{多维证据}
}
\]

例如：

### 硬约束

- 必须可扩展到全市场；
- 必须具有可接受稳定性；
- 必须没有明显前视偏差；
- 必须可解释。

满足硬约束后，再比较 Alpha/Risk 表现。

---

# 8. Stage 4 最终应选“主网络 + 辅助网络”，而不是只选一个

M1 本身允许研究 1–2 类关联关系，因此 Stage 4 的合理结果通常不是只留下唯一网络。

建议最终角色结构：

## Main Network

作为后续全市场 Alpha/Risk 研究的核心网络。

例如：

\[
\boxed{\text{Barra-Residual Correlation Network}}
\]

如果它在稳定性、解释性和下游效果上表现最好。

## Benchmark Network

例如：

\[
\boxed{\text{Raw Correlation Network}}
\]

用于说明主网络相比最简单方法是否提供新增信息。

## Auxiliary Alpha Network

例如：

\[
\boxed{\text{Lead-Lag Network}}
\]

专门服务于信息扩散与邻居动量。

## Auxiliary Risk Network

例如：

\[
\boxed{\text{Tail Network}}
\]

用于尾部风险和 M4 联动。

不要求所有辅助网络都进入第一轮主交付。

---

# 9. Stage 4 必须回答的核心问题

## Q1. 是否应该继续使用 Rolling Graphical Lasso？

只能根据 Stage 4 的全市场比较回答。

可能的结论包括：

1. **保留为全市场主方法**：只有在计算、稳定性和经济解释均优秀时；
2. **保留为 block/sector conditional benchmark**：这是更现实的情况；
3. **只用于代表性股票池验证**：若全市场计算成本过高；
4. **不进入正式主研究**：若稳定性或计算可扩展性不足。

因此：

\[
\boxed{
\text{Rolling GLasso 是候选，不是默认答案。}
}
\]

---

## Q2. Raw Correlation 是否太简单？

“简单”不是缺点。

如果 Raw Correlation：

- 稳定；
- 易扩展；
- 下游 Alpha/Risk 有效；
- 经济解释清楚；

那么它完全可以作为正式 benchmark，甚至成为主网络之一。

方法复杂度不等于研究价值。

---

## Q3. Residual Correlation 一定优于 Raw Correlation 吗？

不一定。

Residualization 会删除已知共同因子，但有可能：

- 去除噪声；
- 也去除真实经济共振。

因此必须实证比较：

\[
Raw
\quad vs\quad
Residual.
\]

不能因为“更高级”就默认 residual 更好。

---

## Q4. 是否应该统一网络密度？

如果要公平比较结构指标和下游因子，**非常建议**。

否则：

\[
Degree,\ Density,\ Centrality
\]

可能只是由不同模型的天然稀疏度决定。

最稳妥的是：

\[
\boxed{
\text{相同股票池 + 相同窗口 + 多组统一密度}
}
\]

做比较。

---

## Q5. 为什么 Stage 4 可以看 IC，但 Stage 3 不可以？

因为 Stage 3 的任务是：

\[
\text{数据结构诊断}.
\]

而 Stage 4 的任务是：

\[
\text{网络方法选择}.
\]

既然最终网络要服务 Alpha/Risk，那么少量 downstream utility 可以作为网络选型证据。

但 Stage 4 只做：

\[
\boxed{\text{screening}}
\]

正式 Alpha 因子工程、neutralization、分层回测、成本、容量分析仍属于后续阶段。

---

## Q6. 是否可以直接用“哪个 IC 最高”决定网络？

不可以。

一个网络可能短期 IC 高，但：

- 极不稳定；
- turnover 极高；
- 计算成本不可接受；
- 本质上只是 Size proxy；
- OOS 不稳。

因此最终选择必须综合：

\[
\boxed{
Stability
+
Interpretability
+
Scalability
+
Alpha
+
Risk
}
\]

---

## Q7. 是否需要神经网络/GNN进入 Stage 4？

第一轮不建议作为必须候选。

原因：

- M1 当前核心是完成全市场关联网络和因子研究；
- 神经图学习需要额外模型设计、训练稳定性和可解释性验证；
- 5000+ 股票和有限滚动样本下容易过拟合。

若后续传统网络基线已经建立，可把 Neural Graph Learning 作为扩展方法单独比较。

---

# 10. Stage 4 的实际执行步骤

## Step 4.1 固定统一实验协议

确定：

- 主股票池；
- 日期范围；
- \(W\) 候选；
- rebalance frequency；
- density/top-k 候选；
- 缺失处理；
- 评价区间。

输出：

`network_comparison_protocol.md`

---

## Step 4.2 构建候选网络

对每个候选方法生成统一格式边表：

`network_edges_<candidate>.parquet`

字段建议：

| trade_date | stock_i | stock_j | weight | abs_weight | selected |
|---|---|---|---:|---:|---|

有向网络增加：

- source；
- target。

---

## Step 4.3 统一稀疏化

输出：

`sparsification_sensitivity.csv`

比较：

- fixed density；
- top-k；
- 多密度水平。

---

## Step 4.4 计算稳定性指标

输出：

- `network_stability_summary.csv`
- `edge_persistence_summary.csv`
- `adjacent_jaccard_summary.csv`

---

## Step 4.5 计算经济结构指标

输出：

- `same_cross_network_summary.csv`
- `community_industry_alignment.csv`
- `centrality_exposure_summary.csv`

---

## Step 4.6 计算可扩展性

输出：

`network_computational_benchmark.csv`

---

## Step 4.7 做最小 Alpha screening

统一计算：

- Weighted Degree；
- Neighbor Momentum；
- Network Residual。

输出：

`network_alpha_screening.csv`

---

## Step 4.8 做最小 Risk screening

统一计算：

- density；
- average weight；
- cross-industry ratio；
- simple change metric。

输出：

`network_risk_screening.csv`

---

## Step 4.9 形成最终选型

输出：

- `network_selection_scorecard.csv`
- `network_selection_decision.md`

最终明确：

1. Main Network；
2. Benchmark Network；
3. Optional Alpha Auxiliary；
4. Optional Risk Auxiliary；
5. 被淘汰方法及原因。

---

# 11. Stage 4 最终目录建议

```text
stage4_network_selection/
├── network_comparison_protocol.md
├── network_edges_raw_corr.parquet
├── network_edges_residual_corr.parquet
├── network_edges_block_precision.parquet
├── network_edges_lead_lag.parquet
├── sparsification_sensitivity.csv
├── network_computational_benchmark.csv
├── network_stability_summary.csv
├── edge_persistence_summary.csv
├── adjacent_jaccard_summary.csv
├── same_cross_network_summary.csv
├── community_industry_alignment.csv
├── centrality_exposure_summary.csv
├── network_alpha_screening.csv
├── network_risk_screening.csv
├── network_selection_scorecard.csv
├── network_selection_decision.md
└── figures/
```

---

# 12. Stage 4 完成判定标准

只有以下任务完成后，Stage 4 才视为完成：

- [ ] 已根据 Stage 3 证据确定候选网络；
- [ ] 所有候选网络使用统一样本与窗口协议；
- [ ] 已解决网络密度可比性问题；
- [ ] 已测试计算可扩展性；
- [ ] 已比较网络稳定性；
- [ ] 已比较 Same/Cross 与行业结构；
- [ ] 已检查中心性对 Size/Liquidity 的暴露；
- [ ] 已做最小 Alpha screening；
- [ ] 已做最小 Risk screening；
- [ ] 已形成 network selection scorecard；
- [ ] 已明确 Main / Benchmark / Auxiliary 网络；
- [ ] 已解释被淘汰方法原因；
- [ ] 未把 Stage 4 screening 误写成正式 Alpha/Risk 结论。

---

# 13. Stage 4 不应该做的事情

Stage 4 不应：

- 完成全部 Alpha 因子库；
- 做完整分层回测与交易成本模型；
- 做容量分析；
- 做正式 Risk factor research；
- 做 Regime / Transition / Novelty 深度分析；
- 因为小样本经验直接指定 Rolling GLasso；
- 只根据网络图视觉效果选模型；
- 只根据最高 IC 选模型。

---

# 14. Stage 4 最终汇报模板

建议最终向项目负责人汇报：

> **Stage 4 Network Selection Conclusion**
>
> 1. **Candidate Networks**：________  
> 2. **Common Evaluation Protocol**：________  
> 3. **Scalability Result**：________  
> 4. **Stability Result**：________  
> 5. **Economic Structure Result**：________  
> 6. **Alpha Screening Result**：________  
> 7. **Risk Screening Result**：________  
> 8. **Main Network**：________  
> 9. **Benchmark Network**：________  
> 10. **Auxiliary Network(s)**：________  
> 11. **Rejected Methods and Reasons**：________  
> 12. **Decision for Stage 5**：进入全市场正式网络构建 / 继续调整

---

# 15. Stage 4 的一句话目标

\[
\boxed{
\textbf{
在统一样本、统一稀疏度和统一评价标准下，对候选股票网络进行可扩展性、稳定性、经济解释和初步 Alpha/Risk 效用比较，最终确定 M1 正式研究所采用的主网络与辅助网络。
}
}
