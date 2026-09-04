# M1 Stage 3：A股全市场数据结构诊断与网络方法选型依据

## 1. Stage 3 的定位

Stage 3 的目标是：

\[
\boxed{\text{在 Stage 2 数据质量通过后，系统识别 A 股全市场数据的统计结构、横截面结构和时间变化特征}}
\]

并据此回答：

\[
\boxed{\text{“什么样的股票关联网络更适合 A 股全市场？”}}
\]

Stage 3 **不是直接选定最终网络**，而是为 Stage 4 的候选网络比较和正式选型提供证据。

研究链条为：

\[
\text{Stage 1：研究对象定义}
\rightarrow
\text{Stage 2：数据质量验收}
\rightarrow
\boxed{\text{Stage 3：市场数据结构诊断}}
\rightarrow
\text{Stage 4：候选网络比较与选型}
\rightarrow
\text{Stage 5：全市场网络构建}
\]

---

## 2. Stage 3 的核心输入

Stage 3 至少需要：

1. Point-in-Time 股票池；
2. 复权日收益；
3. 历史行业分类；
4. 成交额、换手率、市值；
5. ST / 停牌状态；
6. Barra 暴露或可用于因子调整的数据；
7. 交易日历。

第一版主样本建议先采用沪深 A 股，北交所作为扩展样本或稳健性分析。

---

# 3. Stage 3 必须回答的核心问题

## Q1. A 股全市场是否属于典型的 \(p\gg n\) 高维问题？

对不同窗口：

\[
W\in\{60,120,252\}
\]

统计：

\[
p_t=|\mathcal U_t^{network}|
\]

以及：

\[
\frac{p_t}{W}.
\]

需要回答：

- 每个窗口大约有多少可估计股票；
- \(p_t/W\) 是否远大于 1；
- 全维协方差/精度矩阵估计是否面临严重高维问题；
- 是否需要稀疏化、分块、降维或因子调整。

若 \(p_t\gg W\)，则样本协方差矩阵可能奇异或高度不稳定。这不意味着 Graphical Lasso 一定不能用，但意味着不能默认“全市场一次性高维精度矩阵”就是最佳方案。

---

## Q2. A 股股票收益之间的原始相关性有多强？

计算：

\[
\rho_{ij,t}=Corr(r_i,r_j).
\]

至少统计：

- mean；
- median；
- standard deviation；
- quantiles；
- positive correlation ratio；
- \(|\rho|>0.2,0.3,0.5\) 的比例。

需要回答：

> A 股全市场是否存在强烈整体共同波动？

---

## Q3. 同行业股票是否显著比跨行业股票更相关？

定义：

\[
\mathcal E_{same}=\{(i,j):Industry_i=Industry_j\},
\]

\[
\mathcal E_{cross}=\{(i,j):Industry_i\neq Industry_j\}.
\]

比较：

\[
|\rho_{ij}|_{Same}
\quad\text{vs}\quad
|\rho_{ij}|_{Cross}.
\]

可定义：

\[
SCR_t=
\frac{Mean(|\rho|_{Same,t})}{Mean(|\rho|_{Cross,t})}.
\]

需要回答：

- 行业是否是网络的主要组织结构；
- 是否有依据采用 block / hierarchical network；
- 跨行业关联是否仍有重要信息。

若 Same 显著大于 Cross，只能说明行业分层可能是重要先验，并不能直接推出“只做行业内网络”。

---

## Q4. A 股收益是否存在强市场共同因子？

设窗口内收益矩阵：

\[
R_t\in\mathbb R^{W\times p_t}.
\]

对相关矩阵或适当处理后的收益矩阵做 PCA / eigenvalue 分析。

重点计算：

\[
EVR_1=\frac{\lambda_1}{\sum_k\lambda_k},
\]

以及：

\[
EVR_{1:5},\quad EVR_{1:10},\quad EVR_{1:20}.
\]

需要回答：

- 第一主成分是否解释大量共同波动；
- 前若干主成分是否呈明显低秩结构；
- 股票相关性是否主要由市场/行业/风格共同因子驱动。

---

## Q5. 去除 Barra / 市场 / 行业共同因子后，股票关联还剩多少？

设：

\[
r_{i,t}=\alpha_i+\beta_i^\top F_t+\varepsilon_{i,t}.
\]

比较：

\[
Corr(r_i,r_j)
\]

与：

\[
Corr(\varepsilon_i,\varepsilon_j).
\]

至少报告：

\[
Mean|Corr_{raw}|,
\]

\[
Mean|Corr_{resid}|,
\]

并比较 residualization 前后 Same/Cross 结构。

需要回答：

> A 股股票间的表面关联，有多少只是已知共同风险暴露造成的？

若 residualization 后相关性明显下降但局部结构仍稳定，则 Barra-residual association network 应成为重要候选。

---

## Q6. 市值和流动性是否系统性影响相关结构？

按市值、成交额、换手率分组，例如：

\[
Q1,\ldots,Q5.
\]

比较不同组的：

- 平均相关性；
- 缺失率；
- 有效观测数；
- 零收益比例；
- 后续网络稳定性。

需要回答：

- 大市值股票是否机械地更容易成为高中心性节点；
- 低流动性是否因非同步交易导致相关性偏差；
- 后续是否需要流动性过滤或中性化处理。

---

## Q7. 非同步交易和低流动性是否影响相关性与 lead-lag？

重点检查：

- 连续零收益比例；
- 无成交日比例；
- 停牌之外的零成交情况；
- 小市值/低流动性股票的 lagged correlation。

可比较：

\[
Corr(r_{i,t},r_{j,t})
\]

与：

\[
Corr(r_{i,t-1},r_{j,t}).
\]

需要回答：

> 观察到的 lead-lag 是否可能只是非同步交易造成，而不是真实的信息扩散？

---

## Q8. 收益关系是否具有显著时间变化？

对滚动窗口计算：

- average correlation；
- Same/Cross correlation；
- eigenvalue concentration；
- residual correlation；
- cross-sectional dispersion。

观察：

\[
Structure_t.
\]

需要回答：

- 网络是否需要动态化；
- 静态全样本网络是否明显不合适；
- Rolling / time-varying framework 是否有必要。

本阶段只判断时间变化是否明显，不做正式 Regime Detection。

---

## Q9. 不同窗口长度是否导致明显不同的数据结构？

比较：

\[
W=60,120,252.
\]

重点看：

- 有效股票数量；
- correlation dispersion；
- Same/Cross ratio；
- leading eigenvalue；
- residual correlation；
- temporal stability。

Stage 3 不选“唯一最佳窗口”，只识别合理候选范围。

---

## Q10. 原始收益是否存在明显非高斯、厚尾和极端共振？

至少统计：

- skewness；
- kurtosis；
- tail quantiles；
- extreme co-movement frequency。

例如定义：

\[
I_{i,t}^{tail}=1\{r_{i,t}\le q_{i,\alpha}\}.
\]

再考察 pairwise tail co-occurrence。

需要回答：

- Gaussian-based conditional association 是否可能遗漏重要尾部结构；
- tail-dependence network 是否值得进入 Risk 方向候选。

---

## Q11. 全市场是否具有明显 block / community 特征？

除行业标签外，可探索：

- hierarchical clustering；
- spectral clustering；
- community detection on a simple benchmark graph。

需要回答：

- 数据驱动社区是否与行业高度重合；
- 是否存在稳定跨行业社区；
- 行业先验是否足够，还是需要 data-driven block。

Stage 3 中 community detection 只是结构诊断工具，不是最终网络结论。

---

## Q12. Stage 3 最终支持哪些网络候选进入 Stage 4？

形成“数据特征 → 方法含义”的证据映射：

| 数据特征 | 对网络方法的含义 |
|---|---|
| 强市场共同因子 | 考虑 residualization |
| Same >> Cross | 考虑 hierarchical/block network |
| \(p\gg n\) | 避免直接无约束全维 precision |
| 非同步交易明显 | lead-lag 需谨慎校正 |
| 厚尾明显 | Risk 方向保留 tail network |
| 时间变化显著 | 使用 rolling/time-varying framework |
| residual structure 稳定 | residual correlation 作为主候选 |

Stage 3 的结论是：

\[
\boxed{\text{哪些网络值得进入 Stage 4 比较}}
\]

而不是：

\[
\boxed{\text{哪一个网络已经被确定为最终模型}}
\]

---

# 4. Stage 3 的实际执行步骤

## Step 3.1 构造统一收益面板

生成：

`daily_return_panel.parquet`

至少包含：

- trade_date；
- security_id；
- return；
- industry；
- market_cap；
- amount；
- turnover；
- Barra exposures；
- network universe flag。

---

## Step 3.2 统计高维比例

输出：

`dimension_structure_summary.csv`

字段示例：

| trade_date | window | eligible_count | p_over_W |
|---|---:|---:|---:|

---

## Step 3.3 计算原始收益相关结构

输出：

- `raw_correlation_summary.csv`
- `raw_correlation_quantiles.csv`

图：

- `raw_corr_distribution.png`
- `average_corr_over_time.png`

---

## Step 3.4 Same vs Cross Industry

输出：

- `same_cross_correlation_summary.csv`
- `same_cross_by_industry.csv`

图：

- `same_cross_distribution.png`
- `same_cross_ratio_over_time.png`

---

## Step 3.5 PCA / Eigenvalue 结构

输出：

`eigen_structure_summary.csv`

至少包括：

- lambda_1；
- EVR_1；
- EVR_5；
- EVR_10；
- EVR_20。

图：

- `eigenvalue_scree.png`
- `EVR_over_time.png`

---

## Step 3.6 Raw vs Barra Residual

输出：

- `raw_vs_residual_correlation.csv`
- `raw_vs_residual_same_cross.csv`

图：

- `raw_vs_residual_corr_distribution.png`
- `residualization_effect_over_time.png`

---

## Step 3.7 市值与流动性异质性

输出：

- `correlation_by_size_group.csv`
- `correlation_by_liquidity_group.csv`
- `zero_return_liquidity_summary.csv`

---

## Step 3.8 时间变化与窗口敏感性

输出：

- `rolling_structure_summary.csv`
- `window_sensitivity_summary.csv`

---

## Step 3.9 尾部结构初筛

输出：

- `tail_co_movement_summary.csv`
- `tail_dependence_by_industry.csv`

---

## Step 3.10 形成网络候选建议

输出：

`network_candidate_evidence_matrix.csv`

字段建议：

| candidate | supporting_evidence | conflicting_evidence | computational_issue | role |
|---|---|---|---|---|

---

# 5. Stage 3 最终建议目录

```text
stage3_market_structure/
├── daily_return_panel.parquet
├── dimension_structure_summary.csv
├── raw_correlation_summary.csv
├── raw_correlation_quantiles.csv
├── same_cross_correlation_summary.csv
├── same_cross_by_industry.csv
├── eigen_structure_summary.csv
├── raw_vs_residual_correlation.csv
├── raw_vs_residual_same_cross.csv
├── correlation_by_size_group.csv
├── correlation_by_liquidity_group.csv
├── zero_return_liquidity_summary.csv
├── rolling_structure_summary.csv
├── window_sensitivity_summary.csv
├── tail_co_movement_summary.csv
├── network_candidate_evidence_matrix.csv
└── figures/
```

---

# 6. Stage 3 的核心结论模板

实际分析完成后，最终报告建议填写：

## 6.1 High-dimensional Structure

> 在主要滚动窗口下，A 股可用股票数约为 \(p_t=\_\_\_\)，对应 \(p_t/W=\_\_\_\)。因此，全市场网络属于 / 不属于显著 \(p\gg n\) 的高维场景。

## 6.2 Market Common Factor

> 第一主成分解释比例约为 \(\_\_\_\)，前 10 个主成分累计解释约为 \(\_\_\_\)。因此，共同市场/风格因子对股票相关结构的影响为：弱 / 中等 / 强。

## 6.3 Same vs Cross Industry

> 同行业平均绝对相关为 \(\_\_\_\)，跨行业为 \(\_\_\_\)，比值约为 \(\_\_\_\)。因此，行业结构对全市场网络的组织作用为：弱 / 中等 / 强。

## 6.4 Barra Residualization

> Residualization 后平均绝对相关由 \(\_\_\_\) 降至 \(\_\_\_\)，但仍存在 / 不存在稳定局部结构。

## 6.5 Liquidity and Size

> 低流动性股票呈现 \(\_\_\_\)；大市值股票呈现 \(\_\_\_\)。因此后续网络构建需要 / 不需要进行流动性过滤或中性化处理。

## 6.6 Time Variation

> 相关结构在时间上表现为 \(\_\_\_\)，说明静态网络 / 动态网络更合适。

## 6.7 Tail Structure

> 极端下跌共同发生结构为 \(\_\_\_\)，因此 tail-dependence network 在 Risk 方向应保留 / 暂不优先。

## 6.8 Candidate Network Recommendation

Stage 3 只给出进入 Stage 4 的候选，例如：

1. Raw correlation network —— benchmark；
2. Barra-residual correlation network —— 主候选；
3. Block sparse precision network —— conditional-association benchmark；
4. Lead-lag network —— Alpha / spillover candidate；
5. Tail-dependence network —— Risk extension。

---

# 7. Stage 3 的完成判定标准

只有以下任务完成后，Stage 3 才视为完成：

- [ ] 已计算不同窗口下 \(p_t/W\)；
- [ ] 已分析原始收益相关分布；
- [ ] 已比较 Same vs Cross Industry；
- [ ] 已完成 PCA / eigenvalue 结构分析；
- [ ] 已完成 Raw vs Barra Residual 比较；
- [ ] 已分析市值/流动性异质性；
- [ ] 已检查非同步交易和零收益问题；
- [ ] 已分析时间变化；
- [ ] 已比较不同窗口；
- [ ] 已完成尾部结构初筛；
- [ ] 已形成 network candidate evidence matrix；
- [ ] 已明确哪些候选网络进入 Stage 4；
- [ ] 未提前宣布最终网络模型。

---

# 8. Stage 3 不应该做的事情

Stage 3 不应：

- 直接确定最终网络；
- 因为已有 Rolling GLasso 经验而默认全市场继续使用；
- 做正式 Alpha IC / Rank IC；
- 做组合回测；
- 做 Network Regime Detection；
- 做 Transition Intensity；
- 做 Structural Novelty；
- 做正式 tail-risk prediction。

这些属于 Stage 4 及以后。

---

# 9. Stage 3 中常见问题及回答

## 问题 1：为什么不能直接用 Pearson 构网？

因为 Pearson 可能主要反映市场共同波动、行业共同风险和风格共振。因此 Stage 3 需要先判断原始相关中有多少是共同因子造成的。

## 问题 2：为什么不能直接用 Rolling Graphical Lasso？

因为全市场很可能满足 \(p\gg W\)，并且存在强行业、风格和共同因子结构。Rolling GLasso 可以作为候选 benchmark，但不能在诊断前默认其为最优全市场网络。

## 问题 3：为什么要做 Barra residualization？

因为它有助于区分“共同风险暴露导致的相关”和“股票间剩余局部关联”。如果目标是发现传统风险模型之外的新关联，这一步尤其重要。

## 问题 4：为什么 Same/Cross Industry 很重要？

因为如果股票关联具有明显 block structure，则直接把数千只股票视为同质节点可能不是最自然的建模方式。这可能支持“行业内精细网络 + 行业间粗粒度网络”。

## 问题 5：为什么还要分析流动性？

因为低流动性会导致非同步交易、大量零收益和相关估计偏差，甚至产生虚假的 lead-lag。网络边可能反映交易机制，而非真实经济关联。

## 问题 6：为什么 Stage 3 不直接选最终窗口？

因为窗口长度存在 bias-variance trade-off：短窗口动态但噪声高，长窗口稳定但可能过度平滑。Stage 3 只找合理候选范围，最终窗口应结合 Stage 4 网络稳定性和下游 Alpha/Risk 效果共同确定。

## 问题 7：为什么要做尾部结构初筛？

M1 不仅需要 Alpha，也需要 Risk，并且要与尾部风险研究联动。如果普通相关无法描述极端共振，tail network 可能成为 Risk 方向的重要补充。

---

# 10. Stage 3 最终一句话目标

\[
\boxed{
\textbf{通过高维结构、行业结构、共同因子、流动性、时间变化和尾部特征诊断，明确 A 股全市场网络构建所面临的真实数据结构，并形成有证据支持的候选网络集合。}
}
\]
