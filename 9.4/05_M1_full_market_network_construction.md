# M1 Stage 5：A股全市场正式网络构建与网络数据库形成

## 1. Stage 5 的定位

Stage 5 的目标是：

\[
\boxed{
\text{将 Stage 4 已选定的 Main / Benchmark / Auxiliary 网络方法，正式应用到 A 股全市场和完整研究期，形成可供后续 Alpha 与 Risk 研究直接调用的标准化网络数据库。}
}
\]

前后关系为：

\[
\text{Stage 1：股票池定义}
\rightarrow
\text{Stage 2：数据质量验收}
\rightarrow
\text{Stage 3：市场结构诊断}
\rightarrow
\text{Stage 4：候选网络比较与选型}
\rightarrow
\boxed{\text{Stage 5：正式全市场网络构建}}
\rightarrow
\text{Stage 6：Alpha 因子挖掘}
\rightarrow
\text{Stage 7：Risk 因子挖掘}
\]

Stage 5 的核心任务不是继续“选方法”，而是把已经选好的方法工程化、标准化、全样本化。

---

# 2. Stage 5 的核心目标

Stage 5 完成后，应形成：

1. 全研究期、全股票池的网络时间序列；
2. 标准化 edge table / node table；
3. 网络构建参数与版本记录；
4. 全市场网络基础统计；
5. 计算日志与异常日志；
6. 可供 Stage 6/7 直接读取的统一接口。

一句话：

\[
\boxed{
\text{Stage 4 决定“用什么网络”，Stage 5 负责“把它完整、稳定、可复现地构出来”。}
}
\]

---

# 3. Stage 5 的核心输入

Stage 5 至少需要：

- Stage 1 的三层股票池定义；
- Stage 2 通过验收的数据；
- Stage 3 的市场结构诊断结果；
- Stage 4 的 `network_selection_decision.md`；
- Stage 4 固定的：
  - 主网络方法；
  - Benchmark；
  - 辅助网络（如有）；
  - 滚动窗口；
  - 更新频率；
  - 稀疏化规则；
  - 统一密度或 Top-k 方案；
  - 缺失处理；
  - Point-in-Time 规则。

---

# 4. Stage 5 必须首先冻结研究协议

正式全市场跑网络之前，必须建立：

`network_build_config.yaml`

建议至少包括：

```yaml
universe:
  main: hs_a_share
  include_bse: false

sample:
  start_date: YYYY-MM-DD
  end_date: YYYY-MM-DD

network:
  main_method: pending_stage4
  benchmark_method: pending_stage4
  auxiliary_methods: []

window:
  lookback: pending_stage4
  update_frequency: pending_stage4

sparsification:
  method: fixed_density_or_topk
  value: pending_stage4

missing:
  min_valid_obs_ratio: pending_stage2
  pairwise_rule: pending_stage4

point_in_time:
  enabled: true

output:
  save_full_dense_matrix: false
  save_sparse_edges: true
```

原则：

\[
\boxed{
\text{Stage 5 一旦正式开始，不应边跑边随意修改关键参数。}
}
\]

如需修改，应建立新版本配置并记录原因。

---

# 5. Step 5.1：生成每个网络估计日的动态股票池

对每个网络估计日期 \(t\)，生成：

\[
\mathcal U_t^{network}.
\]

输出：

`network_universe_by_date.parquet`

字段建议：

| trade_date | security_id | ts_code | in_network | exclusion_reason |
|---|---|---|---|---|

必须满足：

- Point-in-Time；
- 历史退市股票在上市期间保留；
- 新股按历史观测要求动态进入；
- 长期停牌/数据不足股票按 Stage 2 规则动态退出。

---

# 6. Step 5.2：构造网络输入矩阵

对于每个网络日期 \(t\)，构造窗口：

\[
[t-W+1,t].
\]

根据主网络类型生成输入：

## Raw Correlation

\[
R_t \in \mathbb R^{W\times p_t}.
\]

## Barra Residual Correlation

先得到：

\[
\varepsilon_{i,s},
\quad s=t-W+1,\ldots,t,
\]

再构造：

\[
E_t \in \mathbb R^{W\times p_t}.
\]

## Block Sparse Precision

按 Stage 4 确定的 block 划分，分别构造：

\[
R_{g,t}.
\]

## Lead-Lag

构造：

\[
X_t=(r_{i,s-1}),
\qquad
Y_t=(r_{j,s}).
\]

所有输入必须只使用：

\[
s\le t
\]

的数据，禁止未来信息泄漏。

---

# 7. Step 5.3：正式构建 Main Network

Stage 5 最重要的产出是主网络时间序列。

统一保存为稀疏边表：

`main_network_edges.parquet`

字段建议：

| network_date | source | target | weight | abs_weight | sign | selected | same_industry |
|---|---|---|---:|---:|---|---|---|

无向网络要求：

\[
(i,j)
\]

只保存一次，并满足：

\[
i<j
\]

或其他固定排序。

有向网络则保留：

\[
source\rightarrow target.
\]

---

# 8. Step 5.4：构建 Benchmark Network

至少保留一个基础 Benchmark，例如：

\[
\boxed{\text{Raw Correlation Network}}
\]

输出：

`benchmark_network_edges.parquet`

其作用不是取代主网络，而是为后续回答：

> 主网络的 Alpha/Risk 信息是否真正超过最简单相关网络？

---

# 9. Step 5.5：构建必要的 Auxiliary Network

如果 Stage 4 已明确：

- Lead-Lag 作为 Alpha auxiliary；
- Tail network 作为 Risk auxiliary；
- Block precision 作为 conditional benchmark；

则分别构建并保存。

但原则是：

\[
\boxed{
\text{只构建 Stage 4 已有明确用途的辅助网络，不无限增加新方法。}
}
\]

---

# 10. Step 5.6：统一稀疏化和边选择

如果网络本身是 dense，需要执行 Stage 4 已确定的稀疏化规则。

例如固定密度：

\[
Density_t
=
\frac{|E_t|}{p_t(p_t-1)/2}
=
d.
\]

或每节点 Top-k。

必须记录：

- 稀疏化前边数；
- 稀疏化后边数；
- density；
- threshold；
- Top-k；
- ties 处理规则。

输出：

`sparsification_log.csv`

---

# 11. Step 5.7：生成节点级网络特征底表

Stage 5 可以计算基础网络统计，但不做正式 Alpha 结论。

建议生成：

`network_node_features.parquet`

至少包括：

- degree；
- weighted_degree；
- in_degree / out_degree（有向）；
- PageRank；
- eigenvector centrality；
- clustering coefficient；
- community label；
- same-industry degree；
- cross-industry degree。

字段示例：

| network_date | security_id | degree | weighted_degree | pagerank | community | same_degree | cross_degree |
|---|---|---:|---:|---:|---|---:|---:|

这些是 Stage 6 Alpha 因子工程的基础输入。

---

# 12. Step 5.8：生成网络级时间序列

输出：

`network_level_summary.csv`

建议至少包括：

\[
NodeCount_t,
\]

\[
EdgeCount_t,
\]

\[
Density_t,
\]

\[
MeanAbsWeight_t,
\]

\[
SameEdgeRatio_t,
\]

\[
CrossEdgeRatio_t,
\]

\[
CommunityCount_t,
\]

\[
Centralization_t.
\]

这些是 Stage 7 Risk 研究的基础底表。

---

# 13. Step 5.9：做正式构网后的质量检查

虽然 Stage 4 已经做过候选比较，但 Stage 5 全样本跑完后仍需做一次 operational QA。

至少检查：

## 13.1 网络是否有异常空图

检查：

\[
|E_t|=0
\]

或边数异常骤降。

## 13.2 网络密度是否符合协议

如果固定密度，则检查：

\[
Density_t\approx d.
\]

## 13.3 节点覆盖是否异常

检查：

- isolated nodes；
- 异常大量节点退出；
- 行业集中缺失。

## 13.4 权重是否异常

检查：

- NaN；
- Inf；
- 超出理论范围；
- 极端异常值。

## 13.5 时间连续性

检查相邻网络：

\[
Jaccard_t
\]

是否出现无法解释的技术性跳变。

注意：

此处只做质量检查，不做正式 Regime/Transition 研究。

---

# 14. Step 5.10：记录运行日志和失败窗口

必须生成：

`network_build_log.csv`

字段建议：

| network_date | method | p | W | runtime_sec | success | edge_count | warning | error |
|---|---|---:|---:|---:|---|---:|---|---|

如果某个窗口失败：

- 不得静默跳过；
- 必须记录；
- 必须解释是否重跑；
- 必须明确是否影响后续研究。

---

# 15. Step 5.11：版本控制与可复现性

建议记录：

- data version；
- universe version；
- config version；
- code commit/hash；
- network method version；
- build date。

输出：

`network_metadata.json`

目的：

\[
\boxed{
\text{未来任何一个网络日期，都可以追溯它是如何被构造出来的。}
}
\]

---

# 16. Stage 5 最终建议目录

```text
stage5_full_market_network/
├── network_build_config.yaml
├── network_metadata.json
├── network_universe_by_date.parquet
├── main_network_edges.parquet
├── benchmark_network_edges.parquet
├── auxiliary_network_edges_*.parquet
├── network_node_features.parquet
├── network_level_summary.csv
├── sparsification_log.csv
├── network_build_log.csv
├── network_quality_summary.csv
├── network_sample_dates_summary.csv
└── figures/
```

---

# 17. Stage 5 应生成哪些图？

建议只生成用于质量和结构概览的图，不做过度展示。

至少包括：

1. `node_count_over_time.png`
2. `edge_count_over_time.png`
3. `density_over_time.png`
4. `mean_abs_weight_over_time.png`
5. `same_cross_ratio_over_time.png`
6. `adjacent_jaccard_over_time.png`
7. 若干代表性日期网络图

代表性网络图只用于展示结构，不能替代统计分析。

---

# 18. Stage 5 必须回答的核心问题

## Q1. Stage 5 与 Stage 4 有什么区别？

Stage 4：

\[
\boxed{\text{比较候选网络并决定用谁}}
\]

Stage 5：

\[
\boxed{\text{把选中的网络完整应用到全市场、全样本，并形成标准数据库}}
\]

因此二者不重复。

---

## Q2. Stage 5 是否只构建 Main Network？

不是。

建议至少保留：

- Main Network；
- Benchmark Network。

若 Stage 4 已明确某个 Auxiliary Network 对 Alpha 或 Risk 有必要，也应构建。

但不要把 Stage 4 已淘汰的方法全部再跑一次。

---

## Q3. Stage 5 是否必须使用 Rolling？

不一定。

是否 Rolling 应在 Stage 3/4 根据时间变化和任务目标决定。

Stage 5 只执行已冻结的更新规则，例如：

- rolling daily；
- rolling weekly；
- rolling monthly；
- expanding；
- fixed-period。

不能因为之前 15 股票用了 Rolling GLasso，就默认全市场仍必须 Rolling。

---

## Q4. 全市场 5000+ 股票是否应该保存完整矩阵？

通常不建议。

候选边约为：

\[
\frac{5000\times4999}{2}
\approx 12.5\text{ million}.
\]

如果每个日期都保存 dense matrix，存储成本很高。

建议主存储采用：

\[
\boxed{\text{sparse edge table}}
\]

而不是完整 \(p\times p\) 矩阵。

只有确有需要时再临时恢复矩阵。

---

## Q5. 网络节点每天变化怎么办？

允许：

\[
p_t
\]

随时间变化。

不能为了固定维度而只保留今天仍上市的股票。

应使用 Point-in-Time dynamic universe，并用稳定 `security_id` 跟踪节点。

---

## Q6. 新股如何进入网络？

当新股满足 Stage 2 已确定的最低历史观测要求后，才进入：

\[
\mathcal U_t^{network}.
\]

不是上市第一天立即进入所有网络。

---

## Q7. 停牌股票如何处理？

停牌日不应被视为真实 0 收益。

网络估计应按 Stage 2/4 的有效观测规则处理。

如果窗口内有效数据比例低于要求，则该股票在该网络日期暂不进入网络。

---

## Q8. 是否需要固定网络密度？

如果 Stage 4 已证明固定密度是公平比较和稳定构网的最佳方案，则 Stage 5 应严格执行。

如果最终主方法本身有稳定、可解释的稀疏化规则，也可使用该规则。

关键是：

\[
\boxed{
\text{规则必须在 Stage 4 已经冻结，而不是 Stage 5 边跑边调。}
}
\]

---

## Q9. Stage 5 能不能直接做 Alpha？

可以生成网络节点特征，但不应在本阶段做完整 Alpha 结论。

Stage 5 的输出：

\[
Degree,\ PageRank,\ NeighborStructure,\ldots
\]

是 Stage 6 的输入。

正式的：

\[
IC,\ RankIC,\ ICIR,\ Q1-Q5,\ Turnover,\ Capacity
\]

应放在 Stage 6。

---

## Q10. Stage 5 能不能做 Risk？

同理。

Stage 5 可以生成：

\[
Density_t,
\quad
SameEdgeRatio_t,
\quad
Centralization_t,
\]

但正式研究：

\[
FutureVolatility,
\quad
Drawdown,
\quad
TailLoss,
\]

应放在 Stage 7。

---

## Q11. 是否需要在 Stage 5 做 Regime / Transition / Novelty？

不建议。

Stage 5 只负责形成可靠网络时间序列。

Regime、Transition、Novelty 属于：

\[
\boxed{\text{Risk / Dynamic Network Analysis}}
\]

应在 Stage 7 中根据全市场正式网络重新研究。

---

## Q12. Stage 5 如何判断“构网成功”？

不能只看程序是否跑完。

至少要求：

1. 全研究期网络覆盖完整；
2. 构网失败窗口已解释；
3. 节点数量符合动态股票池；
4. 边密度符合协议；
5. 权重无系统异常；
6. 网络时间序列不存在明显技术性断裂；
7. Main / Benchmark 结构均可被 Stage 6/7 直接读取；
8. 所有结果可以复现。

---

# 19. Stage 5 的完成判定标准

只有以下项目全部完成后，Stage 5 才视为完成：

- [ ] Stage 4 的网络选型已经冻结；
- [ ] 建立统一 `network_build_config.yaml`；
- [ ] 构造全研究期动态 Network Universe；
- [ ] 完成 Main Network 全样本构建；
- [ ] 完成 Benchmark Network 全样本构建；
- [ ] 必要 Auxiliary Network 已构建；
- [ ] 稀疏化规则统一执行；
- [ ] 生成标准化 edge table；
- [ ] 生成 node feature table；
- [ ] 生成 network-level time series；
- [ ] 完成全样本构网 QA；
- [ ] 所有失败窗口有日志；
- [ ] 记录版本与元数据；
- [ ] Stage 6/7 可直接读取结果；
- [ ] 未提前将基础网络统计解释为正式 Alpha/Risk 结论。

---

# 20. Stage 5 最终汇报模板

> **Stage 5 Full-Market Network Construction Conclusion**
>
> 1. **Main Network**：________  
> 2. **Benchmark Network**：________  
> 3. **Auxiliary Network(s)**：________  
> 4. **Sample Period**：________  
> 5. **Network Update Frequency**：________  
> 6. **Lookback Window**：________  
> 7. **Sparsification Rule**：________  
> 8. **Average Node Count**：________  
> 9. **Average Edge Count / Density**：________  
> 10. **Build Success Rate**：________  
> 11. **Main Data Quality Issues**：________  
> 12. **Main Network Structural Characteristics**：________  
> 13. **Ready for Stage 6 Alpha Research**：Yes / No  
> 14. **Ready for Stage 7 Risk Research**：Yes / No

---

# 21. Stage 5 的一句话目标

\[
\boxed{
\textbf{
将 Stage 4 选出的网络方法工程化到 A 股全市场和完整历史样本，形成稳定、标准化、可复现的动态网络数据库，为后续 Alpha 与 Risk 因子研究提供统一基础。
}
}
