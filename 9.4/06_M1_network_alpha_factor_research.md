# M1 Stage 6：网络 Alpha 因子挖掘、检验与回测

## 1. Stage 6 的定位

Stage 6 的目标是：

\[
\boxed{
\text{将 Stage 5 已经构建完成的 A 股全市场网络转化为股票层面的可交易 Alpha 信号，并通过严格的横截面检验和组合回测判断其是否具有新增预测信息。}
}
\]

Stage 6 是 M1 中真正回答“网络能不能产生 Alpha”的阶段。

整体流程为：

\[
\text{Stage 1：股票池定义}
\rightarrow
\text{Stage 2：数据质量验收}
\rightarrow
\text{Stage 3：市场结构诊断}
\rightarrow
\text{Stage 4：网络选型}
\rightarrow
\text{Stage 5：全市场正式构网}
\rightarrow
\boxed{\text{Stage 6：Alpha 因子研究}}
\rightarrow
\text{Stage 7：Risk 因子研究}
\]

Stage 6 不应只看某个网络指标与未来收益是否“相关”，而应完成：

\[
\boxed{
\text{因子定义}
+
\text{横截面检验}
+
\text{中性化}
+
\text{分层回测}
+
\text{换手率}
+
\text{交易成本}
+
\text{容量}
+
\text{稳健性}
+
\text{样本外验证}
}
\]

---

# 2. Stage 6 的核心输入

至少需要 Stage 5 输出：

- `main_network_edges.parquet`
- `benchmark_network_edges.parquet`
- `network_node_features.parquet`
- `network_universe_by_date.parquet`

以及：

- 复权价格/收益；
- 成交额、换手率、市值；
- 行业；
- ST/停牌/涨跌停状态；
- Barra 风格暴露；
- 交易日历；
- Tradable Universe。

所有特征与未来收益必须严格按照 Point-in-Time 原则构造。

---

# 3. Stage 6 的第一原则：定义“信号形成时点”和“可交易时点”

如果网络在交易日 \(t\) 收盘后基于截至 \(t\) 的数据构建，则网络因子：

\[
Factor_{i,t}
\]

最早只能用于：

\[
t+1
\]

或更晚的交易决策。

禁止使用：

\[
r_{i,t+1:t+h}
\]

参与 \(Factor_{i,t}\) 的计算。

建议统一定义：

- Signal Date：\(t\)
- First Tradable Date：\(t+1\)
- Holding Horizon：\(h\in\{1,5,20\}\) 或 Stage 6 固定候选
- Rebalance Frequency：与 Stage 5 网络更新频率一致或整数倍关系

---

# 4. Step 6.1：建立第一版网络 Alpha 因子库

第一版不建议一次构造几十个高度相似的指标，而应从“拓扑位置、邻居信息、相对表现、方向性传播”四类开始。

---

## 4.1 Degree / Weighted Degree

### Degree

\[
Degree_{i,t}
=
\sum_{j}
I\{(i,j)\in E_t\}.
\]

反映股票在当前网络中的连接数量。

### Weighted Degree / Strength

\[
Strength_{i,t}
=
\sum_j |w_{ij,t}|.
\]

若边权正负本身有明确含义，也可进一步构造：

\[
PositiveStrength_{i,t}
=
\sum_j \max(w_{ij,t},0),
\]

\[
NegativeStrength_{i,t}
=
\sum_j \min(w_{ij,t},0).
\]

### 研究问题

- 高连接股票未来收益是否更高或更低？
- 高连接度是否只是大市值/高流动性的代理？
- 同行业连接和跨行业连接是否具有不同 Alpha 含义？

---

## 4.2 Same / Cross Industry Connectivity

构造：

\[
SameDegree_{i,t},
\]

\[
CrossDegree_{i,t},
\]

以及：

\[
CrossShare_{i,t}
=
\frac{CrossDegree_{i,t}}{Degree_{i,t}}.
\]

可进一步构造 weighted versions。

### 研究问题

- 跨行业“桥梁股票”是否包含更多信息扩散 Alpha？
- 高同行业中心性是否只是行业 beta 的替代？
- Cross-connectivity 是否具有更强的样本外收益信息？

---

## 4.3 PageRank / Eigenvector Centrality

生成：

\[
PageRank_{i,t}
\]

和：

\[
EC_{i,t}.
\]

这些指标反映股票与“重要节点”相连的程度，而不是单纯边数。

### 研究问题

- 高中心性股票是否有中心性溢价/折价？
- 中心性与 Size、Beta、Liquidity 的相关性有多高？
- 中性化后是否仍有独立信息？

---

## 4.4 Neighbor Momentum

这是 Stage 6 非常值得重点研究的网络 Alpha。

设股票 \(i\) 的邻居集合为：

\[
N_i(t).
\]

定义：

\[
NM_{i,t}^{(L)}
=
\frac{
\sum_{j\in N_i(t)}
\tilde w_{ij,t}
Mom_{j,t}^{(L)}
}{
\sum_{j\in N_i(t)}
|\tilde w_{ij,t}|
}.
\]

其中 \(Mom_{j,t}^{(L)}\) 为邻居过去 \(L\) 日动量。

建议：

\[
L\in\{5,20,60\}.
\]

### 可进一步区分

\[
NM^{same}_{i,t}
\]

与：

\[
NM^{cross}_{i,t}.
\]

### 研究问题

- 邻居过去上涨是否领先股票自身未来上涨？
- Cross-industry neighbor momentum 是否比 Same-industry 更有预测力？
- 该信号是否只是普通市场/行业动量的复制？

---

## 4.5 Network Peer Return / Network Residual

定义网络隐含同伴收益：

\[
PeerRet_{i,t}
=
\frac{
\sum_j w_{ij,t} r_{j,t}
}{
\sum_j |w_{ij,t}|
}.
\]

构造股票相对网络同伴的偏离：

\[
NetResidual_{i,t}
=
r_{i,t}
-
PeerRet_{i,t}.
\]

也可以使用过去 \(L\) 日累计版本。

### 研究假设

如果网络反映“经济相似股票”，则异常偏离可能存在：

- reversal；
- continuation；

两种可能。

因此 Stage 6 不应先假设符号，而应实证检验。

---

## 4.6 Community-relative Factor

如果 Stage 5 已生成 community label \(c(i,t)\)，定义：

\[
CommunityMom_{c,t}
=
\frac{1}{|c|}
\sum_{j\in c}
Mom_{j,t},
\]

以及：

\[
CommunityRelativeMom_{i,t}
=
Mom_{i,t}
-
CommunityMom_{c(i,t),t}.
\]

还可构造：

- community-relative valuation；
- community-relative volatility；
- community-relative liquidity。

第一版建议先从 momentum/return 开始，避免因子数量过多。

---

## 4.7 Lead-Lag Signal（若 Stage 5 保留有向网络）

如果存在：

\[
j\rightarrow i,
\]

定义：

\[
LeadSignal_{i,t}
=
\sum_j
w_{ji,t}
Signal_{j,t}.
\]

例如：

\[
LeadReturn_{i,t}
=
\sum_j
w_{ji,t} r_{j,t}.
\]

### 研究问题

- leader 的最新收益是否能预测 follower？
- 该效应控制自身动量后是否仍存在？
- 低流动性导致的非同步交易是否解释了结果？

---

# 5. Step 6.2：构造 Benchmark 因子

网络 Alpha 不能脱离传统因子比较。

建议至少保留：

- Size；
- Momentum；
- Reversal；
- Volatility；
- Liquidity；
- Turnover；
- Beta；
- Valuation（若数据可用）；
- Barra 风格因子。

Stage 6 要回答：

\[
\boxed{
\text{网络因子提供的是新增信息，还是传统因子的重包装？}
}
\]

---

# 6. Step 6.3：因子预处理

对每日横截面因子建议统一处理。

---

## 6.1 缺失处理

不建议为了保留股票而大规模填充网络因子。

应明确：

- 无网络邻居；
- 网络因子不可计算；
- 当日不可交易；
- 原始数据缺失；

分别如何处理。

---

## 6.2 去极值

可使用：

- MAD；
- percentile winsorization；

但必须固定规则并记录。

例如：

\[
x_{i,t}^{clip}
=
Clip(x_{i,t},q_{1\%},q_{99\%}).
\]

不要根据结果临时改变阈值。

---

## 6.3 标准化

横截面 z-score：

\[
z_{i,t}
=
\frac{x_{i,t}-\bar x_t}{s_t}.
\]

---

## 6.4 行业/风格中性化

建议至少比较三种版本：

### Raw Factor

\[
F_{i,t}.
\]

### Industry-neutral

横截面回归：

\[
F_{i,t}
=
\alpha_t
+
Industry_{i,t}'\gamma_t
+
u_{i,t}.
\]

使用：

\[
u_{i,t}
\]

作为行业中性因子。

### Barra-neutral

\[
F_{i,t}
=
\alpha_t
+
B_{i,t}'\gamma_t
+
u_{i,t}.
\]

其中 \(B_{i,t}\) 包含：

- Size；
- Beta；
- Momentum；
- Liquidity；
- Volatility；
- Value；
- Industry 等。

### 核心问题

若一个网络因子只有 Raw 版本有效，而 Barra-neutral 后失效，则说明：

\[
\boxed{
\text{其 Alpha 可能主要来自传统风险暴露，而不是网络新增信息。}
}
\]

---

# 7. Step 6.4：构造未来收益标签

至少考虑：

\[
h\in\{1,5,20\}.
\]

定义：

\[
R_{i,t\rightarrow t+h}.
\]

若使用收盘后信号，则避免把 \(t\) 当日收益重复计入未来标签。

例如：

\[
R^{(5)}_{i,t}
=
\log\frac{P_{i,t+5}}{P_{i,t}}.
\]

但真实交易回测应根据实际成交时点进一步调整。

---

# 8. Step 6.5：IC / Rank IC 检验

M1 的正式交付要求之一就是 IC，因此 Stage 6 必须系统完成。

建议主要使用 Spearman Rank IC：

\[
IC_t^{rank}
=
Spearman(F_{i,t},R_{i,t+h}).
\]

同时可辅助报告 Pearson IC。

需要报告：

\[
MeanIC,
\]

\[
MedianIC,
\]

\[
ICStd,
\]

\[
ICIR
=
\frac{MeanIC}{ICStd},
\]

\[
ICPositiveRatio
=
P(IC_t>0).
\]

如果计算 t-stat，要注意：

\[
IC_t
\]

通常具有时间序列自相关，因此不建议简单使用 iid 标准误。

可采用：

- Newey-West/HAC；
- block bootstrap；

进行更稳健的推断。

---

# 9. Step 6.6：IC 的时间稳定性

Alpha 研究不能只看全样本平均 IC。

必须分解：

- 年度；
- 牛/熊/震荡阶段；
- 大小盘；
- 行业；
- 高/低波动；
- 高/低流动性；
- Network Regime（如果后续已有合理定义，可作为附加分析）。

至少输出：

\[
IC_{year}
\]

和 rolling IC。

要回答：

> 因子是否长期有效，还是只由少数年份贡献？

---

# 10. Step 6.7：因子分层回测

M1 明确要求分层回测。

建议每日/每次调仓按因子排序形成：

\[
Q1,Q2,Q3,Q4,Q5
\]

或：

\[
D1,\ldots,D10.
\]

计算每层：

- equal-weight return；
- value-weight return（可选）；
- cumulative return；
- annualized return；
- volatility；
- Sharpe；
- max drawdown。

重点检查：

\[
Q5-Q1
\]

或：

\[
D10-D1.
\]

同时检查是否有：

\[
\boxed{\text{monotonicity}}
\]

而不仅仅是最高组偶然跑赢。

---

# 11. Step 6.8：Long-Short 组合

定义：

\[
LS_t
=
R_{High,t}
-
R_{Low,t}.
\]

报告：

- annualized return；
- annualized volatility；
- Sharpe；
- max drawdown；
- win rate；
- turnover；
- exposure。

如果 A 股实际不允许自由裸卖空，则 Long-Short 主要用于：

\[
\boxed{\text{因子统计评价}}
\]

而不是直接代表可执行策略。

实际可交易版本可进一步测试：

- Long-only top quantile；
- benchmark-enhanced portfolio。

---

# 12. Step 6.9：换手率

M1 明确要求换手率分析。

例如：

\[
Turnover_t
=
\frac{1}{2}
\sum_i |w_{i,t}-w_{i,t-1}|.
\]

对：

- Q5；
- Q1；
- Long-Short；
- Long-only；

分别统计。

需要回答：

> Alpha 是否因为网络变化过快而伴随过高换手？

---

# 13. Step 6.10：交易成本

至少做简单成本敏感性：

\[
Cost_t
=
c\times Turnover_t.
\]

比较：

\[
c\in\{5,10,20,30\}\text{ bps}
\]

或根据公司内部实际假设设置。

净收益：

\[
R_t^{net}
=
R_t^{gross}
-
Cost_t.
\]

真正有效的网络 Alpha 应至少对合理交易成本具有一定稳健性。

---

# 14. Step 6.11：容量分析

M1 明确要求容量分析。

第一版不必建立复杂 market-impact model，但至少完成：

## 14.1 ADV 约束

设：

\[
ADV_{i,t}^{20}
\]

为 20 日平均成交额。

定义 participation：

\[
Participation_{i,t}
=
\frac{TradeAmount_{i,t}}
{ADV_{i,t}^{20}}.
\]

测试例如：

\[
1\%, 5\%, 10\%
\]

的 participation cap。

## 14.2 可投资规模

根据组合权重、ADV 和调仓规模，估算：

\[
AUM^{max}.
\]

### 核心问题

> 因子是否主要集中在小盘、低流动性股票，导致纸面收益很高但无法承载资金？

---

# 15. Step 6.12：Barra 风险暴露检查

对 High-Low 或 Top Portfolio 检查：

- Size；
- Beta；
- Momentum；
- Liquidity；
- Volatility；
- Value；
- Industry。

需要回答：

> 因子收益是否来自未经控制的风格或行业暴露？

输出：

`factor_barra_exposure_summary.csv`

---

# 16. Step 6.13：增量信息检验

这是 Stage 6 很重要的一步。

如果网络因子为：

\[
N_{i,t}
\]

传统因子集合为：

\[
X_{i,t},
\]

可做 Fama-MacBeth 风格横截面回归：

\[
R_{i,t+h}
=
\alpha_t
+
\beta_t N_{i,t}
+
\gamma_t'X_{i,t}
+
u_{i,t+h}.
\]

然后研究：

\[
\bar\beta.
\]

这比单独 IC 更能回答：

\[
\boxed{
\text{网络信号是否提供条件于传统因子之外的增量信息？}
}
\]

若做推断，应考虑时间序列相关，使用 HAC 或适当 bootstrap。

---

# 17. Step 6.14：Main Network vs Benchmark Network

Stage 5 保留 Benchmark 的意义应在这里兑现。

例如同时构造：

\[
NM^{main}_{i,t}
\]

和：

\[
NM^{benchmark}_{i,t}.
\]

比较：

- IC；
- ICIR；
- quantile spread；
- turnover；
- net return；
- capacity。

需要回答：

> 复杂主网络是否真的优于 Raw Correlation Benchmark？

如果不能超过 benchmark，也要诚实记录，这是重要研究结论。

---

# 18. Step 6.15：样本内 / 样本外划分

不要把整个 2010–2026 数据全部用于调因子再报告同一时期结果。

建议至少采用：

\[
\text{Development Period}
\]

和：

\[
\text{OOS Evaluation Period}.
\]

例如：

- 前段用于设计；
- 后段冻结参数后做 OOS。

更严格可采用：

\[
\boxed{\text{walk-forward}}
\]

框架。

任何超参数：

- window；
- density；
- momentum lookback；
- rebalance frequency；

如果基于未来 OOS 表现选择，就会产生 look-ahead/data snooping。

---

# 19. Step 6.16：多重尝试与数据挖掘偏差

如果测试：

- 5 个网络；
- 20 个因子；
- 5 个 horizon；
- 5 个窗口；

很容易从噪声中找到“显著”结果。

因此应：

1. 第一版因子库预先固定；
2. 明确 exploratory vs confirmatory；
3. 保留 OOS；
4. 对大量候选考虑 FDR / multiple-testing awareness；
5. 不只报告最好看的结果。

---

# 20. Step 6.17：稳健性分析

正式保留下来的 Alpha 因子至少要检查：

- 不同持有期；
- 不同 rebalance；
- 不同网络窗口；
- 不同网络密度；
- 沪深300 / 中证500 / 中证1000 / 全市场；
- 大小盘；
- 行业；
- 排除 ST；
- 流动性筛选；
- 交易成本；
- Raw vs Neutralized；
- Main vs Benchmark。

---

# 21. Stage 6 的核心因子分级

建议不要把所有网络特征都叫“Alpha”。

可分三层：

## Level A：Network Descriptors

如：

- Degree；
- PageRank；
- Community。

只有描述意义。

## Level B：Candidate Alpha Factors

有稳定：

- IC；
- 分层单调性；

但尚未完成成本/OOS。

## Level C：Validated Network Alpha

至少满足：

- OOS IC 稳定；
- 分层合理；
- 中性化后仍有信息；
- 成本后仍有价值；
- 容量可接受；
- benchmark comparison 通过。

只有 Level C 才适合称为：

\[
\boxed{\text{正式 Network Alpha Factor}}
\]

---

# 22. Stage 6 建议输出文件

```text
stage6_network_alpha/
├── alpha_research_protocol.md
├── network_alpha_features.parquet
├── factor_definition_table.csv
├── factor_missing_summary.csv
├── factor_correlation_matrix.csv
├── factor_barra_exposure_summary.csv
├── ic_daily.csv
├── ic_summary.csv
├── ic_by_year.csv
├── ic_by_universe.csv
├── quantile_portfolio_returns.csv
├── quantile_backtest_summary.csv
├── long_short_returns.csv
├── turnover_summary.csv
├── transaction_cost_sensitivity.csv
├── capacity_summary.csv
├── fama_macbeth_summary.csv
├── main_vs_benchmark_alpha.csv
├── oos_alpha_summary.csv
├── robustness_summary.csv
├── alpha_factor_scorecard.csv
├── validated_alpha_factor_list.csv
└── figures/
```

---

# 23. Stage 6 最终建议生成的图

至少包括：

1. `factor_ic_cumulative.png`
2. `rolling_ic.png`
3. `ic_by_year.png`
4. `quantile_cumulative_returns.png`
5. `quantile_average_return.png`
6. `long_short_cumulative_return.png`
7. `turnover_over_time.png`
8. `cost_sensitivity.png`
9. `factor_barra_exposure.png`
10. `main_vs_benchmark.png`

---

# 24. Stage 6 必须回答的核心问题

## Q1. Degree、PageRank 本身就是 Alpha 吗？

不是。

它们只是网络特征。

只有在：

\[
Feature_{i,t}
\rightarrow
FutureReturn_{i,t+h}
\]

具有稳定样本外关系后，才可能成为 Alpha。

---

## Q2. 高 IC 就说明因子好？

不够。

还要看：

- IC 稳定性；
- 分层单调性；
- turnover；
- cost；
- capacity；
- OOS；
- risk exposure。

高 IC 但高换手、低容量的因子未必有实际价值。

---

## Q3. 为什么一定要做中性化？

因为网络指标很容易与：

- Size；
- Liquidity；
- Industry；
- Momentum；

高度相关。

如果不控制这些暴露，就无法判断：

\[
\boxed{
\text{收益来自网络信息，还是传统因子。}
}
\]

---

## Q4. Neighbor Momentum 与普通 Momentum 有什么区别？

普通 Momentum：

\[
Mom_{i,t}
\]

只使用股票自己的历史收益。

Neighbor Momentum：

\[
NM_{i,t}
\]

使用网络邻居的历史收益。

核心研究问题是：

\[
\boxed{
\text{邻居的信息是否对股票自身未来收益具有增量预测力？}
}
\]

因此应控制：

\[
Mom_{i,t}
\]

再看：

\[
NM_{i,t}
\]

是否仍有效。

---

## Q5. Network Residual 应预测 reversal 还是 continuation？

没有理论上必须固定的方向。

如果网络代表短期共同定价关系，偏离可能回归：

\[
NetResidual
\rightarrow reversal.
\]

如果偏离代表信息逐步扩散，也可能 continuation。

应由实证决定，不要预设符号。

---

## Q6. 是否所有网络因子都必须在 Main Network 和 Benchmark 上计算？

核心因子最好如此。

这样才能判断：

\[
\boxed{
\text{主网络是否真正带来增量 Alpha。}
}
\]

---

## Q7. IC 用 Pearson 还是 Spearman？

建议：

\[
\boxed{\text{Rank IC / Spearman 为主}}
\]

因为横截面因子研究更关心排序能力，对极端值也更稳健。

Pearson IC 可作为补充。

---

## Q8. 为什么不能只看全样本 IC？

因为全样本平均值可能由少数年份驱动。

必须检查：

- yearly IC；
- rolling IC；
- OOS IC；
- 不同股票池。

---

## Q9. 为什么要做交易成本？

网络特征可能随网络重构频繁变化。

如果：

\[
Turnover
\]

很高，则纸面 Alpha 可能在真实交易中消失。

---

## Q10. 为什么要做容量？

M1 的交付明确要求容量分析。

如果 Alpha 主要出现在：

- 微盘股；
- 极低成交额股票；

那么即使 IC 很高，也不一定有实际资金承载能力。

---

## Q11. 是否一定要做 Fama-MacBeth？

不是绝对必须，但非常推荐。

因为它能帮助判断：

\[
NetworkFactor
\]

在控制传统横截面因子后是否还有边际预测能力。

---

## Q12. 是否需要一开始就做机器学习模型？

不建议。

Stage 6 第一轮应优先完成：

\[
\boxed{
\text{单因子}
+
\text{传统横截面方法}
+
\text{OOS}
}
\]

等基线。

只有明确网络因子确实有信息后，再考虑：

- nonlinear combination；
- boosting；
- neural network。

否则机器学习会增加 data mining 风险。

---

## Q13. Stage 6 是否应该设计大量复合因子？

第一版不建议。

先验证最有经济含义的：

- Strength；
- Cross Connectivity；
- Neighbor Momentum；
- Network Residual；
- Lead-Lag Signal。

之后再组合。

---

## Q14. Long-Short 在 A 股不易直接执行，为什么还做？

因为 Long-Short spread 是横截面因子研究中用于判断排序能力和因子收益的重要统计工具。

真正可执行策略可以另外做：

\[
\boxed{\text{Long-only enhancement}}
\]

例如 benchmark 内 overweight 高分股票、underweight 低分股票。

---

## Q15. 什么时候才能说“网络产生了 Alpha”？

至少需要看到：

1. IC/RankIC 非偶然；
2. 时间上相对稳定；
3. 分层收益近似单调；
4. 中性化后仍有信息；
5. OOS 仍成立；
6. 交易成本后仍有价值；
7. 容量不过度受限；
8. 相比 benchmark network 有新增信息。

---

# 25. Stage 6 完成判定标准

只有以下任务完成后，Stage 6 才视为完成：

- [ ] 第一版网络 Alpha 因子库已经冻结；
- [ ] 所有因子严格 Point-in-Time；
- [ ] Tradable Universe 已正确应用；
- [ ] 因子缺失、异常、标准化规则统一；
- [ ] Raw / Industry-neutral / Barra-neutral 已比较；
- [ ] 已完成 1/5/20 日或预设 horizon 的 IC；
- [ ] 已完成 yearly / rolling IC；
- [ ] 已完成分层回测；
- [ ] 已完成 Long-Short 或 Long-only 评价；
- [ ] 已完成 turnover；
- [ ] 已完成交易成本敏感性；
- [ ] 已完成容量分析；
- [ ] 已检查 Barra 风险暴露；
- [ ] 已完成 Main vs Benchmark；
- [ ] 已完成 OOS / walk-forward；
- [ ] 已完成稳健性；
- [ ] 已形成 Alpha factor scorecard；
- [ ] 已明确哪些指标仅是描述性网络特征，哪些是真正候选 Alpha，哪些通过正式验证。

---

# 26. Stage 6 最终因子评分表建议

`alpha_factor_scorecard.csv`

| Factor | Mean RankIC | ICIR | OOS IC | Q5-Q1 | Net Return | Turnover | Capacity | Barra-neutral | Benchmark Gain | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Weighted Degree | | | | | | | | | | |
| Cross Degree | | | | | | | | | | |
| PageRank | | | | | | | | | | |
| Neighbor Momentum | | | | | | | | | | |
| Network Residual | | | | | | | | | | |
| Lead-Lag Signal | | | | | | | | | | |

最终 `Decision` 可分：

- Reject；
- Exploratory；
- Candidate；
- Validated。

---

# 27. Stage 6 最终汇报模板

> **Stage 6 Network Alpha Research Conclusion**
>
> 1. **Tested Network(s)**：________  
> 2. **Candidate Alpha Factors**：________  
> 3. **Best Raw RankIC**：________  
> 4. **Best Barra-Neutral RankIC**：________  
> 5. **OOS Performance**：________  
> 6. **Quantile Monotonicity**：________  
> 7. **Long-Short / Long-only Result**：________  
> 8. **Turnover**：________  
> 9. **Transaction-Cost Robustness**：________  
> 10. **Capacity**：________  
> 11. **Main Risk Exposures**：________  
> 12. **Main vs Benchmark Improvement**：________  
> 13. **Validated Alpha Factors**：________  
> 14. **Rejected Factors and Reasons**：________  
> 15. **Decision for Further Research**：________

---

# 28. Stage 6 的一句话目标

\[
\boxed{
\textbf{
把全市场股票网络转化为股票层面的候选 Alpha，并通过 IC、分层回测、中性化、换手率、交易成本、容量和样本外检验，判断网络是否真正提供传统因子之外的可投资新增信息。
}
\]
