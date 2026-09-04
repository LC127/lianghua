# M1 Stage 2：A股全市场数据质量检查与可用性验收

## 1. Stage 2 的定位

Stage 2 的目标是对 Stage 1 已定义的 A 股全市场研究对象进行正式的数据质量验收，判断现有数据是否足以支撑后续：

\[
\text{市场结构诊断}
\rightarrow
\text{网络方法选型}
\rightarrow
\text{全市场网络构建}
\rightarrow
\text{Alpha/Risk 因子研究}.
\]

本阶段的核心不是“清洗出一份看起来干净的数据”，而是建立一个**可复现、可追溯、可量化的数据质量结论**。

Stage 2 不负责：

- 选择最终网络方法；
- 确定最佳滚动窗口；
- 确定最终稀疏化阈值；
- 评价 Alpha 因子是否有效；
- 评价 Risk 因子是否有预测能力。

这些任务应在后续 Stage 3 及以后完成。

---

# 2. Stage 2 需要回答的核心问题

Stage 2 完成后，应能够回答以下问题。

## Q1. 历史股票池是否完整？

需要确认：

- 当前上市股票是否完整；
- 历史退市股票是否保留；
- 上市日期、退市日期是否合理；
- 是否存在重复证券记录；
- 是否存在代码变更导致的重复节点；
- 沪深与北交所的数据起点是否清楚；
- 非普通 A 股证券是否被误纳入。

最终要回答：

> 是否可以可靠构造任意历史交易日 \(t\) 的 Point-in-Time Raw Universe？

---

## Q2. 每个交易日的市场覆盖是否完整？

对每个交易日统计：

\[
N_t^{listed},
\quad
N_t^{traded},
\quad
N_t^{suspended},
\quad
N_t^{ST},
\quad
N_t^{valid}.
\]

并定义：

\[
Coverage_t
=
\frac{N_t^{valid}}
{N_t^{listed}}.
\]

需要回答：

- 是否存在异常日期导致市场覆盖率突然下降；
- 是否存在整段历史数据缺失；
- 沪深、北交所是否存在不同数据起点；
- 日频数据是否满足后续滚动网络研究。

---

## Q3. 行情字段是否满足基本逻辑约束？

对 OHLCV 数据检查：

\[
High_{i,t}\ge \max(Open_{i,t},Close_{i,t}),
\]

\[
Low_{i,t}\le \min(Open_{i,t},Close_{i,t}),
\]

\[
Volume_{i,t}\ge0,
\qquad
Amount_{i,t}\ge0.
\]

需要识别：

- 负价格；
- 负成交量；
- 负成交额；
- High/Low 与 Open/Close 冲突；
- 重复日期；
- 同一股票同一交易日重复记录；
- 明显错误价格。

---

## Q4. 复权数据是否可靠？

需要检查：

- 复权因子是否缺失；
- 复权因子是否非正；
- 是否存在异常跳变；
- 前复权/后复权口径是否一致；
- 除权除息附近是否产生错误极端收益。

建议基于统一复权价格构造：

\[
r_{i,t}
=
\log
\frac{P_{i,t}^{adj}}
{P_{i,t-1}^{adj}}.
\]

需要回答：

> 后续网络分析能否使用同一套稳定的日收益口径？

---

## Q5. 停牌是否被正确识别？

需要区分：

\[
\text{真实零收益}
\neq
\text{停牌导致无交易}.
\]

检查：

- 停牌状态与行情缺失是否一致；
- 是否存在停牌日被错误填成 0 收益；
- 复牌首日是否存在异常缺失；
- 长期停牌股票如何识别。

最终需要生成：

- `is_suspended`
- `valid_return_flag`

---

## Q6. ST/*ST 状态是否为历史 Point-in-Time？

需要确认：

\[
ST_{i,t}
\]

是逐日历史状态，而不是当前状态回填历史。

检查：

- ST 起止日期；
- *ST 与普通 ST 的编码方式；
- 历史状态是否连续；
- 状态变化日是否合理。

需要回答：

> 网络估计与 Alpha 回测能否在历史时点使用真实 ST 状态？

---

## Q7. 行业分类是否为历史 Point-in-Time？

至少确认：

\[
Industry_{i,t}
\]

是否随时间变化。

检查：

- 行业字段覆盖率；
- 是否只提供当前行业；
- 是否存在行业版本变更；
- 股票行业发生变化时是否有历史记录；
- 申万一级/二级/三级行业是否可用。

这一问题非常关键，因为后续要研究：

\[
Same\text{-}industry
\quad vs\quad
Cross\text{-}industry.
\]

如果只有当前行业，应在报告中明确标记为限制，不能默认其为历史真实分类。

---

## Q8. Barra 数据是否可用于历史残差化？

需要检查：

- Barra 暴露覆盖年份；
- 每日/每月/其他频率；
- 缺失率；
- 是否为 Point-in-Time；
- 风格因子暴露是否稳定；
- 行业因子是否与行业分类口径一致；
- 是否存在极端暴露值。

需要回答：

> 是否可以构造 factor-adjusted residual return，用于后续 residual association network？

---

## Q9. 市值、换手率、成交额等 Daily Basic 数据是否完整？

至少检查：

- market_cap；
- float_market_cap；
- turnover；
- amount；
- PE；
- PB。

需要统计：

\[
MissingRate_{field,t}
\]

和：

\[
MissingRate_{field,i}.
\]

重点回答：

- 后续是否可以做流动性筛选；
- 是否可以做 Size/Turnover neutralization；
- 是否可以进行容量分析；
- 是否存在大面积历史缺失。

---

## Q10. 是否存在极端异常收益？

对复权日收益：

\[
r_{i,t}
\]

统计：

- mean；
- std；
- quantiles；
- min/max；
- 极端分位数。

建议至少标记：

\[
|r_{i,t}| > q_{0.999}
\]

或使用稳健方法：

\[
|r_{i,t}-median(r)| > k \cdot MAD.
\]

注意：

> 极端收益不能自动删除。

应进一步判断：

- 是否真实涨跌停；
- 是否公司事件；
- 是否复权错误；
- 是否代码映射错误。

Stage 2 的目标是**识别与分类**，而不是盲目 winsorize。

---

## Q11. 是否存在异常缺失结构？

需要区分：

1. 新股导致的自然缺失；
2. 退市后的自然缺失；
3. 停牌导致的缺失；
4. 数据源错误导致的缺失。

应分析：

\[
MissingPattern_{i,t}.
\]

不能把所有 NA 统一解释为“数据缺失”。

---

## Q12. 不同时间窗口下，有多少股票能够参与网络估计？

对：

\[
W\in\{60,120,252\}
\]

统计每个交易日满足：

\[
\frac{N_{obs,i,t}(W)}{W}
\ge c_{obs}
\]

的股票数量。

建议测试：

\[
c_{obs}\in\{0.8,0.9,0.95\}.
\]

输出：

\[
N_t^{network}(W,c_{obs}).
\]

这一步非常重要，因为它会决定后续：

\[
p_t
\]

到底有多大，并直接影响网络方法选择。

---

# 3. Stage 2 的具体执行步骤

## Step 2.1 读取并统一原始数据

建议统一以下数据表：

- 股票主表；
- 交易日历；
- 日频行情；
- 复权因子；
- Daily Basic；
- ST 状态；
- 停牌状态；
- 行业；
- Barra 暴露。

统一：

- 股票代码格式；
- 日期格式；
- 交易所标识；
- 主体 ID；
- 缺失值编码。

---

## Step 2.2 检查证券主表

输出：

`stock_master_quality_summary.csv`

至少包括：

- 当前上市数量；
- 历史退市数量；
- 上交所数量；
- 深交所数量；
- 北交所数量；
- 主板/科创板/创业板/北交所数量；
- 重复代码数；
- 重复主体数；
- list_date 缺失数；
- delist_date 异常数；
- 证券类型异常数。

---

## Step 2.3 检查交易日覆盖

输出：

`daily_market_coverage.csv`

字段建议：

| 字段 | 含义 |
|---|---|
| trade_date | 交易日 |
| listed_count | 当日上市股票数 |
| traded_count | 当日有行情股票数 |
| suspended_count | 当日停牌数 |
| st_count | 当日 ST 数 |
| valid_price_count | 有效价格数 |
| coverage_ratio | 行情覆盖率 |

并绘制：

- `daily_listed_count.png`
- `daily_coverage_ratio.png`

---

## Step 2.4 检查字段缺失率

输出：

`field_missing_summary.csv`

至少统计：

- open
- high
- low
- close
- volume
- amount
- adj_factor
- market_cap
- turnover
- PE
- PB
- industry
- ST
- suspension
- Barra

建议同时输出：

- 全样本缺失率；
- 年度缺失率；
- 股票层面缺失率；
- 日期层面缺失率。

---

## Step 2.5 检查 OHLCV 异常

输出：

`ohlcv_anomalies.csv`

至少包含：

- `high_lt_open`
- `high_lt_close`
- `low_gt_open`
- `low_gt_close`
- `negative_price`
- `negative_volume`
- `negative_amount`
- `duplicate_record`

---

## Step 2.6 检查复权与收益

输出：

- `adj_factor_anomalies.csv`
- `return_outliers.csv`
- `return_distribution_summary.csv`

建议绘制：

- 全市场收益分布；
- 年度收益尾部分布；
- 极端收益数量随时间变化。

---

## Step 2.7 检查 ST、停牌与可交易状态

输出：

`stock_status_quality_summary.csv`

至少统计：

- 每日 ST 数量；
- 每日停牌数量；
- ST 状态缺失；
- 停牌状态缺失；
- 停牌日仍有正常成交的异常记录；
- 非停牌日无行情的异常记录。

---

## Step 2.8 检查行业历史覆盖

输出：

`industry_coverage_summary.csv`

至少包括：

- 行业缺失率；
- 一级行业数量；
- 每个行业股票数；
- 历史行业变更记录数；
- 是否存在仅当前行业而无历史行业的问题。

---

## Step 2.9 检查 Barra 覆盖

输出：

`barra_coverage_summary.csv`

至少包括：

- 可用起始日期；
- 可用结束日期；
- 各风格因子缺失率；
- 每日有 Barra 暴露的股票数；
- 与当日 Network Universe 的覆盖比例；
- 极端暴露值统计。

---

## Step 2.10 计算不同窗口下的可用股票数

对：

\[
W=60,120,252
\]

和：

\[
c_{obs}=0.8,0.9,0.95
\]

生成：

`network_universe_feasibility.csv`

字段建议：

| trade_date | window | valid_ratio | eligible_count |
|---|---:|---:|---:|

并计算：

\[
p_t/W.
\]

这会成为 Stage 3 判断全维精度矩阵是否可行的重要输入。

---

# 4. Stage 2 最终应生成的核心数据文件

建议目录：

```text
stage2_data_quality/
├── stock_master_quality_summary.csv
├── daily_market_coverage.csv
├── field_missing_summary.csv
├── ohlcv_anomalies.csv
├── adj_factor_anomalies.csv
├── return_outliers.csv
├── return_distribution_summary.csv
├── stock_status_quality_summary.csv
├── industry_coverage_summary.csv
├── barra_coverage_summary.csv
├── network_universe_feasibility.csv
└── figures/
```

---

# 5. Stage 2 最终应形成的结论

Stage 2 最终必须明确给出以下结论，而不是只输出 CSV。

## 结论 A：数据是否总体可用？

明确分为：

- **Pass**：可直接进入 Stage 3；
- **Conditional Pass**：存在局部问题，但通过明确规则处理后可进入 Stage 3；
- **Fail**：存在系统性缺失或 PIT 问题，必须补数据后再继续。

---

## 结论 B：主研究起始日期应从何时开始？

不能简单因为数据“从 2010 年开始”就直接使用 2010 年。

应根据：

- 行情覆盖；
- 行业覆盖；
- Barra 覆盖；
- Daily Basic 覆盖；
- 网络窗口数据需求；

综合确定最终主研究起点。

例如：

\[
T_0
=
\max(
T_{price},
T_{industry},
T_{Barra},
T_{basic}
)
\]

再考虑网络窗口 \(W\)。

---

## 结论 C：第一版是否使用北交所？

根据：

- 样本历史长度；
- 数据完整性；
- 流动性；
- 与沪深口径一致性；

给出：

- 主样本是否排除北交所；
- 北交所是否作为 robustness extension。

---

## 结论 D：Network Universe 的最低观测要求是多少？

根据：

\[
W=60,120,252
\]

与：

\[
c_{obs}=0.8,0.9,0.95
\]

的实际股票覆盖情况，确定合理规则。

不能仅凭经验决定。

---

## 结论 E：哪些字段需要修复或补充？

必须形成明确问题清单：

| 问题 | 严重程度 | 影响 | 处理方案 |
|---|---|---|---|
| 历史行业缺失 | High | Same/Cross 分析 | 补历史行业或限制解释 |
| Barra 起点较晚 | Medium | residual network | 调整主研究期 |
| 个别复权异常 | Low | return | 局部修复 |
| ... | ... | ... | ... |

---

# 6. Stage 2 的完成判定标准

Stage 2 只有满足以下条件才视为完成：

- [ ] 股票主表通过基本一致性检查；
- [ ] 历史上市/退市信息可用于 PIT Universe；
- [ ] 每日市场覆盖率已量化；
- [ ] 所有核心行情字段缺失率已统计；
- [ ] OHLCV 逻辑错误已识别；
- [ ] 复权异常和收益异常已识别；
- [ ] ST 与停牌状态已验证；
- [ ] 行业历史覆盖已确认；
- [ ] Barra 历史覆盖已确认；
- [ ] Daily Basic 覆盖已确认；
- [ ] 不同窗口下 Network Universe 可用股票数已计算；
- [ ] 已形成 Data Issue Log；
- [ ] 已给出 Pass / Conditional Pass / Fail 结论；
- [ ] 已确定 Stage 3 可使用的数据范围。

---

# 7. Stage 2 不应做的事情

本阶段不要：

- 因为某个相关图好看就选择 Pearson；
- 因为已有经验就直接使用 Rolling GLasso；
- 提前确定最佳 \(W\)；
- 做 IC/RankIC；
- 做 Alpha 分层回测；
- 研究 Network Novelty；
- 进行 Regime Detection。

Stage 2 只负责：

\[
\boxed{
\text{确认“数据是否可信、可用、可进入正式研究”}
}
\]

---

# 8. Stage 2 与 Stage 3 的衔接

Stage 2 完成后，Stage 3 才进行：

\[
\boxed{
\text{A股全市场数据结构诊断}
}
\]

重点分析：

- \(p_t/W\)；
- 原始收益相关分布；
- Same vs Cross Industry；
- 相关矩阵谱结构；
- 市场共同因子强度；
- Raw Return vs Barra Residual；
- 市值/流动性异质性；
- 不同步交易与缺失结构。

Stage 3 的结果才用于决定：

\[
\boxed{
\text{最终候选网络应保留哪些方法}
}
\]

---

# 9. Stage 2 最终研究结论模板

完成实际数据检查后，建议最终报告按照下面格式填写：

> **Stage 2 Data Quality Conclusion**
>
> 1. **Overall Status**：Pass / Conditional Pass / Fail  
> 2. **Main Research Universe**：沪深 / 沪深北  
> 3. **Usable Sample Period**：YYYY-MM-DD 至 YYYY-MM-DD  
> 4. **Daily Market Coverage**：中位数 / 最低值 / 异常日期  
> 5. **Price Data Quality**：核心问题与处理  
> 6. **Adjustment Quality**：核心问题与处理  
> 7. **ST/Suspension PIT Quality**：是否满足要求  
> 8. **Industry PIT Quality**：是否满足要求  
> 9. **Barra PIT Quality**：是否满足要求  
> 10. **Network Universe Feasibility**：W=60/120/252 下可用股票数  
> 11. **Remaining Data Issues**：需补充或修复的数据  
> 12. **Decision for Stage 3**：允许进入 / 条件允许进入 / 暂停进入

---

# 10. Stage 2 的一句话目标

\[
\boxed{
\textbf{
用可量化的数据质量证据，确认 A 股全市场数据是否足以支撑后续网络与因子研究，并明确可用样本范围、股票池规则及剩余数据问题。
}
}
