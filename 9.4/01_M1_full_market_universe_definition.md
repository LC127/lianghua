# M1 Stage 1：A股全市场研究股票池定义

## 1. 任务目的

Stage 1 的目标不是构建股票网络，而是为后续数据质量检查、网络估计、Alpha 因子回测与风险研究统一定义研究对象，形成可重复、可追溯、可参数化的股票池规则。

本阶段完成后，应能够对任意交易日 t 明确回答：

1. 哪些证券属于当日 A 股原始市场股票池；
2. 哪些股票可进入网络估计；
3. 哪些股票可进入 Alpha 回测；
4. 各类排除规则的原因是什么；
5. 后续所有研究代码如何基于同一股票池口径运行。

本阶段不负责确定最终网络模型，也不负责判断 Alpha/Risk 因子是否有效。

---

## 2. 三层股票池

建议定义：

U_raw(t) ⊇ U_network(t) ⊇ U_trade(t)

### 2.1 Raw Universe

历史时点实际上市的 A 股普通股票：

U_raw(t) = {i: ListDate_i <= t < DelistDate_i}。

第一版主研究建议使用沪深 A 股普通股票，包括：

- 上交所主板 A 股；
- 科创板；
- 深交所主板 A 股；
- 创业板。

北交所数据保留，但第一版作为扩展样本或稳健性检验，不强制纳入主结果。

Raw Universe 应保留历史退市股票在其实际上市期间的数据，以避免幸存者偏差。

不包含：

- B 股；
- ETF/基金；
- 指数；
- 债券、可转债；
- 其他非普通 A 股证券。

### 2.2 Network Universe

定义：

U_network(t) = {i in U_raw(t): Q_data(i,t)=1}。

Q_data(i,t) 表示股票 i 在时点 t 满足网络估计所需的数据质量要求。

建议参数化以下条件：

1. 窗口内最低有效观测比例：
   N_obs(i,t;W)/W >= c_obs，
   初始可测试 c_obs in {0.8, 0.9}，最终由数据诊断确定。
2. 上市历史足够支撑网络窗口，不机械固定为 252 天，而与 W 联动。
3. 长期停牌或有效交易比例过低的股票不进入当期网络。
4. 存在严重价格、复权、成交量或证券映射异常的股票暂时剔除，并记录原因。

网络研究层面原则上不自动剔除 ST/*ST 股票，因为 ST 可能携带风险传播信息；但必须保留历史 is_st 状态，以便后续做含/不含 ST 的稳健性比较。

### 2.3 Tradable Alpha Universe

定义：

U_trade(t) = {i in U_network(t): Q_trade(i,t)=1}。

用于 IC、Rank IC、分层、多空组合、换手率和容量分析。

建议初始考虑：

- 非 ST/*ST；
- 当日非停牌；
- 上市时间足够；
- 历史数据足够；
- 流动性满足最低要求；
- 调仓时不存在明显不可成交状态。

流动性阈值不要在 Stage 1 主观定死，应在后续根据成交额/换手率分布确定。

---

## 3. Point-in-Time 原则

所有股票池判断必须遵守 Point-in-Time 原则。

交易日 t 只能使用 t 当时已经可观测的信息，重点包括：

- 上市/退市状态；
- ST/*ST 状态；
- 停复牌状态；
- 历史行业分类；
- Barra 风格暴露；
- 指数成分；
- 证券简称和代码变更。

禁止使用当前状态回填历史。

---

## 4. 北交所处理

第一版建议：

- 主样本：沪深 A 股；
- 扩展样本：沪深北 A 股。

原因是北交所历史较短，流动性、市场制度及数据起点与沪深不同。数据层应完整保留北交所，但不强制第一版主网络与沪深完全同口径。

---

## 5. 代码变更与证券主体映射

建议区分稳定证券主体 ID 与时点证券代码，建立：

- security_id；
- ts_code；
- valid_from；
- valid_to。

网络节点与因子研究优先使用稳定证券主体标识，展示层再映射为当期代码/简称。

---

## 6. Stage 1 最低输入字段

证券主表至少包括：

- security_id / ts_code
- stock_name
- exchange
- board / market
- security_type
- list_date
- delist_date
- list_status

历史状态至少包括：

- trade_date
- is_st
- is_suspended
- industry_l1
- industry_l2（如有）
- industry_l3（如有）

后续 Network Universe 还需要：

- adjusted_close / return
- trading_flag
- valid_observation_flag

后续 Tradable Universe 还需要：

- amount
- turnover
- market_cap
- limit_up / limit_down（如有）

---

## 7. 建议后续生成的股票池状态表

文件：

stock_universe_daily.parquet

推荐字段：

| 字段 | 含义 |
|---|---|
| trade_date | 交易日 |
| security_id | 稳定证券主体 ID |
| ts_code | 当期股票代码 |
| exchange | 交易所 |
| board | 板块 |
| is_listed | 当日是否上市 |
| is_st | 当日是否 ST |
| is_suspended | 当日是否停牌 |
| listed_trading_days | 已上市交易日数 |
| valid_obs_60 | 近60日有效观测数 |
| valid_obs_120 | 近120日有效观测数 |
| valid_obs_252 | 近252日有效观测数 |
| in_raw_universe | 是否进入 Raw Universe |
| in_network_universe | 是否进入 Network Universe |
| in_trade_universe | 是否进入 Tradable Universe |
| exclusion_reason | 排除原因 |

---

## 8. 参数化配置建议

建议将筛选规则写入配置文件，而不是散落在代码中：

```yaml
main_market:
  include_sse: true
  include_szse: true
  include_bse: false

network_universe:
  min_valid_obs_ratio: 0.8
  min_listed_trading_days: null
  exclude_st: false
  exclude_long_suspension: true

trade_universe:
  exclude_st: true
  exclude_suspended: true
  min_valid_obs_ratio: 0.8
  liquidity_filter:
    enabled: true
    method: pending_data_diagnosis
```

对尚未由实际数据确定的阈值，使用 `pending_data_diagnosis` 标记，不应在 Stage 1 提前固定。

---

## 9. Stage 1 验收标准

只有以下项目全部明确，Stage 1 才视为完成：

- [ ] 明确 Raw / Network / Tradable 三层股票池；
- [ ] 明确沪深与北交所第一版研究范围；
- [ ] 明确历史退市股票保留原则；
- [ ] 明确 ST 股票在网络研究与 Alpha 回测中的不同处理；
- [ ] 明确停牌、新股、低流动性股票的处理逻辑；
- [ ] 明确 Point-in-Time 原则；
- [ ] 明确代码变更与证券主体映射要求；
- [ ] 所有筛选规则参数化；
- [ ] 不提前固定需由数据分布决定的最终阈值；
- [ ] 明确后续 stock_universe_daily 的字段结构。

---

## 10. Stage 1 不应提前回答的问题

本阶段不应决定：

- 最终使用 Pearson、GLasso、Lead-lag 还是其他网络；
- 最佳滚动窗口；
- 最佳稀疏化阈值；
- 是否一定采用 Barra residualization；
- 哪个网络 Alpha 因子有效；
- 哪个风险指标有预测能力。

这些问题应由后续数据质量检查、市场结构诊断和候选网络比较决定。

---

## 11. 与后续阶段衔接

完成本阶段后进入：

Stage 1 Universe Definition
→ Stage 2 Data Quality Check
→ Stage 3 Market Structure Diagnosis

Stage 2 再用实际数据统计：

- 每日上市股票数量；
- 每日有效交易股票数量；
- 缺失率；
- ST/停牌比例；
- 历史行业覆盖；
- Barra 数据覆盖；
- 复权与收益异常。

只有通过数据质量验收后，才进入基于 A 股实际数据特征的网络方法选择。
