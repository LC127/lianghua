# M1 A股全市场股票关联网络：Temporal Stability and Secular-Confounding Validation

**日期：2026-09-28**  
**项目：M1 — A股全市场股票关联网络研究**  
**研究主题：网络风险信息的时间稳定性与实施容量的长期趋势混杂检验**

---

## 1. 今日研究目标

Day 9 已经发现：

\[
\text{Network State}
\rightarrow
\{\text{Conditional Alpha},\ \text{Conditional IVOL Information},\ \text{Capacity}\},
\]

但同时出现一个重要现象：

\[
Capacity_{HIGH\ NetworkStress}
>
Capacity_{LOW\ NetworkStress}.
\]

由于 HIGH Network Stress 更可能集中于样本后期，而 A 股市场的股票数量、成交活跃度与整体流动性环境也具有明显长期变化，因此今天的核心任务不是继续增加新的网络因子或新的 regime，而是回答两个识别问题：

\[
\boxed{
Q_1:\ \text{Day 9 的 Network Stress--Capacity 正向关系是否主要来自长期时期构成？}
}
\]

\[
\boxed{
Q_2:\ \text{网络特征对 future idiosyncratic volatility 的信息是否具有跨时期稳定性？}
}
\]

因此 Day 10 定义为：

\[
\boxed{\textbf{Temporal Stability and Secular-Confounding Validation}}
\]

今天的研究流程为：

1. Calendar-Time / Market-Liquidity Diagnostic  
2. Time-Adjusted Network-Stress Capacity Validation  
3. Within-Year Capacity Validation  
4. Future-IVOL Temporal Stability  
5. Frozen 9-Factor Temporal Stability Matrix  
6. Temporal Robustness Summary（已完成设计，待最终运行汇总）

---

# 2. Step 1 — Calendar-Time / Market-Liquidity Diagnostic

## 2.1 研究目的

首先诊断 Day 9 的：

\[
Capacity_{HIGH}>Capacity_{LOW}
\]

是否存在明显的 secular-liquidity / calendar-time confounding。

严格复用已冻结的：

- PIT `ADV20`
- Network Stress regime
- rebalance-level pair capacity

不重新估计 ADV、Capacity 或 Regime。

核心市场流动性指标定义为：

\[
MarketADV20_t
=
\sum_{i\in\mathcal U_t}
ADV20_{i,t},
\]

用于衡量时点 \(t\) 的全市场总体交易流动性环境。

同时保留：

\[
MedianADV20_t
=
Median_i(ADV20_{i,t}),
\]

以区分“股票数量扩张”与“典型个股流动性变化”。

---

## 2.2 QA

Step 1 主要 QA：

- analysis dates = **138**
- panel rows = **402**
- duplicate date-window = **0**
- market-liquidity missing dates = **0**
- negative Market ADV20 = **0**
- capacity missing rows = **6**
- ADV reestimated = **False**
- Capacity reestimated = **False**
- Regime reestimated = **False**
- future information used = **False**
- `all_formal_qa_pass = True`

其中 6 个 capacity missing rows 均位于早期 PIT regime 尚未形成的 burn-in 区间，不进入后续正式 regime comparison，不做 0 填充。

---

## 2.3 关键诊断结果

### Capacity 与 calendar time

\[
Corr(Capacity,time):
\]

- W60：**0.690**
- W120：**0.737**
- W252：**0.805**

说明：

\[
\boxed{
Capacity_t
\text{存在非常明显的长期时间趋势。}
}
\]

---

### Capacity 与 Market ADV20

\[
Corr(Capacity,MarketADV20):
\]

- W60：**0.877**
- W120：**0.884**
- W252：**0.901**

因此：

\[
\boxed{
\text{实施容量与全市场流动性环境高度相关。}
}
\]

这一结果符合 Capacity 本身基于 ADV execution constraint 构造的经济含义。

---

### Network Stress 与 calendar time

\[
Corr(NetworkStress,time):
\]

- W60：**0.254**
- W120：**0.417**
- W252：**0.554**

尤其 W252 表现出明显时间依赖。

---

### Network Stress 与 Market ADV20

\[
Corr(NetworkStress,MarketADV20):
\]

- W60：**0.067**
- W120：**0.120**
- W252：**0.254**

因此 Network Stress 本身并不是 Market ADV 的简单代理，但它与 calendar period 存在明显构成差异。

---

## 2.4 Step 1 结论

Step 1 表明：

\[
\boxed{
\text{Day 9 的 HIGH--LOW Capacity 差异存在显著 secular-composition concern。}
}
\]

但不能据此直接认为 Day 9 的结果错误。

更准确地说：

> Network Stress 与 Capacity 的无条件关系可能部分来自不同 network regime 所处的历史时期和市场流动性环境差异，因此必须进一步做时间调整。

---

# 3. Step 2 — Time-Adjusted Network-Stress Capacity Validation

## 3.1 研究设计

以：

\[
\widetilde A_{t,W}^{cap}
=
Median_f(A^{cap}_{t,f,W})
\]

作为 date-window 层面的 Capacity，以避免将同一 Network Stress observation 因 9 个 factor 重复使用而造成 pseudo-replication。

固定五个 nested models：

\[
M0:\ Regime
\]

\[
M1:\ Regime+\log MarketADV20
\]

\[
M2:\ Regime+YearFE
\]

\[
\boxed{
M3:\ Regime+\log MarketADV20+YearFE
}
\]

以及 robustness：

\[
M4:\ Regime+\log MedianADV20+YearFE.
\]

LOW 为 reference，核心参数：

\[
\beta_H
=
Capacity_{HIGH}-Capacity_{LOW}.
\]

主规格固定为 M3，不根据结果选择模型。

---

## 3.2 QA

- input rows = **402**
- effective rows = **294**
- common dates = **93**
- expected fits = **30**
- actual fits = **30**
- full-rank fits = **30**
- primary-model fits = **6**
- Capacity / Regime / Liquidity 均未重新估计
- model/window 未根据结果选择
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 3.3 M0：复现 Day 9 的原始正向 Capacity gap

Native：

\[
W60:+4.783\text{亿元},
\]

\[
W120:+6.368\text{亿元},
\]

\[
W252:+8.628\text{亿元}.
\]

原始关系表现为：

\[
\boxed{
HIGH\ NetworkStress
\leftrightarrow
higher\ rebalance\text{-}level\ Capacity.
}
\]

---

## 3.4 M1：单独控制 Market ADV 后仍为正

Native：

- W60：\(+4.856\) 亿元，\(t=3.20\)
- W120：\(+6.115\) 亿元，\(t=3.86\)
- W252：\(+5.372\) 亿元，\(t=3.92\)

因此：

\[
\boxed{
\text{Market ADV alone 不能解释掉 Day 9 的正向关系。}
}
\]

---

## 3.5 M2：加入 Year FE 后关系发生明显变化

Native：

\[
W60:-4.733,
\]

\[
W120:-3.470,
\]

\[
W252:-0.744
\]

亿元。

这说明真正显著影响原始关系的是：

\[
\boxed{\text{calendar-period composition}}
\]

而不仅仅是当期 aggregate market ADV。

---

## 3.6 M3：主规格下正向关系基本消失

主规格：

\[
Capacity
=
Regime
+
\log MarketADV20
+
YearFE.
\]

Native：

| Window | Adjusted HIGH−LOW | HAC t | BH q |
|---|---:|---:|---:|
| W60 | **−2.095亿元** | −1.24 | 0.645 |
| W120 | **−0.929亿元** | −0.62 | 0.718 |
| W252 | **−0.314亿元** | −0.36 | 0.718 |

Common-Date：

| Window | Adjusted HIGH−LOW | HAC t | BH q |
|---|---:|---:|---:|
| W60 | −3.111亿元 | −1.77 | 0.232 |
| W120 | −0.918亿元 | −0.58 | 0.718 |
| W252 | −0.314亿元 | −0.36 | 0.718 |

因此：

\[
\boxed{
\text{原始正向 Capacity gap 在时间与流动性调整后基本消失。}
}
\]

虽然调整后的点估计转为负值，但证据较弱，因此不能进一步声称：

\[
NetworkStress\uparrow
\Rightarrow
Capacity\downarrow.
\]

---

## 3.7 Step 2 结论

最合理的解释为：

> **Day 9 中观察到的 HIGH Network Stress–高 Capacity 无条件关系主要反映不同历史时期的市场环境构成，而不是一个稳定、独立的 Network Stress–Capacity effect。**

---

# 4. Step 3 — Within-Year Capacity Validation

## 4.1 研究目的

Step 2 发现 Year FE 是改变结论的关键，因此 Step 3 不继续增加控制变量，而是直接检查：

\[
\Delta A_{W,y}
=
\overline A^{cap}_{HIGH,W,y}
-
\overline A^{cap}_{LOW,W,y}.
\]

只在同一 calendar year 同时存在 HIGH 与 LOW 的情况下进行直接比较。

本步骤定位为：

\[
\boxed{\text{descriptive within-year identification-support diagnostic}}
\]

不增加年度 HAC 或新的显著性检验。

---

## 4.2 QA

- effective rows = **294**
- common dates = **93**
- duplicate date-window = **0**
- all effective rows use 9 frozen factors = **True**
- annual rows = **50**
- overlap rows = **13**
- summary rows = **6**
- Capacity / Regime 未重新估计
- factor/window/year 未根据结果选择
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 4.3 W60：年内方向较一致

Native overlap years：

\[
2018,\ 2019,\ 2022,\ 2026.
\]

对应 HIGH−LOW：

\[
-0.051,\quad
-0.335,\quad
-1.757,\quad
-12.005
\]

亿元。

因此：

\[
\boxed{4/4<0}.
\]

Common-Date 删除 2018 后：

\[
\boxed{3/3<0}.
\]

年度 gap 中位数：

Native：

\[
-1.046\text{亿元},
\]

Common-Date：

\[
-1.757\text{亿元}.
\]

因此 W60 存在方向一致的负向年内倾向，但幅度受到 2026 partial year 的大幅差异影响。

---

## 4.4 W120：直接年内支持非常有限

只有：

\[
\boxed{1}
\]

个年份同时存在 HIGH 和 LOW：

\[
2019.
\]

该年：

\[
HIGH-LOW
=
+2.880\text{亿元}.
\]

所以：

\[
\boxed{
\text{W120 并没有直接的 within-year negative evidence。}
}
\]

这提醒我们：Step 2 中 Year-FE 回归的负向点估计并不等同于“多数年份内部 HIGH 比 LOW 容量更低”。

---

## 4.5 W252：方向混合

两个 overlap years：

\[
2019:+2.623\text{亿元},
\]

\[
2022:-1.289\text{亿元}.
\]

因此：

\[
1/2>0,\qquad1/2<0.
\]

说明：

\[
\boxed{
\text{W252 不存在稳定的直接年内 Capacity 方向。}
}
\]

---

## 4.6 Step 3 结论

综合来看：

\[
\boxed{
\text{只有 W60 呈现较一致的负向年内倾向，}
}
\]

而：

\[
\boxed{
\text{W120 直接支持极少，W252 方向混合。}
}
\]

因此不能把 Step 2 的调整后负向点估计泛化成：

\[
High\ NetworkStress
\Rightarrow
Lower\ Capacity.
\]

更合理的总结仍是：

\[
\boxed{
\text{不存在跨网络窗口稳定的独立 Network Stress–Capacity relation。}
}
\]

---

# 5. Step 4 — Future-IVOL Temporal Stability

## 5.1 研究目的

离开 Capacity 线，回到 M1 当前最重要的风险结果：

\[
\boxed{
delta\_degree\_percentile
\rightarrow
future\_idio\_vol\_annualized.
}
\]

严格复用冻结的：

- `delta_degree_percentile`
- `future_idio_vol_annualized`
- `FULL_CHARACTERISTIC_NEUTRAL`
- monthly controlled risk rank IC

不重新估计 risk IC。

---

## 5.2 QA

- Native monthly rows = **393**
- Common-Date count = **127**
- Common rows = **381**
- windows = **3**
- duplicate date-window = **0**
- annual result rows = **68**
- summary rows = **6**
- risk IC reestimated = **False**
- factor reestimated = **False**
- factor/window/year 未根据结果选择
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 5.3 W60

Native：

\[
8/12
\]

个年度平均 IC 为正：

\[
PositiveYearShare
=
66.7\%.
\]

Common-Date：

\[
7/11
=
63.6\%.
\]

因此 W60 有正向风险信息，但时间稳定性相对有限。

---

## 5.4 W120

Native：

\[
11/12
\]

个年度平均 IC 为正：

\[
PositiveYearShare
=
91.7\%.
\]

Common-Date：

\[
10/11
=
90.9\%.
\]

2026 的年度 mean IC 约为：

\[
-0.0018,
\]

几乎等于 0，而且 2026 是不完整年度。

因此：

\[
\boxed{
\text{W120 的 delta-degree–future-IVOL relation 具有较强跨时期稳定性。}
}
\]

---

## 5.5 W252

Native / Common-Date：

\[
\boxed{
11/11
}
\]

个年度平均 IC 全部为正：

\[
PositiveYearShare
=
100\%.
\]

平均年度 IC：

\[
0.0600,
\]

年度 IC 中位数：

\[
0.0669,
\]

最低年度平均 IC：

\[
0.0256>0.
\]

月度正 IC 比例约：

\[
93.7\%.
\]

因此：

\[
\boxed{
\text{W252 的 delta-degree–future-IVOL relation 表现出非常明显的 temporal persistence。}
}
\]

---

## 5.6 Step 4 结论

对于 `delta_degree_percentile`：

\[
\boxed{
W252 > W120 > W60
}
\]

这里仅表示其**observed temporal directional stability**，不表示重新进行窗口优选。

---

# 6. Step 5 — Frozen 9-Factor Temporal Stability Matrix

## 6.1 研究目的

为避免只研究表现较好的 `delta_degree_percentile`，将同样的 temporal stability framework 扩展到此前已经冻结的全部 9 个网络因子。

固定：

\[
9\ factors
\times
3\ windows
\times
future\ IVOL.
\]

方向无关的主要稳定性指标定义为：

\[
SignConsistency_{f,W}
=
\frac{
\#\{y:
sign(\overline{IC}_{f,W,y})
=
sign(\overline{IC}_{f,W})
\}
}{
\#\{y\}
}.
\]

没有进行 factor sign flip。

---

## 6.2 QA

- frozen factors = **9**
- windows = **3**
- Native rows = **3529**
- Common dates = **126**
- Common rows = **3402**
- duplicate date-window-factor = **0**
- summary rows = **54**
- matrix rows = **18**
- risk IC / factor 未重新估计
- factor sign flipped = **False**
- factor/window/year 未根据结果选择
- future information used = **False**
- `all_formal_qa_pass = True`

---

## 6.3 高度稳定的网络位置/结构因子

以下三个因子在 Native 与 Common-Date 下，三个窗口全部：

\[
\boxed{
SameSignYearShare=100\%.
}
\]

### `residual_degree_percentile`

全样本 IVOL IC：

\[
W60:-0.0691,
\]

\[
W120:-0.0570,
\]

\[
W252:-0.0489.
\]

所有年度方向均保持负向。

---

### `cross_industry_degree_percentile`

三个窗口同样全部：

\[
SameSignYearShare=1.
\]

全样本关系稳定为负。

---

### `outside_community_degree_percentile`

三个窗口也是：

\[
\boxed{
1.0,\ 1.0,\ 1.0.
}
\]

Native 全样本 IC：

\[
-0.0782,\quad
-0.0769,\quad
-0.0631.
\]

因此：

\[
\boxed{
\text{多个网络位置类变量具有极强跨年份方向稳定性。}
}
\]

---

## 6.4 `delta_degree_percentile`

Native：

\[
0.667,\quad
0.917,\quad
1.000.
\]

Common-Date：

\[
0.636,\quad
0.909,\quad
1.000.
\]

完整复现 Step 4。

---

## 6.5 Neighbor-based variables

`neighbor_jaccard_1m`：

\[
1.00,\quad1.00,\quad0.727.
\]

`neighbor_retention_1m`：

\[
1.00,\quad1.00,\quad0.545.
\]

表明其短中窗口风险关系很稳定，但长期窗口的年度方向稳定性下降。

---

## 6.6 稳定性较弱的变量

### `cross_industry_degree_ratio`

Native：

\[
0.583,\quad0.500,\quad0.455.
\]

其 W252 全样本 IC 本身也接近：

\[
0.
\]

---

### `delta_residual_degree_percentile_1m`

Native：

\[
0.833,\quad0.417,\quad0.545.
\]

### `delta_cross_industry_degree_percentile_1m`

Native：

\[
0.833,\quad0.583,\quad0.545.
\]

因此 short-run change / ratio measures 的方向变化更加明显。

---

## 6.7 全部因子的窗口平均稳定性

Native 9 个因子的平均 sign consistency 大致为：

\[
W60:0.880,
\]

\[
W120:0.824,
\]

\[
W252:0.758.
\]

因此不能将 delta degree 的：

\[
W252>W120>W60
\]

泛化到所有网络因子。

更准确的结论是：

\[
\boxed{
\text{Temporal persistence is broad but structurally heterogeneous across network features.}
}
\]

---

# 7. 今天最重要的综合发现

今天的研究把 M1 的两个维度清楚地区分开了。

## 7.1 Implementation Capacity

Day 9 中：

\[
Capacity_{HIGH}>Capacity_{LOW}.
\]

但 Day 10 发现：

\[
\boxed{
\text{这一正向关系具有明显 calendar-period composition。}
}
\]

仅控制 Market ADV 仍然为正，但加入 Year FE 后原始差异消失；进一步控制 Year + ADV 后，三个窗口的调整后差异均较小、为负且缺乏强统计证据。

因此：

\[
\boxed{
\text{Capacity regime heterogeneity 主要受到长期市场环境影响。}
}
\]

---

## 7.2 Forward Idiosyncratic-Risk Information

与 Capacity 不同，网络风险信息表现出明显的时间稳定性。

`delta_degree_percentile`：

\[
W120:
11/12\ positive\ years,
\]

\[
W252:
11/11\ positive\ years.
\]

而且多个其他 frozen network-position factors：

\[
residual\ degree,
\]

\[
cross\text{-}industry\ degree,
\]

\[
outside\text{-}community\ degree
\]

三个窗口均达到：

\[
100\%\ sign\ consistency.
\]

因此：

\[
\boxed{
\text{网络结构包含跨时期持续的 future-IVOL information。}
}
\]

---

# 8. Day 10 对 Day 9 结论的修正

Day 9 的描述：

> High Network Stress observations are associated with higher rebalance-level capacity.

作为无条件描述仍然正确。

但 Day 10 后必须补充：

> **This positive association is largely attributable to calendar-period composition and is not robust after temporal and market-liquidity adjustment.**

因此不应再把 Network Stress–Capacity 关系作为独立、稳健的经济机制。

另一方面，Day 9 的风险结论得到 Day 10 的进一步支持：

\[
\boxed{
\text{网络风险信息不仅具有 state robustness，也具有 temporal robustness。}
}
\]

---

# 9. 当前 M1 风险主线

经过 Day 6–10，可以形成如下证据链：

\[
\boxed{
\text{Network Structure}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Forward Idiosyncratic-Risk Information}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Robust to conventional characteristic controls}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Robust across network states}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Persistent across calendar periods}.
}
\]

因此当前较稳健的核心表述可以是：

> **Several network-position characteristics contain temporally persistent forward idiosyncratic-risk information beyond conventional stock characteristics.**

---

# 10. 当前 M1 实施层面的结论

Capacity 结果更适合描述为：

\[
\boxed{
\text{Implementation capacity is strongly affected by the secular market environment.}
}
\]

而不是：

\[
NetworkStress
\Rightarrow
Capacity.
\]

这意味着：

> 网络风险信息的统计稳定性与组合实施环境的时间变化是两个不同层面的问题。

这两个结论可以同时成立，并不存在矛盾。

---

# 11. 研究限制与解释纪律

今天所有结果仍需遵守以下原则：

- 所有关系均为统计关联，不作因果解释；
- 不根据 Day 10 结果重新选择因子、窗口或方向；
- 不因为 W252 对 delta degree 最稳定就称其为“最优窗口”；
- `same_sign_year_share=100%` 不表示“100% 概率稳定”；
- 2015/2026 等边界年份为 partial years，应注明但不事后删除；
- Step 3 的年度 HIGH–LOW overlap 很有限，尤其 W120 只有 1 个直接 overlap year；
- Capacity 的 year-adjusted 负向点估计缺乏统一证据，不应解释为 High Stress 导致低 Capacity；
- 稳定为负的 IVOL IC 同样属于有效风险信息，不应人为翻转 sign；
- Step 5 的稳定性分类仅用于描述，不用于重新构造新因子。

---

# 12. 今日完成状态

当前已完成：

- Step 1 — Calendar-Time / Market-Liquidity Diagnostic：**PASS**
- Step 2 — Time-Adjusted Network-Stress Capacity Validation：**PASS**
- Step 3 — Within-Year Capacity Validation：**PASS + FROZEN**
- Step 4 — Future-IVOL Temporal Stability：**PASS + FROZEN**
- Step 5 — Frozen 9-Factor Temporal Stability Matrix：**PASS + FROZEN**
- Step 6 — Temporal Robustness Summary：**代码与汇总设计已完成，待最终运行**

因此今天当前状态可以写为：

\[
\boxed{
\textbf{Day 10 empirical validation substantially complete}
}
\]

在 Step 6 最终运行并通过 QA 后，可进一步记录：

\[
\boxed{
\textbf{Day 10 = PASS + FROZEN}
}
\]

---

# 13. 今日最核心的一句话总结

\[
\boxed{
\textbf{Implementation capacity is strongly shaped by secular market conditions,}
}
\]

\[
\boxed{
\textbf{whereas network-based forward idiosyncratic-risk information exhibits}
}
\]

\[
\boxed{
\textbf{substantially stronger temporal persistence.}
}
\]

中文：

> **实施容量明显受到长期市场环境与时期构成影响，而网络结构所包含的未来特质风险信息则表现出更强的跨时期稳定性。**

---

# 14. 后续建议

在 Step 6 正式运行并通过 QA 后，Day 10 不建议继续增加新的 temporal test。

下一阶段应优先整合 Day 6–10，形成 M1 的正式实证主线：

\[
\boxed{
\text{Network Feature}
\rightarrow
\text{Forward Risk Information}
\rightarrow
\text{State Robustness}
\rightarrow
\text{Temporal Robustness}
\rightarrow
\text{Implementation Reality}.
}
\]

后续重点应转向：

1. 整理最终核心表格与图形；
2. 区分主结论、robustness 和 exploratory findings；
3. 准备 M1 因子研究报告与汇报材料；
4. 在不改变 frozen design 的前提下讨论与 M4 尾部风险传导路径研究的衔接。

