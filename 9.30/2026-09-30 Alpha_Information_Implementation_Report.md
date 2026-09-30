# M1 股票关联网络量化研究：Frozen Network Alpha Economic Value and Implementability

**日期：2026-09-29**\
**主题：Frozen Network Alpha Economic Value and Implementability
Validation**\
**研究主线：信息含量 → 组合单调性 → 换手与交易成本 → 容量约束 → 联合证据汇总**

------------------------------------------------------------------------

## 1. 今日研究目标

Day 12
在此前已经冻结的网络特征、收益标签、传统股票特征与流动性/容量口径基础上，不再进行新的因子搜索或参数选择，而是集中回答网络
Alpha 的三个问题：

1.  网络特征是否具有稳定的未来收益排序信息？
2.  该信息能否转化为单调的组合收益，并在交易成本后保留经济价值？
3.  在冻结的流动性与容量约束下，该收益关系能够承载多大的实际资金规模？

今日工作严格遵循 **frozen specification / point-in-time / no post-hoc
selection**
原则。因子方向、网络窗口、成本水平以及容量口径均不根据结果事后选择。

------------------------------------------------------------------------

## 2. 冻结研究对象

### 2.1 网络因子

今日全程使用以下 9 个 Day 5 冻结网络因子：

-   `residual_degree_percentile`
-   `delta_degree_percentile`
-   `cross_industry_degree_percentile`
-   `cross_industry_degree_ratio`
-   `delta_residual_degree_percentile_1m`
-   `delta_cross_industry_degree_percentile_1m`
-   `neighbor_jaccard_1m`
-   `neighbor_retention_1m`
-   `outside_community_degree_percentile`

网络窗口固定为：

\[ W`\in`{=tex}{60,120,252}. \]

### 2.2 传统特征控制

特征中和使用冻结的行业与传统股票特征：

\[ `\text{industry FE}`{=tex} +`\log`{=tex}(MV)
+`\text{Momentum}`{=tex}*{120,20} +`\text{Reversal}`{=tex}*{20}
+`\text{Volatility}`{=tex}*{60}
+`\log`{=tex}(`\text{Turnover}`{=tex}*{20}). \]

### 2.3 主要收益标签

Day 6 的 `future_total_return` 被统一重命名为：

`future_holding_return`

并继续使用 `future_return_label_valid` 等字段进行 point-in-time
与标签有效性审计。

------------------------------------------------------------------------

# 3. Step 1 --- Frozen Alpha Evaluation Panel & Return-Label Audit

## 3.1 目的

建立 Day 12 后续所有 Alpha 检验唯一使用的冻结评价面板，并确认：

-   当前时点网络特征不被未来收益可得性反向筛选；
-   forward return 标签确实对应未来区间；
-   同一股票、同一日期在不同网络窗口下的未来收益保持一致；
-   网络因子、传统控制变量、流动性变量的合并没有改变 Day 5
    冻结股票宇宙。

## 3.2 核心结果

最终面板保持 Day 5 基准宇宙：

\[ N=1,549,521. \]

主要 QA 结果：

-   duplicate stock-date-window = 0；
-   frozen factors = 9；
-   windows = 3；
-   controls = 6；
-   `future_total_return` 成功冻结为主要收益标签；
-   有效 future return 数量 = 1,518,496；
-   cross-window future-return mismatch = 0；
-   non-forward label = 0；
-   non-finite return = 0；
-   unexpected future columns = 0。

可评价样本规模约为：

-   Raw Alpha：1,404,119；
-   Characteristic-Neutral Alpha：1,394,615；
-   Implementation：1,394,418。

**结论：Step 1 = PASS + FROZEN。**

------------------------------------------------------------------------

# 4. Step 2 --- Frozen Alpha IC and Characteristic-Neutral Validation

## 4.1 方法

Raw Alpha 使用日期内横截面 Spearman Rank IC：

\[ IC_t=`\operatorname{Corr}`{=tex}*{rank}
`\left`{=tex}(F*{i,t},R\_{i,t`\rightarrow `{=tex}t+h}`\right`{=tex}). \]

Characteristic-neutral 检验采用显式冻结的 partial-rank / FWL 方法：

1.  对网络因子做横截面 rank-z；
2.  对未来收益做横截面 rank-z；
3.  对 5 个连续传统特征做 rank-z；
4.  使用完整行业固定效应 + 5 个传统特征分别残差化因子和未来收益；
5.  计算两个残差之间的相关系数。

同时使用：

-   `NATIVE` sample；
-   `COMMON_DATE` sample；
-   Newey-West HAC lag = 3；
-   27 个 factor-window cells 内 BH 多重检验修正。

## 4.2 最重要结果：delta degree

`delta_degree_percentile` 是 Raw IC 中最清晰的负向关系：

  Window      Raw IC   HAC t
  -------- --------- -------
  60         -0.0497   -6.11
  120        -0.0457   -4.98
  252        -0.0379   -3.50

Characteristic-neutral 后：

  Window     Neutral IC   HAC t
  -------- ------------ -------
  60            -0.0134   -2.86
  120           -0.0097   -1.91
  252           -0.0090   -1.70

因此：

\[ \|IC^{Neutral}\|`\ll `{=tex}\|IC^{Raw}\|. \]

这说明 Raw `delta_degree`
的未来收益关系中相当一部分与传统股票特征重叠，但控制后仍残留较弱的负向网络信息。

其他部分网络位置/邻居稳定性变量在某些窗口存在正向 neutral
IC，但整体强度明显弱于 Raw `delta_degree`。

**结论：Step 2 = PASS + FROZEN。**

------------------------------------------------------------------------

# 5. Step 3 --- Monotonic Portfolio and Long--Short Validation

## 5.1 方法

对每个 factor-window-date 将股票按当前因子排序为五组：

\[ Q1,Q2,Q3,Q4,Q5, \]

其中 Q1 为最低网络因子组，Q5 为最高组。

主要组合价差始终冻结为：

\[ H-L=Q5-Q1. \]

即使 Step 2 已发现 `delta_degree` 为负向 IC，也不事后翻转因子方向。

同时构建：

-   `RAW_FACTOR_SORT`
-   `CHARACTERISTIC_NEUTRAL_FACTOR_SORT`

并分别在 `NATIVE` 与 `COMMON_DATE` 下评价。

未来收益只在组合形成完成以后用于 realized-return evaluation，不参与
formation。

## 5.2 主要发现

`delta_degree_percentile` 的 quintile portfolio 与 Step 2 的负向 IC
一致：

\[ Q5-Q1\<0. \]

即较高的 delta degree 对应较低的未来收益，因此机械反向组合：

\[ Q1-Q5 \]

具有正的 gross return。

Step 3 最终 COMMON_DATE portfolio calendar 为 **95
个有效日期**。该步骤建立了 IC 信息与实际组合收益之间的第一层经济映射。

**最终版本：Step 3 = PASS + FROZEN。**

------------------------------------------------------------------------

# 6. Step 4 --- Turnover and Transaction-Cost Stress Test

## 6.1 换手定义

对 Q1、Q5 等权组合，根据相邻再平衡日股票权重变化定义 one-way traded
notional：

\[ Trade_t= `\sum`{=tex}*i\|w\^{Q5}*{i,t}-w\^{Q5}*{i,t-1}\| +
`\sum`{=tex}*i\|w\^{Q1}*{i,t}-w\^{Q1}*{i,t-1}\|. \]

同时报告：

\[ Turnover\^{1/2}\_t=`\frac{1}{2}`{=tex}Trade_t. \]

首次建仓同样计入交易成本。

固定 one-way transaction-cost grid：

\[ c`\in`{=tex}{10,20,30,50}`\text{ bps}`{=tex}. \]

净收益：

\[ R_t^{net}=R_t^{gross}-c`\times `{=tex}Trade_t. \]

## 6.2 换手规律

9 个因子平均来看，较长网络窗口产生更稳定的股票排序，因此：

\[ Turnover\_{60}\>Turnover\_{120}\>Turnover\_{252}. \]

NATIVE Raw portfolios 的平均 one-way traded notional 约为：

-   W60：2.38；
-   W120：2.01；
-   W252：1.78。

Characteristic-neutral portfolio 的换手总体略高于 Raw portfolio。

## 6.3 delta degree 的成本韧性

NATIVE Raw `delta_degree` 的机械 (Q1-Q5)：

  Window     Gross Q1-Q5   Mean Trade   Break-even Cost   Net @20bps
  -------- ------------- ------------ ----------------- ------------
  60              1.158%        1.813          63.9 bps       0.796%
  120             1.146%        1.215          94.3 bps       0.903%
  252             0.921%        0.817         112.6 bps       0.757%

因此 Raw delta-degree spread 在 10--30 bps
的固定成本压力下仍具有较明显的经济韧性。

但 characteristic-neutral 后，delta-degree 的 break-even cost 仅约：

\[ 15.9, 16.7, 20.6`\text{ bps}`{=tex}, \]

在 20 bps 成本附近，剩余独立 Alpha 基本被交易成本耗尽。

此外，短期网络变化类变量的 one-way traded notional 经常达到约
3--3.4，说明：

\[ `\text{predictive information}`{=tex} `\neq`{=tex}
`\text{implementable alpha}`{=tex}. \]

**结论：Step 4 = PASS + FROZEN。**

------------------------------------------------------------------------

# 7. Step 5 --- Capacity-Constrained Alpha Validation

## 7.1 容量定义

Step 5 不重新估计容量，而是直接复用 Day 8 已冻结的：

-   ADV20；
-   participation rate = 5%；
-   execution horizon = 1 day；
-   executable-share threshold = 95%。

报告两种容量：

严格全日期容量：

\[ Capacity\^{AllDates} = `\min`{=tex}\_t Capacity_t, \]

以及更适合描述典型实施能力的 95%-dates capacity：

\[ Capacity\^{95%Dates} = Q\_{0.05}(Capacity_t). \]

## 7.2 Formal QA

修复 Step 4 → Step 5 的字段接口后，最终结果：

-   capacity source rows = 3,556；
-   duplicate capacity key = 0；
-   negative capacity = 0；
-   monthly rows = 12,126；
-   capacity match share = **99.596%**；
-   required minimum = 95%；
-   summary rows = 108；
-   capacity reestimated = False；
-   Day 8 capacity reused directly = True；
-   all formal QA = True。

## 7.3 delta degree 的容量结果

NATIVE Raw `delta_degree`：

  -----------------------------------------------------------------------
  Window         95%-Dates Capacity   Net Q1-Q5 @20bps  Annual P&L at 95%
                                                                 Capacity
  -------------- ------------------ ------------------ ------------------
  60                      3.69 亿元          0.796%/期       约 3526 万元

  120                     2.86 亿元          0.903%/期       约 3095 万元

  252                             0          0.757%/期                  0
  -----------------------------------------------------------------------

COMMON_DATE Raw：

  Window     95%-Dates Capacity   Net Q1-Q5 @20bps
  -------- -------------------- ------------------
  60                  3.92 亿元          0.328%/期
  120                 3.52 亿元          0.564%/期
  252                 2.36 亿元          0.370%/期

其中 W120 COMMON_DATE 对应的 20 bps、95%-dates capacity 年化 arithmetic
P&L 约为：

\[ 2381`\text{ 万元}`{=tex}. \]

需要注意，当前 capacity 是 Day 8 冻结的 factor-date-window
implementation benchmark；若 Day 8 容量源没有 method/scope 维度，则 Raw
与 Neutral 共享该冻结 benchmark，因此不能将其解释为每一种 portfolio
method 单独重新估计的精确容量。

## 7.4 经济解释

Raw `delta_degree` 的收益并非只能存在于极小资金规模下。W60/W120
在固定交易成本后仍保留正收益，并对应数亿元量级的 95%-dates capacity。

但 characteristic-neutral `delta_degree`
的主要问题并不是没有容量，而是：

\[ `\boxed{\text{独立 Alpha 本身过薄}}`{=tex} \]

其收益在约 20 bps 成本下已接近或低于零。

**结论：Step 5 = PASS + FROZEN。**

------------------------------------------------------------------------

# 8. Step 6 --- Alpha Information--Implementation Joint Summary

## 8.1 目的

Step 6 不再进行新的估计，而是将 Step 2--5 已冻结证据统一到同一张 joint
matrix 中：

\[ `\boxed{
IC
\rightarrow
Portfolio
\rightarrow
Turnover/Cost
\rightarrow
Capacity
}`{=tex} \]

联合键固定为：

`sample_scope × method × window × factor`

因此完整 joint summary 应有：

\[ 2`\times2`{=tex}`\times3`{=tex}`\times9`{=tex}=108 \]

个 cells。

成本-容量 stress matrix 则进一步展开：

\[ 108`\times4`{=tex}=432 \]

个 cells。

## 8.2 汇总内容

联合表将同时保存：

-   mean IC、HAC t、BH q；
-   gross Q5-Q1；
-   quintile monotonicity；
-   portfolio sign consistency；
-   one-way traded notional；
-   break-even transaction cost；
-   10/20/30/50 bps 下 Q5-Q1 与机械 Q1-Q5 的 net return；
-   frozen 95%-dates capacity；
-   capacity-scaled annual P&L。

Step 6 不允许：

-   根据结果选择 factor；
-   根据结果选择 window；
-   根据结果选择交易方向；
-   根据结果选择成本水平；
-   重新估计 capacity。

## 8.3 当前状态

今日已经完成 Step 6 的精简可运行脚本，并完成 Python 语法检查。脚本只读取
Step 2--5 的冻结 summary，不重新估计任何因子、收益、交易成本或容量。

**截至本汇报生成时，Step 6 尚未提供实际运行后的
QA/output，因此当前应记录为：**

\[ `\boxed{\text{CODE READY / EXECUTION PENDING}}`{=tex} \]

不能在正式运行 `day12_step6_qa.csv` 并确认 `all_formal_qa_pass=True`
之前将 Step 6 标记为 PASS + FROZEN。

------------------------------------------------------------------------

# 9. 今日最重要的研究结论

Day 12 的证据链表明，网络 Alpha
的"统计预测能力"和"独立、可实施经济价值"需要严格区分。

对目前最明显的 `delta_degree_percentile`：

\[ `\text{Raw IC}`{=tex} `\rightarrow`{=tex} `\text{明显负向}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ Q5-Q1\<0 \]

\[ `\Downarrow`{=tex} \]

\[ Q1-Q5 `\text{ 形成正 gross spread}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ 10`\text{--}`{=tex}30`\text{ bps 成本后仍有韧性}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ W60/W120 \\text{ 可对应数亿元级 95%-dates capacity}. \]

但是：

\[ `\text{Characteristic Neutralization}`{=tex} \]

之后，

\[ \|IC\|`\downarrow`{=tex},`\qquad`{=tex} \|Q5-Q1\|`\downarrow`{=tex},
\]

并且约 20 bps 的交易成本已经足以使大部分剩余独立收益接近零。

因此当前最稳妥的结论是：

> **delta degree contains a strong gross return signal, but a
> substantial part of its economic value overlaps with conventional
> stock characteristics. The residual characteristic-neutral network
> alpha is statistically weaker and economically thin after realistic
> turnover and transaction-cost adjustments.**

换言之：

\[ `\boxed{
\text{Raw network-return relation is economically meaningful,}
}`{=tex} \]

但不能直接等价为：

\[ `\boxed{
\text{large independent tradable network alpha}.
}`{=tex} \]

------------------------------------------------------------------------

# 10. 今日研究状态

  -----------------------------------------------------------------------------
  Step                    内容                          状态
  ----------------------- ----------------------------- -----------------------
  Step 1                  Frozen Alpha Evaluation Panel **PASS + FROZEN**
                          & Return-Label Audit          

  Step 2                  Frozen Alpha IC &             **PASS + FROZEN**
                          Characteristic-Neutral        
                          Validation                    

  Step 3                  Monotonic Portfolio &         **PASS + FROZEN**
                          Long--Short Validation        

  Step 4                  Turnover & Transaction-Cost   **PASS + FROZEN**
                          Stress Test                   

  Step 5                  Capacity-Constrained Alpha    **PASS + FROZEN**
                          Validation                    

  Step 6                  Alpha                         **CODE READY /
                          Information--Implementation   EXECUTION PENDING**
                          Joint Summary                 
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

# 11. 今日形成的完整 Alpha 研究框架

经过 Step 1--6，M1 Alpha 侧已经形成如下标准化研究流程：

\[ `\boxed{
\text{Frozen PIT Panel}
}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ `\boxed{
\text{Raw / Characteristic-Neutral IC}
}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ `\boxed{
\text{Monotonic Quintile Portfolio}
}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ `\boxed{
\text{Turnover + Transaction Cost}
}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ `\boxed{
\text{Capacity Constraint}
}`{=tex} \]

\[ `\Downarrow`{=tex} \]

\[ `\boxed{
\text{Information–Implementation Joint Evidence}
}`{=tex} \]

这一流程避免只根据 IC 或 gross backtest
判断因子价值，而是要求一个网络信号同时经历
**统计信息、组合映射、交易摩擦和容量约束** 四层验证。

------------------------------------------------------------------------

# 12. 下一步

下一步首先运行 Step 6，并检查：

-   joint rows 是否为 108；
-   stress rows 是否为 432；
-   joint/stress duplicate key 是否为 0；
-   Step 2、Step 4、Step 5 formal QA 是否全部继承通过；
-   是否不存在 factor/window/direction/cost 的事后选择；
-   `all_formal_qa_pass` 是否为 True。

若 Step 6 正式通过，则 Day 12 Alpha 侧可整体冻结，并据此生成最终
9-factor × 3-window Alpha 信息---实施联合矩阵，作为 M1 因子研究报告中
IC、分层回测、换手率、交易成本和容量分析的统一结果表。
