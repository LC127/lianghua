# 2026-08-14 工作总结：Graphical Lasso 股票条件关联网络

## 一、今日工作主题

今日继续围绕**股票关联网络**展开研究。在前两天已经完成 Pearson 相关网络、样本协方差矩阵、精度矩阵、偏相关矩阵及偏相关阈值网络的基础上，今日进一步将研究推进到：

\[
\boxed{
\text{稀疏精度矩阵估计}
\rightarrow
\text{Graphical Lasso}
\rightarrow
\text{正则化参数选择}
\rightarrow
\text{网络稳健性比较}
\rightarrow
\text{重采样稳定性分析}
}
\]

今日的核心研究问题为：

> **如何利用稀疏精度矩阵估计得到更加稳定、可解释的股票条件关联网络，并判断这些网络边是否对正则化参数、构网方法以及样本扰动稳健？**

---

# 二、阶段一：理解为什么需要稀疏精度矩阵

设股票收益率向量为

\[
\mathbf r=(r_1,\ldots,r_p)^\top,
\]

协方差矩阵为

\[
\Sigma=\operatorname{Cov}(\mathbf r),
\]

精度矩阵为

\[
\Omega=\Sigma^{-1}.
\]

偏相关系数满足

\[
\rho_{ij\mid-rest}
=
-
\frac{\omega_{ij}}
{\sqrt{\omega_{ii}\omega_{jj}}}.
\]

因此对于非对角元素：

\[
\omega_{ij}=0
\Longleftrightarrow
\rho_{ij\mid-rest}=0.
\]

在多元高斯假设下，还可进一步解释为

\[
\omega_{ij}=0
\iff
r_i\perp r_j
\mid
r_{-\{i,j\}}.
\]

由此，精度矩阵的零/非零结构可以直接对应网络中的“无边/有边”。

但有限样本下直接计算

\[
\widehat\Omega
=
\widehat\Sigma^{-1}
\]

通常会得到一个稠密矩阵，即许多理论上很弱甚至应为零的关系都会由于样本误差表现为小的非零值。若简单使用

\[
\widehat\omega_{ij}\neq0
\]

作为连边规则，就会得到过于稠密、难以解释的网络。

因此需要估计一个**稀疏精度矩阵**：

\[
\boxed{
\text{大量非对角元素被直接估计为0}
}
\]

以达到：

- 删除弱或不稳定条件关联；
- 降低高维参数估计复杂度；
- 提高网络结构可解释性；
- 为后续高维股票池提供可扩展方法。

---

# 三、阶段二：理解 Graphical Lasso

Graphical Lasso 的目标函数为

\[
\boxed{
\widehat\Omega_\lambda
=
\arg\min_{\Omega\succ0}
\left\{
\operatorname{tr}(S\Omega)
-
\log\det(\Omega)
+
\lambda
\sum_{i\neq j}
|\omega_{ij}|
\right\}.
}
\]

其中前两项

\[
\operatorname{tr}(S\Omega)-\log\det(\Omega)
\]

来自高斯模型的负对数似然，用于拟合样本协方差结构；

第三项

\[
\lambda\sum_{i\neq j}|\omega_{ij}|
\]

为非对角元素的 \(\ell_1\) 正则化项，用于促使部分

\[
\widehat\omega_{ij}=0.
\]

因此 Graphical Lasso 可以理解为：

\[
\boxed{
\text{精度矩阵估计}
+
\text{网络稀疏化}
}
\]

同时进行。

正则化参数满足：

\[
\lambda\uparrow
\Rightarrow
\text{更多 }\widehat\omega_{ij}=0
\Rightarrow
\text{网络更加稀疏}.
\]

今日进一步明确了 Graphical Lasso 与昨日偏相关阈值方法的本质区别：

### 偏相关阈值法

\[
S^{-1}
\rightarrow
\rho^{Partial}
\rightarrow
|\rho^{Partial}|\ge\tau.
\]

属于：

\[
\boxed{\text{先估计稠密矩阵，再人工删边}}
\]

### Graphical Lasso

\[
S
\rightarrow
\widehat\Omega_\lambda.
\]

属于：

\[
\boxed{\text{在估计过程中直接产生稀疏结构}}
\]

---

# 四、阶段三：正则化参数路径分析

## 4.1 研究目的

阶段三不直接寻找“最优 \(\lambda\)”，而是通过 regularization path 直观观察：

\[
\boxed{
\lambda
\rightarrow
\text{精度矩阵稀疏程度}
\rightarrow
\text{股票网络拓扑变化}
}
\]

主要考察：

- 网络边数；
- Density；
- Connected Components；
- Isolated Nodes；
- Mean Degree；
- Max Degree；
- 非零边的平均/最大绝对偏相关。

## 4.2 实际结果

在 15 只股票中，最多存在

\[
\binom{15}{2}=105
\]

条无向边。

正则化路径结果显示：

| \(\alpha\) | Edges | Density | Components | Isolated |
|---:|---:|---:|---:|---:|
| 0.0088 | 83 | 0.790 | 1 | 0 |
| 0.0972 | 59 | 0.562 | 1 | 0 |
| 0.2169 | 50 | 0.476 | 1 | 0 |
| 0.3510 | 40 | 0.381 | 1 | 0 |
| 0.4122 | 29 | 0.276 | 2 | 1 |
| 0.4839 | 18 | 0.171 | 5 | 3 |
| 0.5681 | 13 | 0.124 | 7 | 5 |
| 0.6670 | 6 | 0.057 | 10 | 7 |
| 0.7832 | 4 | 0.038 | 12 | 10 |
| 0.9195 | 0 | 0 | 15 | 15 |

由此可以看到非常清晰的规律：

\[
\boxed{
\alpha\uparrow
\Rightarrow
|E|\downarrow
\Rightarrow
\text{网络逐渐由稠密变为稀疏并最终碎片化}.
}
\]

其中

\[
\boxed{
\alpha\approx0.35\sim0.57
}
\]

是当前数据中最明显的结构转折区间。

在

\[
\alpha\approx0.35
\]

时，网络仍保持整体连通；进一步提高正则化后，连通分量和孤立节点迅速增加。

这说明正则化路径并非简单均匀变化，而存在明显的网络结构压缩区间。

---

# 五、阶段四：选择合适的 \(\lambda\) 并构建 Graphical Lasso 网络

## 5.1 CV-optimal 参数

采用时间顺序交叉验证选择正则化参数，得到：

\[
\boxed{
\lambda_{\mathrm{CV}}
=
0.031617.
}
\]

对应平均验证 Gaussian log-likelihood 约为：

\[
-17.147454.
\]

最终 CV-GLasso 网络统计为：

\[
|E_{\mathrm{CV}}|=69,
\]

\[
Density=0.6571,
\]

\[
Mean\ Degree=9.2,
\]

\[
Max\ Degree=11,
\]

\[
Components=1,
\qquad
Isolated=0.
\]

说明基于验证似然的参数选择偏向较弱正则化，得到的网络仍较稠密。

## 5.2 CV 曲线的特点

最优参数附近的 CV score 非常平坦，例如：

\[
\alpha=0.031617
\]

与

\[
\alpha=0.037121
\]

的平均验证得分差异仅约

\[
1.8\times10^{-4}.
\]

而不同时间折之间的 score 标准差约为

\[
1.94.
\]

因此不能将

\[
0.031617
\]

理解为一个非常精确、唯一的最优参数，而应理解为：

\[
\boxed{
\text{CV 最优区域位于弱正则化区间}.
}
\]

## 5.3 1-SE 简约网络

为了获得更具有解释性的网络，引入 1-SE 式参数选择。

得到：

\[
\boxed{
\lambda_{\mathrm{1SE}}
=
0.216910.
}
\]

其正则化强度约为 CV 最优参数的

\[
6.86
\]

倍。

对应网络：

\[
|E_{\mathrm{1SE}}|=50,
\]

\[
Density=0.4762,
\]

\[
Mean\ Degree=6.67,
\]

\[
Max\ Degree=11,
\]

\[
Components=1,
\qquad
Isolated=0.
\]

即在显著减少网络边数的同时，15只股票仍全部保持连通。

CV 与 1-SE 网络共有：

\[
48
\]

条边，

Jaccard similarity 为：

\[
0.6761.
\]

CV 网络约：

\[
69.6\%
\]

的边在 1-SE 网络中仍被保留，而 1-SE 网络中：

\[
96\%
\]

的边也同时存在于 CV 网络中。

因此 1-SE 网络主要表现为：

\[
\boxed{
\text{在 CV 网络基础上删除一批正则化敏感的弱连接}.
}
\]

---

# 六、阶段五：偏相关阈值网络与 Graphical Lasso 网络比较

## 6.1 三个网络规模

此前偏相关阈值网络采用

\[
|\rho^{Partial}_{ij}|\ge0.20,
\]

共得到：

\[
|E_{\mathrm{Partial}}|=9.
\]

今日得到：

\[
|E_{\mathrm{GLasso,CV}}|=69,
\]

\[
|E_{\mathrm{GLasso,1SE}}|=50.
\]

最重要的发现是：

\[
\boxed{
E_{\mathrm{Partial}}
\subset
E_{\mathrm{GLasso,CV}}
}
\]

并且

\[
\boxed{
E_{\mathrm{Partial}}
\subset
E_{\mathrm{GLasso,1SE}}.
}
\]

即：

\[
\boxed{
\text{偏相关阈值网络中的9条强边全部被两种 GLasso 网络保留}.
}
\]

因此偏相关阈值网络与 Graphical Lasso 的主要差异，不是对强关系产生分歧，而是 Graphical Lasso 额外保留了较多弱但非零的条件关联。

## 6.2 九条跨方法共同强边

主要包括：

| 股票关系 | Partial | GLasso CV | GLasso 1-SE |
|---|---:|---:|---:|
| 工商银行–农业银行 | 0.781 | 0.756 | 0.609 |
| 五粮液–泸州老窖 | 0.586 | 0.561 | 0.427 |
| 平安银行–招商银行 | 0.486 | 0.464 | 0.393 |
| 贵州茅台–五粮液 | 0.400 | 0.367 | 0.313 |
| 中国平安–中信证券 | 0.332 | 0.319 | 0.260 |
| 平安银行–中国平安 | 0.307 | 0.299 | 0.266 |
| 韦尔股份–立讯精密 | 0.286 | 0.278 | 0.218 |
| 贵州茅台–泸州老窖 | 0.211 | 0.225 | 0.243 |
| 贵州茅台–伊利股份 | 0.205 | 0.193 | 0.170 |

九条边的关联方向：

\[
\boxed{100\%\text{ 一致}}
\]

且全部表现为正条件关联。

## 6.3 行业结构比较

三个网络同行业边比例为：

\[
\boxed{
\text{Partial threshold}:88.9\%
}
\]

\[
\boxed{
\text{GLasso 1-SE}:38.0\%
}
\]

\[
\boxed{
\text{GLasso CV}:27.5\%.
}
\]

偏相关阈值网络主要保留强行业内部关联，而 Graphical Lasso 进一步保留了大量跨行业中等或弱条件关联。

随着正则化增强：

\[
69\text{ edges}
\rightarrow
50\text{ edges},
\]

同行业边数量仍维持在 19 条，而跨行业边显著减少，因此行业结构变得更加突出。

## 6.4 节点角色差异

不同构网方法会明显改变股票的网络角色。

例如：

\[
d_{\mathrm{Partial}}(\text{中信证券})=1,
\]

而

\[
d_{\mathrm{1SE}}(\text{中信证券})=11.
\]

海康威视、赣锋锂业和京东方A在偏相关阈值网络中为孤立节点，但在 1-SE 网络中 Degree 分别约为：

\[
8,\quad7,\quad5.
\]

因此：

\[
\boxed{
\text{某股票是否表现为“核心节点”高度依赖构网规则}.
}
\]

---

# 七、阶段六：Graphical Lasso 网络重采样稳定性分析

## 7.1 方法

固定：

\[
\lambda_{\mathrm{1SE}}=0.216910,
\]

采用：

\[
B=200
\]

次 moving-block resampling。

每次使用约：

\[
80\%
\]

的样本，block length 取：

\[
L=20
\]

个交易日。

对每次重采样数据重新标准化并拟合 Graphical Lasso。

定义股票边的选择频率：

\[
\boxed{
\widehat\pi_{ij}
=
\frac{1}{B}
\sum_{b=1}^{B}
I\{(i,j)\in E^{(b)}\}.
}
\]

使用：

\[
\widehat\pi_{ij}\ge0.80
\]

作为描述性稳定边阈值，

以及：

\[
\widehat\pi_{ij}\ge0.90
\]

作为核心稳定边阈值。

## 7.2 数值稳定性

200 次重采样：

\[
\boxed{200/200\text{ 次全部有效并成功收敛}}
\]

每次网络边数：

\[
39\sim60,
\]

平均约：

\[
\boxed{49.37}
\]

标准差约：

\[
3.59.
\]

与全样本 1-SE 网络的：

\[
50\text{ 条边}
\]

高度一致。

## 7.3 Stable Graphical Lasso 网络

共有：

\[
\boxed{41}
\]

条边满足：

\[
\widehat\pi_{ij}\ge0.80.
\]

其中：

\[
\boxed{39}
\]

条满足：

\[
\widehat\pi_{ij}\ge0.90,
\]

且：

\[
\boxed{21}
\]

条边在全部 200 次重采样中均被重新选择：

\[
\widehat\pi_{ij}=1.
\]

稳定网络统计为：

\[
|E_{\mathrm{stable}}|=41,
\]

\[
Density=0.3905,
\]

\[
Mean\ Degree=5.47,
\]

\[
Max\ Degree=8,
\]

\[
Components=1,
\]

\[
Isolated=0.
\]

说明删除样本敏感边后，网络仍保持完整的条件关联骨架。

## 7.4 1-SE 网络的稳定边保留率

全样本 1-SE 网络共有：

\[
50
\]

条边，其中：

\[
41
\]

条达到稳定性阈值。

因此：

\[
\boxed{
\frac{41}{50}
=
82\%.
}
\]

即：

\[
\boxed{
82\%\text{ 的 1-SE 网络边对样本扰动具有较高稳定性}.
}
\]

稳定网络中没有出现：

\[
\boxed{
\text{full-sample 未选择、但重采样中高度稳定出现}
}
\]

的矛盾性边。

## 7.5 方向稳定性

对于全部：

\[
41
\]

条稳定边：

\[
\boxed{
\text{sign consistency}=100\%.
}
\]

即在每次被选中时，偏相关方向均保持一致。

## 7.6 九条跨方法共同强边的稳定性

阶段五识别出的 9 条跨方法共同强边，在阶段六进一步表现为：

\[
\boxed{
\widehat\pi_{ij}=1.00
}
\]

即全部在：

\[
200/200
\]

次重采样中被重新选择。

因此这 9 条边同时满足：

\[
\boxed{
\text{强 Partial}
+
\text{GLasso 支持}
+
\text{CV/1-SE 正则化稳健}
+
\text{重采样稳健}
}
\]

可以视为当前样本中最稳健的一组强条件关联。

## 7.7 行业结构进一步增强

同行业边比例呈现：

\[
27.5\%
\rightarrow
38.0\%
\rightarrow
43.9\%
\]

即：

\[
\boxed{
\text{CV GLasso}
\rightarrow
\text{1-SE GLasso}
\rightarrow
\text{Stable GLasso}.
}
\]

随着正则化和样本稳定性筛选增强，同行业条件关联在网络中的相对占比进一步提高。

这表明当前股票池中：

\[
\boxed{
\text{行业内部条件关系总体上比很多跨行业弱关系更加稳健}.
}
\]

---

# 八、今日形成的完整方法链条

今日六个阶段已经形成完整的静态条件关联网络分析框架：

\[
\boxed{
\begin{array}{c}
\text{普通精度矩阵}\\
\downarrow\\
\text{理解稀疏性需求}\\
\downarrow\\
\text{Graphical Lasso}\\
\downarrow\\
\lambda\text{ regularization path}\\
\downarrow\\
\text{CV选择 }\lambda\\
\downarrow\\
\text{1-SE简约网络}\\
\downarrow\\
\text{Partial vs GLasso方法比较}\\
\downarrow\\
\text{Moving-block resampling}\\
\downarrow\\
\text{Stable Graphical Lasso Network}
\end{array}
}
\]

这使研究对象由前一天的：

\[
\boxed{
\text{“控制其他股票后哪些股票仍具有较强条件关联？”}
}
\]

进一步推进到：

\[
\boxed{
\text{“哪些条件关联不仅被估计出来，而且对正则化参数、构网方法和样本扰动都稳健？”}
}
\]

---

# 九、今日最重要的研究发现

今日可以总结出以下几个主要结论：

1. **Graphical Lasso 的正则化参数显著影响股票网络结构。**  
   随着 \(\lambda\) 增大，网络由稠密、整体连通逐渐转向稀疏和碎片化，其中约 \(0.35\sim0.57\) 是明显结构转折区。

2. **CV 最优参数偏向较弱正则化。**  
   \[
   \lambda_{\mathrm{CV}}\approx0.0316
   \]
   对应69条边，网络较稠密。

3. **1-SE 参数提供了更简约且仍保持连通的网络。**  
   \[
   \lambda_{\mathrm{1SE}}\approx0.2169
   \]
   对应50条边，网络密度降至0.476，但仍保持单一连通分量。

4. **偏相关阈值网络中的9条强条件关联全部得到两种 GLasso 网络支持。**  
   因此两种方法对强关系具有高度一致性，差异主要来自 GLasso 额外保留大量较弱条件关联。

5. **强条件关联具有明显行业聚集特征。**  
   偏相关阈值网络同行业边比例达到88.9%。

6. **稳定性筛选进一步提取出41条稳定 GLasso 边。**  
   其中39条选择频率至少0.90，21条在全部200次重采样中出现。

7. **全样本1-SE网络中的82%边经受住了样本扰动。**

8. **9条跨方法共同强边在200次重采样中全部100%出现。**  
   这组边目前具有最强的跨方法、跨正则化和样本稳健性证据。

9. **行业内部条件关联在进一步筛选后更加突出。**  
   同行业边比例由 CV 网络的27.5%，上升到1-SE网络的38.0%，最终在稳定网络中达到43.9%。

---

# 十、当前方法的解释边界

今日分析仍需要保持以下解释边界：

- Graphical Lasso 网络属于条件关联网络，不等于因果网络；
- 精度矩阵零元素与严格条件独立的等价关系依赖多元高斯等假设；
- 1-SE 参数选择属于简约化的经验规则，不代表唯一最优网络；
- 稳定边阈值 \(0.80/0.90\) 当前属于描述性工作阈值；
- Moving-block resampling 中的 block length \(L=20\) 仍需进一步做敏感性检查；
- 当前股票池仅包含15只股票，因此行业 assortativity 和中心性结论应以描述性解释为主；
- 当前研究基于静态全样本网络，尚未考虑网络结构随时间变化。

---

# 十一、下一步建议

在今日静态 Graphical Lasso 条件关联网络已经较完整的基础上，后续可以优先开展两项工作。

## 11.1 Block length 稳健性检查

比较：

\[
L=10,\quad20,\quad40
\]

下的：

\[
\widehat\pi_{ij}.
\]

重点检查：

- 41条稳定边是否仍然稳定；
- 39条 core-stable 边是否保持；
- 9条跨方法共同强边是否始终达到高选择频率。

## 11.2 从静态网络进入动态网络

下一阶段可以进一步研究：

\[
\boxed{
\text{Static Graphical Lasso}
\rightarrow
\text{Rolling-window Graphical Lasso}
}
\]

例如使用滚动窗口构建：

\[
G_t,
\]

分析：

- 市场关联强度是否随时间变化；
- 行业内部连接何时增强；
- 金融股或消费股何时成为核心节点；
- 网络密度、中心性和社区结构如何随市场状态变化；
- 稳定强边是否在不同市场阶段持续存在。

---

# 十二、今日工作成果汇总

今日主要生成和使用了以下结果文件：

```text
glasso_regularization_path.csv
glasso_cv_results.csv
graphical_lasso_network_summary.csv

glasso_1se_selection.csv
graphical_lasso_1se_network_summary.csv
glasso_cv_vs_1se_network_summary.csv
glasso_cv_vs_1se_edge_comparison.csv

partial_vs_glasso_overlap_summary.csv
partial_vs_glasso_network_summary.csv
partial_vs_glasso_node_comparison.csv
partial_vs_glasso_common_edges.csv

glasso_stability_run_diagnostics.csv
glasso_edge_stability.csv
stable_glasso_edges.csv
stable_glasso_network_summary.csv
glasso_1se_vs_stable_summary.csv
```

并完成了：

- Graphical Lasso 正则化路径可视化；
- CV 参数选择图；
- CV / 1-SE Graphical Lasso 网络；
- 偏相关阈值网络与 GLasso 网络边变化图；
- selection frequency histogram；
- selection frequency heatmap；
- stable Graphical Lasso network。

---

# 十三、今日工作一句话总结

\[
\boxed{
\text{今日完成了从“条件关联网络”到“稀疏、参数稳健、方法稳健且样本稳健的条件依赖骨架”的完整推进。}
}
\]

相比前两天仅分析“哪些股票相关”以及“控制其他股票后哪些关系仍然较强”，今日进一步建立了一套用于识别**更加稳健的股票条件关联结构**的 Graphical Lasso 分析框架，为后续动态网络与更大股票池研究奠定了方法基础。
