# Day18 Final 下游一致性核验

- Final 主表：`D:\M1_StockNetwork\output\M1_day1\01_stage1_final\stock_master.parquet`
- 证券键：`security_id`；唯一证券数：5553
- Day4 网络候选：2857；识别并检查的边文件：0

输出：`final_panel_mapping_checks.csv`（收益面板/映射覆盖）；`final_network_node_checks.csv`（网络节点映射）。

注意：主表独有或面板独有证券不自动判错，须结合股票池、日期范围和网络节点定义解释。未识别端点列的文件不会进入节点检查。差异全部解释清楚后才能冻结 Final。
