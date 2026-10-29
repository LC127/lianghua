# Day4 网络文件结构审计

- Day4目录：`D:\M1_StockNetwork\output\M1_day4`
- CSV/Parquet文件总数：2857
- 网络相关路径候选数：2027
- 识别出端点字段组合的文件数：0

## 输出文件
- day4_schema_inventory.csv：全部文件的字段结构。
- day4_network_edge_candidates.csv：网络相关候选文件。

如果端点识别数仍为0，请检查候选文件的columns_json，
再根据真实字段名补充PAIRS；不能据此断言网络文件不存在。
