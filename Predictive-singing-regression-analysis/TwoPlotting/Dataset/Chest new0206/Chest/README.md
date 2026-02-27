# Chest TwoPlotting Scripts

本目录包含用于生成 Chest 声乐技术特征二维散点图的 Python 脚本。

## 目录结构

- `TwoPlot_A1.py`: 生成 A1 组（后缀为 -A 或 -1）数据的特征两两对比散点图。
- `TwoPlot_B1.py`: 生成 B1 组（后缀为 -B 或 -1）数据的特征两两对比散点图。
- `TwoPlot_ALL.py`: 生成所有数据的特征两两对比散点图。
- `TwoPlot_A1_B1_ALL.py`: 生成 A1、B1 和 ALL 三组数据的对比图。

## 环境依赖

请确保已安装 Python 3.x，并安装以下依赖库：

```bash
pip install -r requirements.txt
```

依赖库列表：
- numpy
- pandas
- matplotlib
- openpyxl

## 运行方式

在当前目录下运行对应的 Python 脚本即可。例如：

```bash
python TwoPlot_A1.py
```

或者一次性运行所有脚本：

```bash
python TwoPlot_A1.py
python TwoPlot_B1.py
python TwoPlot_ALL.py
python TwoPlot_A1_B1_ALL.py
```

## 输出结果

- `TwoPlot_A1.py` 的结果保存在 `A1` 子目录。
- `TwoPlot_B1.py` 的结果保存在 `B1` 子目录。
- `TwoPlot_ALL.py` 的结果保存在 `ALL` 子目录。
- `TwoPlot_A1_B1_ALL.py` 的结果保存在 `A1_B1_ALL` 子目录。

生成的图片为 PNG 格式，散点图根据评分（1, 3, 5）使用不同颜色（蓝、绿、橙）进行标记。
