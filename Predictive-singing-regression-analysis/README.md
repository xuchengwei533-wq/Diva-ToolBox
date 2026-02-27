# Predictive Singing Regression Analysis

本项目用于对声乐技巧评分（1/3/5）进行建模与解释分析，基于音频提取的声学参数进行序数回归训练，并提供特征组合对比与统计关联分析。

## 项目目的
- 使用可解释的声学特征对声乐技巧评分进行预测
- 比较不同特征组合的分类效果，找到表现最优的组合
- 输出统计分析结果，识别对评分影响显著的声学参数

## 目录与文件说明

本项目包含以下主要文件夹和文件，其作用如下：

### 1. 核心目录

*   **`Dataset/`**
    *   **作用**：存放原始音频数据集。
    *   **内容**：包含 `.wav` 格式的录音文件，通常按不同的声乐技巧（如 Chest, Falsetto 等）分类存放。

*   **`Extract/`**
    *   **作用**：特征提取模块。
    *   **内容**：
        *   `ExtractTool/`: 包含用于提取各种声学特征（如 Jitter, Shimmer, H1H2, HNR, QValue, CPP 等）的 Python 脚本。
        *   `ExtractOutput/`: 存放特征提取后的数据文件（`.csv` 格式），通常每个特征对应一个子文件夹。

*   **`OnePlotting/`**
    *   **作用**：单特征绘图。
    *   **内容**：包含用于绘制**单个声学特征**分布图的脚本及生成的图像。用于观察单一特征在不同评分下的分布情况。

*   **`TwoPlotting/`**
    *   **作用**：双特征（2D）绘图。
    *   **内容**：包含用于绘制**两个声学特征**之间关系的二维散点图的脚本及生成的图像。用于观察特征对在不同评分下的聚类或分布模式。

*   **`ThreePlotting/`**
    *   **作用**：三特征（3D）绘图。
    *   **内容**：包含用于绘制**三个声学特征**之间关系的三维散点图的脚本及生成的图像。用于在更高维空间中观察特征的交互关系。

*   **`OrdinalRegression_9Features_Output/`**
    *   **作用**：序数回归分析结果。
    *   **内容**：存放 `OrdinalRegression_9Features.py` 脚本的输出结果。
        *   `dataset_9features.csv`: 汇总了9个特征的原始数据。
        *   `dataset_9features_scaled.csv`: 标准化后的数据。
        *   `ordinal_regression_summary.txt`: 模型的详细统计摘要（系数、P值、置信区间等）。
        *   `ordinal_regression_coefficients.csv`: 特征影响力的排序表，包含回归系数和优势比（OR）。

*   **`PCA_Lasso/`**
    *   **作用**：主成分分析与 Lasso 回归结果。
    *   **内容**：
        *   **`PCA_LASSO_9Features_Output/`**: 存放整体数据的 PCA 和 Lasso 分析结果，包括载荷图、热力图、筛选出的特征系数等。
        *   **`PCA_LASSO_9Features_Output_A1_B1/`**: 存放针对 **A1**（后缀 A/1）和 **B1**（后缀 B/1）子数据集的分组分析结果。

### 2. 根目录主要脚本

*   **`PCA_LASSO_9Features.py`**
    *   **作用**：对所有数据进行整体的 PCA（降维）和 Lasso（特征筛选）分析，生成相关图表和统计文件。

*   **`PCA_LASSO_9Features_A1_B1.py`**
    *   **作用**：专门针对 A1 和 B1 两个子集进行 PCA 和 Lasso 分析，用于对比不同数据子集的特征表现。

*   **`OrdinalRegression_9Features.py`**
    *   **作用**：执行序数回归分析，建立声学特征与评分等级（1/3/5）之间的统计模型，评估各特征的显著性。

*   **`OrdinalFeatureAnalysis.py`**
    *   **作用**：进行更深入的声学参数影响分析，包括单因素分析、多重共线性检测（VIF）等。

*   **`TrainModels.py`** (如果存在)
    *   **作用**：用于训练不同的机器学习模型并对比其在特征组合上的表现。

## 运行环境
- Python 3.x
- 主要依赖：numpy、pandas、scikit-learn、mord、statsmodels、librosa、matplotlib

## 工作流程简述
1.  **特征提取**：运行 `Extract/` 下的脚本，从音频中提取声学参数。
2.  **数据分析**：
    *   运行 `OrdinalRegression_9Features.py` 查看特征对评分的统计显著性。
    *   运行 `PCA_LASSO_9Features.py` 进行特征降维和筛选。
3.  **可视化**：运行 `OnePlotting/`, `TwoPlotting/`, `ThreePlotting/` 下的脚本生成分布图。
