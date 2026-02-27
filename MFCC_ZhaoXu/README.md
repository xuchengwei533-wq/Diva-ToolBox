# MFCC_ZhaoXu

本目录用于把音频转换为 MFCC，并基于 MFCC/标签做训练与结果输出。

## 使用方法

1. 安装依赖
   - `python -m pip install -r requirements.txt`
2. 准备音频
   - 音频文件在 `Predictive-singing-regression-analysis\Chest new0206\` 下（`.wav`）。
3. 生成 MFCC
   - 修改 `MFCCnew.py` 中的路径：
     - `directory = r"d:\xuchengwei\声音提取工具包\Predictive-singing-regression-analysis\Chest new0206"`
     - 若音频直接放在上述目录，请把 `Audio_path` 改成 `directory`；若存在 `Audio` 子目录则保留默认写法。
     - `MFCC_out = r"d:\xuchengwei\声音提取工具包\MFCC_ZhaoXu\MFCC"`
   - 运行 `python MFCCnew.py`
   - 处理好的 MFCC 会输出到 `d:\xuchengwei\声音提取工具包\MFCC_ZhaoXu\MFCC`。
4. 训练（可选）
   - `CAM_S.py` 读取 `data_dir` 下的 `MFCC_Output/` 与 `Label/`。
   - 修改 `CAM_S.py` 末尾 `data_dir` 为你的数据目录后，运行 `python CAM_S.py`。

## 依赖

- 依赖版本见 [requirements.txt](file:///d:/xuchengwei/声音提取工具包/MFCC_ZhaoXu/requirements.txt)。

## 文件说明

### MFCCnew.py

 - 作用：遍历音频目录中的 `.wav` 文件，统一重采样到 44100Hz，提取 MFCC（默认 128 维），并保存为 `*_MFCC.xlsx`。
 - 输入：
   - `Predictive-singing-regression-analysis\Chest new0206\` 下的 `.wav`（路径在脚本内可改）。
 - 输出：
   - `d:\xuchengwei\声音提取工具包\MFCC_ZhaoXu\MFCC\*_MFCC.xlsx`（路径在脚本内可改）。
 - 依赖：`librosa`、`numpy`、`pandas`；可选 `soundfile` 用于保存重采样 wav。

### CAM_S.py

- 作用：基于 MFCC 的深度学习训练脚本，内部包含网络结构定义（`CAMPPlus` 等）与训练/验证循环。
- 数据组织约定（`data_dir` 下）：
  - `MFCC_Output/`：每个样本一个 `*_MFCC.xlsx`（作为输入特征）。
  - `Label/`：每个样本一个 `.xlsx`（作为标签），脚本读取 `values[:, 1:][:10, :].ravel()` 并减 1，构造成 10 个技巧的分类标签（每个技巧 1~5 分映射为 0~4）。
- 训练方式：
  - 80% 训练 / 20% 验证（按样本对配对后随机打乱再切分）。
  - 输出为 `num_classes=50`（10 个技巧 × 5 个等级），训练时把模型输出 reshape 成 `(B, 5, 10)` 后按 10 个技巧分别取 argmax。
- 输出：
  - 默认保存到 `logs_ddnet_sopran_2637/<timestamp>/best_model.pth`（当验证准确率提升时覆盖保存）。
  - 若安装了 `tensorboardX` 会写 TensorBoard 日志；未安装则自动降级为空实现。
- 依赖：`torch`、`pandas`、`openpyxl`（读取 xlsx）、`torchlibrosa`（SpecAugmentation）。


