# Extract 使用说明

## 用法

推荐先运行批处理脚本完成环境安装：

```bat
.\Install_Extract_Environment.bat
```

脚本会自动创建虚拟环境并安装依赖：

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

完成环境安装后，手动运行特征提取程序：

```bash
python .\ExtractTool\Extract_NewFeatures.py
```

### 数据目录与输出目录

`Extract_NewFeatures.py` 会按以下规则寻找输入音频：

- 优先使用环境变量 `PSRA_DATA_ROOT`
- 未设置时默认使用 `Predictive-singing-regression-analysis\Dataset`
- 期望的数据路径为：
  - `Dataset\Chest new0206\Chest new0206`
  - 若上层目录下仅存在 `Dataset\Chest new0206`，也可以自动兼容

输出目录固定为：

- `Predictive-singing-regression-analysis\Extract\ExtractOutput\Dataset\Chest new0206`

每个特征都会写入单独的子目录（Jitter、Shimmer、H1H2、Hnr、QValue、SpectralSlope、LowFreqEnergyRatio、HighFreqNoiseRatio、Cpp），每个音频对应一个 CSV。

## Extract_NewFeatures.py 提取的特征与逻辑

该脚本对每个 WAV 文件提取 9 组时序特征，并保存为一维 CSV。文件名与原音频保持一致，仅后缀改为 `.csv`。

### 1) Jitter

函数：`extract_jitter`

逻辑：

- 优先使用 `parselmouth`（Praat 标准实现）
- 将音频转换为 Praat `PointProcess (periodic, cc)`（`pitch_floor=65 Hz`, `pitch_ceiling=1000 Hz`）
- 调用 `praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)` 计算标准 Jitter(Local)
- 输出为单值（标量），保存为一维 CSV（覆盖已有同名文件）
- 若未安装 `parselmouth`，则回退到 `librosa.pyin` 基于周期差的近似计算

### 2) Shimmer

函数：`extract_shimmer`

逻辑：

- 用 `librosa.pyin` 得到 F0 序列
- 用 `librosa.feature.rms` 计算每帧 RMS 幅度
- 对齐 F0 与 RMS 的长度
- 只保留 F0 有效的帧
- 计算相邻帧 RMS 幅度差 `|A(i+1) - A(i)|`
- 用平均幅度归一化，得到幅度抖动序列

### 3) H1H2

函数：`extract_h1h2`

逻辑：

- 用 `librosa.pyin` 得到每帧 F0
- 做 STFT 得到幅度谱 `|S|`
- 对每帧：
  - 计算基频对应的谱线 bin（H1）和二次谐波 bin（H2）
  - 将两者幅度转为 dB
  - 计算 `H1(dB) - H2(dB)`
- 形成 H1H2 序列

### 4) HNR

函数：`extract_hnr`

逻辑：

- 如果安装了 `parselmouth`：
  - 调用 Praat 的 `to_harmonicity` 获得 HNR 序列
- 否则使用近似方法：
  - 用 `librosa.effects.harmonic` 分离谐波分量
  - 噪声分量 = 原信号 - 谐波分量
  - 分别计算谐波与噪声的 RMS
  - 计算 `10 * log10( (RMS_h^2) / (RMS_n^2 + 1e-12) )`

### 5) QValue

函数：`extract_q_values`

逻辑：

- 对每帧 STFT 幅度谱找到最大峰值频率 `f0`
- 在谱线上搜索半功率点（幅度降到峰值的 `1/sqrt(2)`）
- 带宽 `BW = f_right - f_left`
- 计算 `Q = f0 / BW`

### 6) SpectralSlope

函数：`extract_spectral_slope`

逻辑：

- 计算每帧 STFT 幅度谱 `|S|`
- 取对数幅度 `log10(|S|)`
- 对频率轴做一次线性拟合，斜率即为谱斜率

### 7) LowFreqEnergyRatio

函数：`extract_low_freq_energy_ratio`

逻辑：

- 计算每帧功率谱 `|S|^2`
- 统计 0–500 Hz 的能量
- 统计 0–1000 Hz 的能量
- 计算 `low_energy / total_energy`

### 8) HighFreqNoiseRatio

函数：`extract_high_freq_noise_ratio`

逻辑：

- 用 `librosa.effects.harmonic` 得到谐波分量
- 噪声分量 = 原信号 - 谐波分量
- 对噪声做功率谱
- 统计高频能量（频率 >= max(4000Hz, 0.45*采样率)）
- 计算 `high_energy / total_energy`

### 9) CPP

函数：`extract_cpp`

逻辑：

- 对每帧做 STFT 幅度谱
- 对幅度取对数后做实数倒谱（IRFFT）
- 在 1/400s 到 1/60s 的倒谱区间寻找峰值
- 计算 `peak - mean(cepstrum_range)` 得到 CPP 序列
