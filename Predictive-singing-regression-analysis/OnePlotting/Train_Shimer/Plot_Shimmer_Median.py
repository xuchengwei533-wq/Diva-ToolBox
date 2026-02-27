import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, ".."))
data_root = os.environ.get("PSRA_DATA_ROOT", os.path.join(repo_root, "Dataset"))
if not os.path.exists(os.path.join(data_root, "Chest new0206")):
    data_root = repo_root
output_root = os.environ.get("PSRA_OUTPUT_ROOT", repo_root)

data_dir = os.path.join(data_root, "Chest new0206", "Chest new0206")
if not os.path.exists(data_dir):
    data_dir = os.path.join(data_root, "Chest new0206")
output_dir = os.path.join(output_root, "Train_Shimer")
os.makedirs(output_dir, exist_ok=True)

feature_dir = os.path.join(data_dir, "ShimmerOutput")

def parse_label(filename):
    name = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[-1])

def load_series_csv(path):
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        return np.array(data, dtype=np.float32)
    except Exception:
        return None

files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
files.sort()

xs = []
ys = []
for f in files:
    label = parse_label(f)
    if label is None:
        continue
    path = os.path.join(feature_dir, f)
    series = load_series_csv(path)
    if series is None:
        continue
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 2:
        series = series.reshape(-1)
    series = series[np.isfinite(series)]
    if series.size == 0:
        continue
    median_val = float(np.median(series))
    xs.append(label)
    ys.append(median_val)

xs = np.asarray(xs, dtype=np.int32)
ys = np.asarray(ys, dtype=np.float32)

if xs.size == 0:
    raise RuntimeError("没有可用样本，请确认 ShimmerOutput 已生成")

plt.figure(figsize=(6, 4.5), dpi=150)
np.random.seed(42)
xs_jitter = xs + (np.random.rand(xs.size) - 0.5) * 0.15
plt.scatter(xs_jitter, ys, s=18, alpha=0.8)
plt.xticks([1, 3, 5], ["1", "3", "5"])
plt.xlabel("Score")
plt.ylabel("Shimmer Median")
plt.tight_layout()
out_path = os.path.join(output_dir, "Shimmer_median_by_score.png")
plt.savefig(out_path)
plt.close()
print(f"✅ 已保存：{out_path}")
