import os
import re
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
data_root = os.path.join(repo_root, "Extract", "ExtractOutput", "Dataset", "Chest new0206")
output_dir = os.path.join(here, "A1_B1_ALL")
os.makedirs(output_dir, exist_ok=True)

TECHNIQUE = "chest"

features = [
    {
        "name": "Jitter",
        "feature_dir": os.path.join(data_root, "Jitter"),
        "ylabel": "Jitter Median"
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_root, "Shimmer"),
        "ylabel": "Shimmer Median"
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_root, "H1H2"),
        "ylabel": "H1H2 Median"
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_root, "Hnr"),
        "ylabel": "HNR Median"
    },
    {
        "name": "QValue",
        "feature_dir": os.path.join(data_root, "QValue"),
        "ylabel": "QValue Median"
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_root, "SpectralSlope"),
        "ylabel": "SpectralSlope Median"
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_root, "LowFreqEnergyRatio"),
        "ylabel": "LowFreqEnergyRatio Median"
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_root, "HighFreqNoiseRatio"),
        "ylabel": "HighFreqNoiseRatio Median"
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_root, "Cpp"),
        "ylabel": "CPP Median"
    }
]

def parse_label(filename):
    base = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return None

def parse_suffix_type(filename):
    base = os.path.splitext(filename)[0]
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
    return None

def parse_pitch_digit(filename):
    base = os.path.splitext(filename)[0]
    match = re.search(r"([A-Ga-g])(\d)", base)
    if match:
        return int(match.group(2))
    return None

def load_series(path):
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        return None
    if data is None or np.size(data) == 0:
        return None
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr

def build_median_points(feature_dir, allowed_suffixes=None):
    xs = []
    ys = []
    pitches = []
    if not os.path.isdir(feature_dir):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        label = parse_label(f)
        if label is None:
            continue
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        series = load_series(os.path.join(feature_dir, f))
        if series is None:
            continue
        pitch = parse_pitch_digit(f)
        if pitch is None:
            continue
        xs.append(label)
        ys.append(float(np.median(series)))
        pitches.append(pitch)
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32), np.asarray(pitches, dtype=np.int32)

def scatter_subplot(ax, scores, medians, pitches, ylabel=None, title=None):
    if scores.size == 0:
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        return
    rng = np.random.default_rng(42)
    xs = scores.astype(np.float32) + rng.uniform(-0.12, 0.12, size=scores.shape[0]).astype(np.float32)
    style_map = {
        3: {"color": "#1f77b4", "marker": "^"},
        4: {"color": "#2ca02c", "marker": "s"},
        5: {"color": "#ff7f0e", "marker": "o"}
    }
    for pitch_val, style in style_map.items():
        mask = pitches == pitch_val
        if np.any(mask):
            ax.scatter(xs[mask], medians[mask], s=18, alpha=0.8, color=style["color"], marker=style["marker"])
    uniq = sorted(list(set(scores.tolist())))
    ax.set_xticks(uniq)
    ax.set_xticklabels([str(u) for u in uniq])
    ax.set_xlabel("Score")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

def save_feature_triplets():
    for feature in features:
        a1_scores, a1_medians, a1_pitches = build_median_points(feature["feature_dir"], {"A", "1"})
        b1_scores, b1_medians, b1_pitches = build_median_points(feature["feature_dir"], {"B", "1"})
        all_scores, all_medians, all_pitches = build_median_points(feature["feature_dir"], None)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, sharey=False)
        for ax, (scores, medians, pitches, title) in zip(
            axes,
            [
                (a1_scores, a1_medians, a1_pitches, "A1"),
                (b1_scores, b1_medians, b1_pitches, "B1"),
                (all_scores, all_medians, all_pitches, "ALL")
            ],
        ):
            scatter_subplot(ax, scores, medians, pitches, ylabel=feature["ylabel"], title=title)
        fig.suptitle(f"{TECHNIQUE} - {feature['name']}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(output_dir, f"{feature['name']}_A1_B1_ALL.png")
        fig.savefig(out_path)
        plt.close(fig)

save_feature_triplets()
