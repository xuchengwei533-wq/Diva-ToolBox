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
        "ylabel": "Jitter Median (a.u.)"
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_root, "Shimmer"),
        "ylabel": "Shimmer Median (a.u.)"
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_root, "H1H2"),
        "ylabel": "H1H2 Median (dB)"
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_root, "Hnr"),
        "ylabel": "HNR Median (dB)"
    },
    {
        "name": "Q1",
        "feature_dir": os.path.join(data_root, "Q1"),
        "ylabel": "Q1 Median (a.u.)"
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_root, "SpectralSlope"),
        "ylabel": "SpectralSlope Median (a.u./Hz)"
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_root, "LowFreqEnergyRatio"),
        "ylabel": "LowFreqEnergyRatio Median (a.u.)"
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_root, "HighFreqNoiseRatio"),
        "ylabel": "HighFreqNoiseRatio Median (a.u.)"
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_root, "Cpp"),
        "ylabel": "CPP Median (dB)"
    },
    {
        "name": "FormantF1",
        "feature_dir": os.path.join(data_root, "Formant"),
        "ylabel": "Formant F1 Median (Hz)",
        "col": 0
    },
    {
        "name": "FormantF2",
        "feature_dir": os.path.join(data_root, "Formant"),
        "ylabel": "Formant F2 Median (Hz)",
        "col": 1
    },
    {
        "name": "FormantF3",
        "feature_dir": os.path.join(data_root, "Formant"),
        "ylabel": "Formant F3 Median (Hz)",
        "col": 2
    },
    {
        "name": "AbsH1MinusF1",
        "feature_dir": os.path.join(data_root, "H1H2"),
        "paired_feature_dir": os.path.join(data_root, "Formant"),
        "paired_col": 0,
        "op": "abs_diff",
        "ylabel": "|H1 - F1| Median (a.u.)"
    }
]

plot_features = os.environ.get("PLOT_FEATURES")
if plot_features:
    allow = {item.strip() for item in plot_features.split(",") if item.strip()}
    if allow:
        features = [f for f in features if f["name"] in allow]

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

def load_series(path, col=None):
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        return None
    if data is None or np.size(data) == 0:
        return None
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        if col is not None and col > 0:
            return None
        arr = arr.reshape(-1)
    else:
        if col is not None:
            if arr.shape[1] <= col:
                return None
            arr = arr[:, col]
        else:
            arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr

def load_abs_diff_series(path_a, path_b, col_b=0):
    series_a = load_series(path_a)
    series_b = load_series(path_b, col=col_b)
    if series_a is None or series_b is None:
        return None
    min_len = min(series_a.shape[0], series_b.shape[0])
    if min_len <= 0:
        return None
    values = np.abs(series_a[:min_len] - series_b[:min_len])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return values.astype(np.float32)

def build_median_points(feature, allowed_suffixes=None):
    xs = []
    ys = []
    pitches = []
    feature_dir = feature["feature_dir"]
    col = feature.get("col")
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
        if feature.get("op") == "abs_diff":
            series = load_abs_diff_series(
                os.path.join(feature_dir, f),
                os.path.join(feature["paired_feature_dir"], f),
                col_b=feature.get("paired_col", 0),
            )
        else:
            series = load_series(os.path.join(feature_dir, f), col=col)
        if series is None:
            continue
        pitch = parse_pitch_digit(f)
        if pitch is None:
            continue
        xs.append(label)
        ys.append(float(np.median(series)))
        pitches.append(pitch)
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32), np.asarray(pitches, dtype=np.int32)

def build_median_records(feature, allowed_suffixes=None):
    xs = []
    ys = []
    pitches = []
    names = []
    feature_dir = feature["feature_dir"]
    col = feature.get("col")
    if not os.path.isdir(feature_dir):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.int32), np.array([], dtype=object)
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        label = parse_label(f)
        if label is None:
            continue
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        if feature.get("op") == "abs_diff":
            series = load_abs_diff_series(
                os.path.join(feature_dir, f),
                os.path.join(feature["paired_feature_dir"], f),
                col_b=feature.get("paired_col", 0),
            )
        else:
            series = load_series(os.path.join(feature_dir, f), col=col)
        if series is None:
            continue
        pitch = parse_pitch_digit(f)
        if pitch is None:
            continue
        xs.append(label)
        ys.append(float(np.median(series)))
        pitches.append(pitch)
        names.append(f)
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(pitches, dtype=np.int32),
        np.asarray(names, dtype=object),
    )

def drop_by_name(scores, medians, pitches, names, drop_name):
    if drop_name is None or names.size == 0:
        return scores, medians, pitches, names
    mask = names != drop_name
    return scores[mask], medians[mask], pitches[mask], names[mask]

def scatter_subplot(ax, scores, medians, pitches, ylabel=None, title=None, ylim=None):
    if scores.size == 0:
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        return
    rng = np.random.default_rng(42)
    xs = scores.astype(np.float32) + rng.uniform(-0.12, 0.12, size=scores.shape[0]).astype(np.float32)
    score_style_map = {
        1: {"color": "#ED949A", "marker": "o"},
        3: {"color": "#B2A3DD", "marker": "o"},
        5: {"color": "#96CCEA", "marker": "o"}
    }
    for score_val, style in score_style_map.items():
        mask = scores == score_val
        if np.any(mask):
            ax.scatter(xs[mask], medians[mask], s=18, alpha=0.8, color=style["color"], marker=style["marker"])
    known_scores = set(score_style_map.keys())
    for score_val in sorted(set(scores.tolist())):
        if score_val in known_scores:
            continue
        mask = scores == score_val
        if np.any(mask):
            ax.scatter(xs[mask], medians[mask], s=18, alpha=0.8, color="#9E9E9E", marker="o")
    uniq = sorted(list(set(scores.tolist())))
    ax.set_xticks(uniq)
    ax.set_xticklabels([str(u) for u in uniq])
    ax.set_xlabel("Score")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

def compute_ylim_from_medians(medians):
    if medians.size == 0:
        return None
    y_min = float(np.min(medians))
    y_max = float(np.max(medians))
    if np.isclose(y_min, y_max):
        pad = 0.05 if y_min == 0 else abs(y_min) * 0.05
    else:
        pad = (y_max - y_min) * 0.02
    return y_min - pad, y_max + pad

def compute_feature_ylim(feature, drop_name=None):
    scores, medians, pitches, names = build_median_records(feature, None)
    scores, medians, pitches, names = drop_by_name(scores, medians, pitches, names, drop_name)
    return compute_ylim_from_medians(medians)

def get_b1_max_name(feature):
    scores, medians, pitches, names = build_median_records(feature, {"B", "1"})
    if medians.size == 0:
        return None
    idx = int(np.argmax(medians))
    return names[idx]

def save_feature_triplets():
    for feature in features:
        drop_name = None
        if feature["name"] == "Jitter":
            drop_name = get_b1_max_name(feature)
        a1_scores, a1_medians, a1_pitches, a1_names = build_median_records(feature, {"A", "1"})
        b1_scores, b1_medians, b1_pitches, b1_names = build_median_records(feature, {"B", "1"})
        all_scores, all_medians, all_pitches, all_names = build_median_records(feature, None)
        a1_scores, a1_medians, a1_pitches, a1_names = drop_by_name(a1_scores, a1_medians, a1_pitches, a1_names, drop_name)
        b1_scores, b1_medians, b1_pitches, b1_names = drop_by_name(b1_scores, b1_medians, b1_pitches, b1_names, drop_name)
        all_scores, all_medians, all_pitches, all_names = drop_by_name(all_scores, all_medians, all_pitches, all_names, drop_name)
        ylim = compute_feature_ylim(feature, drop_name=drop_name)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150, sharey=True)
        for ax, (scores, medians, pitches, title) in zip(
            axes,
            [
                (a1_scores, a1_medians, a1_pitches, "A1"),
                (b1_scores, b1_medians, b1_pitches, "B1"),
                (all_scores, all_medians, all_pitches, "ALL")
            ],
        ):
            scatter_subplot(ax, scores, medians, pitches, ylabel=feature["ylabel"], title=title, ylim=ylim)
        fig.suptitle(f"{TECHNIQUE} - {feature['name']}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(output_dir, f"{feature['name']}_A1_B1_ALL.png")
        fig.savefig(out_path)
        plt.close(fig)

save_feature_triplets()
