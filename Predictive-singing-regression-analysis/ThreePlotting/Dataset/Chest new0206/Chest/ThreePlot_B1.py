import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
if not os.path.exists(os.path.join(repo_root, "README.md")):
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
data_root = os.path.join(repo_root, "Extract", "ExtractOutput", "Dataset", "Chest new0206")
data_dir = data_root
output_dir = os.path.join(here, "B1")
os.makedirs(output_dir, exist_ok=True)
score_path = os.path.join(repo_root, "打分Chest_new0206_scores_matrix.xlsx")

TECHNIQUE = "chest"

def get_feature_dir(data_dir, name, fallback_name=None):
    if fallback_name is None:
        fallback_name = name
    d1 = os.path.join(data_dir, name + "Output")
    if os.path.isdir(d1):
        return d1
    d2 = os.path.join(data_dir, fallback_name)
    if os.path.isdir(d2):
        return d2
    return d1

FEATURES = [
    {"name": "Jitter", "dir": get_feature_dir(data_dir, "Jitter"), "label": "Jitter"},
    {"name": "Shimmer", "dir": get_feature_dir(data_dir, "Shimmer"), "label": "Shimmer"},
    {"name": "H1H2", "dir": get_feature_dir(data_dir, "H1H2"), "label": "H1H2 (dB)"},
    {"name": "HNR", "dir": get_feature_dir(data_dir, "HNR", "Hnr"), "label": "HNR"},
    {"name": "QValue", "dir": get_feature_dir(data_dir, "QValue"), "label": "QValue"},
    {"name": "SpectralSlope", "dir": get_feature_dir(data_dir, "SpectralSlope"), "label": "SpectralSlope"},
    {"name": "LowFreqEnergyRatio", "dir": get_feature_dir(data_dir, "LowFreqEnergyRatio"), "label": "LowFreqEnergyRatio"},
    {"name": "HighFreqNoiseRatio", "dir": get_feature_dir(data_dir, "HighFreqNoiseRatio"), "label": "HighFreqNoiseRatio"},
    {"name": "CPP", "dir": get_feature_dir(data_dir, "CPP", "Cpp"), "label": "CPP (dB)"}
]

plot_features = os.environ.get("PLOT_FEATURES")
if plot_features:
    allow_features = {item.strip() for item in plot_features.split(",") if item.strip()}
else:
    allow_features = None

jitter_feature = next((feature for feature in FEATURES if feature["name"] == "Jitter"), None)

def normalize_id(value):
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    s = re.sub(r"\.(wav|csv|xlsx)$", "", s, flags=re.IGNORECASE)
    return s.lower()

def load_score_matrix(path):
    df = pd.read_excel(path)
    if df.empty:
        return df
    mask_delete = df.apply(lambda row: row.astype(str).str.contains("删除", na=False).any(), axis=1)
    df = df[~mask_delete].copy()
    df = df.dropna(axis=0, how="all")
    return df

def detect_id_column(df):
    keywords = ["文件", "filename", "file", "name", "音频", "sample", "id"]
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            return col
    return df.columns[0]

def build_score_maps(df):
    id_col = detect_id_column(df)
    id_series = df[id_col].astype(str)
    id_norm = id_series.apply(normalize_id)
    score_cols = [c for c in df.columns if c != id_col]
    maps = {}
    for col in score_cols:
        scores = pd.to_numeric(df[col], errors="coerce")
        if scores.notna().sum() == 0:
            continue
        mapping = {}
        for sid, score in zip(id_norm, scores):
            if not sid:
                continue
            if pd.isna(score):
                continue
            mapping[sid] = int(round(float(score)))
        if mapping:
            maps[str(col)] = mapping
    return maps

def select_score_map(score_maps):
    if not score_maps:
        return {}
    return max(score_maps.values(), key=lambda m: len(m))

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

def build_feature_medians(feature, score_map, allowed_suffixes=None, drop_ids=None):
    vals = {}
    if not os.path.isdir(feature["dir"]):
        print(f"❌ DEBUG: Directory not found: {feature['dir']}")
        return vals
    files = [f for f in os.listdir(feature["dir"]) if f.lower().endswith(".csv")]
    files.sort()
    print(f"ℹ️ DEBUG: Found {len(files)} csv files in {feature['dir']}")

    match_count = 0
    skipped_suffix = 0
    skipped_id = 0
    skipped_pitch = 0
    skipped_data = 0

    for f in files:
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            skipped_suffix += 1
            continue
        base_id = normalize_id(os.path.splitext(f)[0])
        if drop_ids is not None and base_id in drop_ids:
            skipped_id += 1
            continue
        if base_id not in score_map:
            skipped_id += 1
            continue
        series = load_series(os.path.join(feature["dir"], f))
        if series is None:
            skipped_data += 1
            continue
        pitch = parse_pitch_digit(f)
        if pitch is None:
            skipped_pitch += 1
            continue
        vals[base_id] = (float(np.median(series)), pitch)
        match_count += 1
    
    print(f"✅ DEBUG: {feature['name']} - Matched: {match_count}, Skipped Suffix: {skipped_suffix}, Skipped ID: {skipped_id}, Skipped Pitch: {skipped_pitch}, Skipped Data: {skipped_data}")
    return vals

def get_b1_max_id(jitter_feature, score_map):
    if jitter_feature is None:
        return None
    vals = build_feature_medians(jitter_feature, score_map, {"B", "1"})
    if not vals:
        return None
    return max(vals.items(), key=lambda item: item[1][0])[0]

def build_triplet_points(feature_x, feature_y, feature_z, score_map, allowed_suffixes=None, drop_ids=None):
    x_vals = build_feature_medians(feature_x, score_map, allowed_suffixes, drop_ids)
    y_vals = build_feature_medians(feature_y, score_map, allowed_suffixes, drop_ids)
    z_vals = build_feature_medians(feature_z, score_map, allowed_suffixes, drop_ids)
    xs = []
    ys = []
    zs = []
    scores = []
    keys = sorted(set(x_vals.keys()) & set(y_vals.keys()) & set(z_vals.keys()))
    for key in keys:
        x_val, _ = x_vals[key]
        y_val, _ = y_vals[key]
        z_val, _ = z_vals[key]
        if key in score_map:
            xs.append(x_val)
            ys.append(y_val)
            zs.append(z_val)
            scores.append(score_map[key])
    if not xs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(zs, dtype=np.float32), np.asarray(scores, dtype=np.int32)

def scatter_feature_triplet(ax, xs, ys, zs, scores, title, xlabel, ylabel, zlabel, xlim=None, ylim=None, zlim=None, point_size=30):
    if xs.size == 0:
        ax.set_axis_off()
        ax.set_title(title)
        return
    rng = np.random.default_rng(42)
    x_span = float(np.max(xs) - np.min(xs))
    y_span = float(np.max(ys) - np.min(ys))
    z_span = float(np.max(zs) - np.min(zs))
    x_jitter = 0.0 if x_span == 0 else x_span * 0.02
    y_jitter = 0.0 if y_span == 0 else y_span * 0.02
    z_jitter = 0.0 if z_span == 0 else z_span * 0.02
    xs_j = xs + rng.uniform(-x_jitter, x_jitter, size=xs.shape[0]).astype(np.float32)
    ys_j = ys + rng.uniform(-y_jitter, y_jitter, size=ys.shape[0]).astype(np.float32)
    zs_j = zs + rng.uniform(-z_jitter, z_jitter, size=zs.shape[0]).astype(np.float32)
    style_map = {
        1: {"color": "#1f77b4", "label": "1"},
        3: {"color": "#2ca02c", "label": "3"},
        5: {"color": "#ff7f0e", "label": "5"}
    }
    for score_val in [1, 3, 5]:
        mask = scores == score_val
        if np.any(mask):
            style = style_map[score_val]
            ax.scatter(xs_j[mask], ys_j[mask], zs_j[mask], s=point_size, alpha=0.8, color=style["color"], marker="o", label=style["label"])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)

def compute_axis_limits(values):
    if values.size == 0:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    span = vmax - vmin
    if vmin == vmax:
        pad = 0.01 if vmin == 0 else abs(vmin) * 0.1
        return (vmin - pad, vmax + pad)
    pad = max(span * 0.1, abs(vmax) * 0.02, abs(vmin) * 0.02)
    return (vmin - pad, vmax + pad)

def get_triplet_limits(feature_x, feature_y, feature_z, score_map, drop_ids):
    xs_all, ys_all, zs_all, _ = build_triplet_points(feature_x, feature_y, feature_z, score_map, None, drop_ids)
    xlim = compute_axis_limits(xs_all)
    ylim = compute_axis_limits(ys_all)
    zlim = compute_axis_limits(zs_all)
    return xlim, ylim, zlim

def save_triplet_figures(suffix_tag, allowed_suffixes, score_map):
    drop_id = get_b1_max_id(jitter_feature, score_map)
    drop_ids = {drop_id} if drop_id else None
    count = 0
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            for k in range(j + 1, len(FEATURES)):
                feature_x = FEATURES[i]
                feature_y = FEATURES[j]
                feature_z = FEATURES[k]
                if allow_features and feature_x["name"] not in allow_features and feature_y["name"] not in allow_features and feature_z["name"] not in allow_features:
                    continue
                use_drop = drop_ids if "Jitter" in (feature_x["name"], feature_y["name"], feature_z["name"]) else None
                xs, ys, zs, scores = build_triplet_points(feature_x, feature_y, feature_z, score_map, allowed_suffixes, use_drop)
                xlim, ylim, zlim = get_triplet_limits(feature_x, feature_y, feature_z, score_map, use_drop)
                fig = plt.figure(figsize=(6.5, 4.5), dpi=150)
                ax = fig.add_subplot(111, projection='3d')
                title = f"{TECHNIQUE} - {suffix_tag}\n{feature_x['name']} vs {feature_y['name']} vs {feature_z['name']}"
                point_size = 22 if "Jitter" in (feature_x["name"], feature_y["name"], feature_z["name"]) else 30
                scatter_feature_triplet(ax, xs, ys, zs, scores, title, feature_x["label"], feature_y["label"], feature_z["label"], xlim, ylim, zlim, point_size)
                fig.tight_layout()
                out_path = os.path.join(output_dir, f"{feature_x['name']}_vs_{feature_y['name']}_vs_{feature_z['name']}_{suffix_tag}.png")
                fig.savefig(out_path)
                plt.close(fig)
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count} plots...")

score_map = select_score_map(build_score_maps(load_score_matrix(score_path)))
save_triplet_figures("B1", {"B", "1"}, score_map)
