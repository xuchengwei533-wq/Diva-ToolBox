import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
if not os.path.exists(os.path.join(repo_root, "README.md")):
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
data_root = os.path.join(repo_root, "Extract", "ExtractOutput", "Dataset", "Chest new0206")
data_dir = data_root
output_dir = os.path.join(here, "A1_B1_ALL")
os.makedirs(output_dir, exist_ok=True)
score_path = os.path.join(repo_root, "打分Chest_new0206_scores_matrix.xlsx")

TECHNIQUE = "chest"

matplotlib.rcParams["font.family"] = "Times New Roman"

PLOT_CONFIG = {
    "dpi": 600,
    "figsize": (18, 6),
    "scatter_colors": ["#ED949A", "#B2A3DD", "#96CCEA", "#FFDD8E"],
    "main_scatter_size": 40,
    "main_scatter_alpha": 1,
    "main_scatter_edgecolors": "black",
    "main_scatter_linewidth": 0.5,
    "margin": 0.2,
    "elevation": 33,
    "azimuth": 65
}

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
    {"name": "Jitter", "dir": get_feature_dir(data_dir, "Jitter"), "label": "Jitter (a.u.)"},
    {"name": "Shimmer", "dir": get_feature_dir(data_dir, "Shimmer"), "label": "Shimmer (a.u.)"},
    {"name": "H1H2", "dir": get_feature_dir(data_dir, "H1H2"), "label": "H1H2 (dB)"},
    {"name": "HNR", "dir": get_feature_dir(data_dir, "HNR", "Hnr"), "label": "HNR (dB)"},
    {"name": "Q1", "dir": get_feature_dir(data_dir, "Q1"), "label": "Q1 (a.u.)"},
    {"name": "SpectralSlope", "dir": get_feature_dir(data_dir, "SpectralSlope"), "label": "SpectralSlope (a.u./Hz)"},
    {"name": "LowFreqEnergyRatio", "dir": get_feature_dir(data_dir, "LowFreqEnergyRatio"), "label": "LowFreqEnergyRatio (a.u.)"},
    {"name": "HighFreqNoiseRatio", "dir": get_feature_dir(data_dir, "HighFreqNoiseRatio"), "label": "HighFreqNoiseRatio (a.u.)"},
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

def setup_3d_axes(ax, xs, ys, zs, xlabel, ylabel, zlabel):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("k")
    ax.yaxis.pane.set_edgecolor("k")
    ax.zaxis.pane.set_edgecolor("k")
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray", alpha=0.7)
    ax.xaxis._axinfo["grid"].update({"linestyle": "--", "color": "lightgray", "linewidth": 0.5})
    ax.yaxis._axinfo["grid"].update({"linestyle": "--", "color": "lightgray", "linewidth": 0.5})
    ax.zaxis._axinfo["grid"].update({"linestyle": "--", "color": "lightgray", "linewidth": 0.5})
    margin = PLOT_CONFIG["margin"]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min
    x_pad = x_span * margin if x_span > 0 else max(abs(x_max), 1.0) * 0.1
    y_pad = y_span * margin if y_span > 0 else max(abs(y_max), 1.0) * 0.1
    z_pad = z_span * margin if z_span > 0 else max(abs(z_max), 1.0) * 0.1
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([y_min - y_pad, y_max + y_pad])
    ax.set_zlim([z_min - z_pad, z_max + z_pad])
    ax.set_xlabel(xlabel, weight="bold")
    ax.set_ylabel(ylabel, weight="bold")
    ax.set_zlabel(zlabel, weight="bold")
    ax.view_init(elev=PLOT_CONFIG["elevation"], azim=PLOT_CONFIG["azimuth"])

def draw_confidence_ellipsoid(ax, xs, ys, zs, color, alpha=0.18):
    from scipy.stats import chi2
    data = np.vstack((xs, ys, zs)).astype(np.float64)
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    if not np.all(np.isfinite(cov)):
        return
    scale = np.sqrt(chi2.ppf(0.95, df=3))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    radii = scale * np.sqrt(eigvals)
    if np.any(radii <= 0):
        return
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1)
    ellipsoid = ellipsoid @ eigvecs.T
    x_e = ellipsoid[..., 0] + mean[0]
    y_e = ellipsoid[..., 1] + mean[1]
    z_e = ellipsoid[..., 2] + mean[2]
    ax.plot_surface(x_e, y_e, z_e, color=color, alpha=alpha, linewidth=0, shade=True, zorder=5)

def scatter_feature_triplet(ax, xs, ys, zs, scores, title, xlabel, ylabel, zlabel):
    if xs.size == 0:
        ax.set_axis_off()
        ax.set_title(title)
        return
    setup_3d_axes(ax, xs, ys, zs, xlabel, ylabel, zlabel)
    unique_scores = sorted(np.unique(scores))
    for i, s in enumerate(unique_scores):
        mask = scores == s
        color = PLOT_CONFIG["scatter_colors"][i % len(PLOT_CONFIG["scatter_colors"])]
        xs_g = xs[mask]
        ys_g = ys[mask]
        zs_g = zs[mask]
        if xs_g.size > 5:
            draw_confidence_ellipsoid(ax, xs_g, ys_g, zs_g, color=color, alpha=0.15)
        ax.scatter(
            xs_g,
            ys_g,
            zs_g,
            color=color,
            label=str(s),
            s=PLOT_CONFIG["main_scatter_size"],
            alpha=PLOT_CONFIG["main_scatter_alpha"],
            edgecolors=PLOT_CONFIG["main_scatter_edgecolors"],
            linewidth=PLOT_CONFIG["main_scatter_linewidth"],
            zorder=30
        )
    ax.legend(frameon=False)

def save_triplet_figures(score_map):
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
                a1_xs, a1_ys, a1_zs, a1_scores = build_triplet_points(feature_x, feature_y, feature_z, score_map, {"A", "1"}, use_drop)
                b1_xs, b1_ys, b1_zs, b1_scores = build_triplet_points(feature_x, feature_y, feature_z, score_map, {"B", "1"}, use_drop)
                all_xs, all_ys, all_zs, all_scores = build_triplet_points(feature_x, feature_y, feature_z, score_map, None, use_drop)
                fig = plt.figure(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])
                ax1 = fig.add_subplot(131, projection="3d")
                scatter_feature_triplet(ax1, a1_xs, a1_ys, a1_zs, a1_scores, "A1", feature_x["label"], feature_y["label"], feature_z["label"])
                ax2 = fig.add_subplot(132, projection="3d")
                scatter_feature_triplet(ax2, b1_xs, b1_ys, b1_zs, b1_scores, "B1", feature_x["label"], feature_y["label"], feature_z["label"])
                ax3 = fig.add_subplot(133, projection="3d")
                scatter_feature_triplet(ax3, all_xs, all_ys, all_zs, all_scores, "ALL", feature_x["label"], feature_y["label"], feature_z["label"])
                fig.suptitle(f"{TECHNIQUE} - {feature_x['name']} vs {feature_y['name']} vs {feature_z['name']}", fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                out_path = os.path.join(output_dir, f"{feature_x['name']}_vs_{feature_y['name']}_vs_{feature_z['name']}_A1_B1_ALL.png")
                plt.savefig(out_path, dpi=PLOT_CONFIG["dpi"])
                plt.close()
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count} plots...")

score_map = select_score_map(build_score_maps(load_score_matrix(score_path)))
save_triplet_figures(score_map)
