import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import chi2, gaussian_kde

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

def build_pair_points(feature_x, feature_y, score_map, allowed_suffixes=None, drop_ids=None):
    x_vals = build_feature_medians(feature_x, score_map, allowed_suffixes, drop_ids)
    y_vals = build_feature_medians(feature_y, score_map, allowed_suffixes, drop_ids)
    xs = []
    ys = []
    scores = []
    keys = sorted(set(x_vals.keys()) & set(y_vals.keys()))
    for key in keys:
        x_val, _ = x_vals[key]
        y_val, _ = y_vals[key]
        if key in score_map:
            xs.append(x_val)
            ys.append(y_val)
            scores.append(score_map[key])
    if not xs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(scores, dtype=np.int32)

def draw_panel(fig, left, bottom, width, height, xs, ys, scores, xlabel, ylabel, title, xlim=None, ylim=None):
    if xs.size == 0:
        return
    spacing = 0.01
    main_w = width * 0.70
    main_h = height * 0.70
    margin_w = width * 0.18
    margin_h = height * 0.18
    ax_main = fig.add_axes([left, bottom, main_w, main_h])
    ax_top = fig.add_axes([left, bottom + main_h + spacing, main_w, margin_h], sharex=ax_main)
    ax_right = fig.add_axes([left + main_w + spacing, bottom, margin_w, main_h], sharey=ax_main)
    if xlim is None:
        xlim = compute_axis_limits(xs)
    if ylim is None:
        ylim = compute_axis_limits(ys)
    if xlim is not None:
        ax_main.set_xlim(xlim)
    if ylim is not None:
        ax_main.set_ylim(ylim)
    groups = sorted(np.unique(scores))
    colors = {1: "#ED949A", 3: "#B2A3DD", 5: "#96CCEA"}
    legend_elements = []
    bins = 25
    x_ref = np.asarray(xlim if xlim is not None else (float(np.min(xs)), float(np.max(xs))), dtype=np.float32)
    y_ref = np.asarray(ylim if ylim is not None else (float(np.min(ys)), float(np.max(ys))), dtype=np.float32)
    if np.isclose(x_ref[0], x_ref[1]):
        x_ref[1] = x_ref[0] + 1e-6
    if np.isclose(y_ref[0], y_ref[1]):
        y_ref[1] = y_ref[0] + 1e-6
    x_bins = np.linspace(float(x_ref[0]), float(x_ref[1]), bins + 1)
    y_bins = np.linspace(float(y_ref[0]), float(y_ref[1]), bins + 1)
    for g in groups:
        mask = scores == g
        xg = xs[mask]
        yg = ys[mask]
        color = colors.get(int(g), "#9E9E9E")
        ax_main.scatter(xg, yg, s=45, alpha=0.75, edgecolor=color, color=color, linewidth=1, marker="o")
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", label=f"Score {int(g)}", markerfacecolor=color, markeredgecolor=color, markersize=8)
        )
        if xg.size > 2:
            pts = np.column_stack([xg, yg]).astype(np.float64)
            center = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False)
            if np.all(np.isfinite(cov)):
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                order = eigenvals.argsort()[::-1]
                e0 = max(float(eigenvals[order[0]]), 0.0)
                e1 = max(float(eigenvals[order[1]]), 0.0)
                scale = 2 * np.sqrt(chi2.ppf(0.95, 2))
                width_e = scale * np.sqrt(e0)
                height_e = scale * np.sqrt(e1)
                angle = np.degrees(np.arctan2(*eigenvecs[:, order[0]][::-1]))
                if width_e > 0 and height_e > 0:
                    ellipse = mpatches.Ellipse(
                        xy=center,
                        width=width_e,
                        height=height_e,
                        angle=angle,
                        facecolor=color,
                        alpha=0.15,
                        edgecolor=color,
                        linestyle="--",
                        linewidth=1.5
                    )
                    ax_main.add_patch(ellipse)
        ax_top.hist(xg, bins=x_bins, alpha=0.35, color=color, density=True)
        ax_right.hist(yg, bins=y_bins, orientation="horizontal", alpha=0.35, color=color, density=True)
        if xg.size > 1:
            try:
                kde_x = gaussian_kde(xg.astype(np.float64))
                x_plot = np.linspace(float(x_ref[0]), float(x_ref[1]), 200)
                ax_top.plot(x_plot, kde_x(x_plot), linestyle="--", color=color, linewidth=1.5)
            except Exception:
                pass
        if yg.size > 1:
            try:
                kde_y = gaussian_kde(yg.astype(np.float64))
                y_plot = np.linspace(float(y_ref[0]), float(y_ref[1]), 200)
                ax_right.plot(kde_y(y_plot), y_plot, linestyle="--", color=color, linewidth=1.5)
            except Exception:
                pass
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.legend(handles=legend_elements, fontsize=9, frameon=False)
    ax_main.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)

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

def save_triplet_figures(score_map):
    drop_id = get_b1_max_id(jitter_feature, score_map)
    drop_ids = {drop_id} if drop_id else None
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            feature_x = FEATURES[i]
            feature_y = FEATURES[j]
            if allow_features and feature_x["name"] not in allow_features and feature_y["name"] not in allow_features:
                continue
            use_drop = drop_ids if "Jitter" in (feature_x["name"], feature_y["name"]) else None
            a1_xs, a1_ys, a1_scores = build_pair_points(feature_x, feature_y, score_map, {"A", "1"}, use_drop)
            b1_xs, b1_ys, b1_scores = build_pair_points(feature_x, feature_y, score_map, {"B", "1"}, use_drop)
            all_xs, all_ys, all_scores = build_pair_points(feature_x, feature_y, score_map, None, use_drop)
            fig = plt.figure(figsize=(18, 6), dpi=150)
            panel_width = 0.30
            panel_height = 0.80
            bottom = 0.12
            draw_panel(fig, 0.03, bottom, panel_width, panel_height, a1_xs, a1_ys, a1_scores, feature_x["label"], feature_y["label"], "A1")
            draw_panel(fig, 0.35, bottom, panel_width, panel_height, b1_xs, b1_ys, b1_scores, feature_x["label"], feature_y["label"], "B1")
            draw_panel(fig, 0.67, bottom, panel_width, panel_height, all_xs, all_ys, all_scores, feature_x["label"], feature_y["label"], "ALL")
            fig.suptitle(f"{TECHNIQUE} - {feature_x['name']} vs {feature_y['name']}", fontsize=12)
            out_path = os.path.join(output_dir, f"{feature_x['name']}_vs_{feature_y['name']}_A1_B1_ALL.png")
            fig.savefig(out_path)
            plt.close(fig)

score_map = select_score_map(build_score_maps(load_score_matrix(score_path)))
save_triplet_figures(score_map)
