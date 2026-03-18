import numpy as np
import librosa
import os
import time
try:
    import tqdm as tq
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False
try:
    import parselmouth as pm
    HAS_PM = True
except Exception:
    HAS_PM = False

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", ".."))
data_root = os.environ.get("PSRA_DATA_ROOT", os.path.join(repo_root, "Dataset"))
if not os.path.exists(os.path.join(data_root, "Chest new0206")):
    data_root = repo_root
directory = os.path.join(data_root, "Chest new0206", "Chest new0206")
if not os.path.exists(directory):
    directory = os.path.join(data_root, "Chest new0206")
Audio_path = directory
output_root = os.path.join(repo_root, "Extract", "ExtractOutput", "Dataset", "Chest new0206")
os.makedirs(output_root, exist_ok=True)

Formant_out = os.path.join(output_root, "Formant")
os.makedirs(Formant_out, exist_ok=True)

files = os.listdir(Audio_path)
wav_files = []
for f in files:
    if not f.lower().endswith(".wav"):
        continue
    in_path = os.path.join(Audio_path, f)
    if not os.path.isfile(in_path):
        continue
    wav_files.append(f)
wav_files.sort()

print(f"📁 输入目录: {Audio_path}")
print(f"📦 共检测到 WAV 文件: {len(wav_files)}")
print(f"📄 Formant 输出目录: {Formant_out}")
if not HAS_PM:
    print("⚠️ 未检测到 parselmouth，将使用 LPC 近似提取")
print("-" * 60)

def load_audio(file_path, target_sr=44100):
    if HAS_SF:
        audio_raw, original_sr = sf.read(file_path, dtype="float32", always_2d=False)
        if isinstance(original_sr, np.ndarray):
            original_sr = int(original_sr)
        if hasattr(audio_raw, "ndim") and audio_raw.ndim > 1:
            audio_raw = np.mean(audio_raw, axis=1)
    else:
        audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)
    if original_sr != target_sr:
        audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
    else:
        audio = audio_raw
    return audio, original_sr, target_sr

def extract_formants_parselmouth(audio, sr, hop_length=512, max_formants=5, max_formant_hz=5500.0, window_length=0.025, pre_emphasis=50.0):
    snd = pm.Sound(audio, sampling_frequency=sr)
    time_step = hop_length / float(sr)
    formant = pm.praat.call(snd, "To Formant (burg)", time_step, max_formants, max_formant_hz, window_length, pre_emphasis)
    n_frames = int(pm.praat.call(formant, "Get number of frames"))
    if n_frames <= 0:
        return None
    # 始终初始化为 (n_frames, 3) 确保维度一致
    values = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for i in range(n_frames):
        t = pm.praat.call(formant, "Get time from frame number", i + 1)
        for k in range(3):
            v = pm.praat.call(formant, "Get value at time", k + 1, t, "Hertz", "Linear")
            if v is None:
                continue
            v = float(v)
            if not np.isfinite(v) or v <= 0:
                continue
            values[i, k] = v
    return values

def extract_formants_lpc(audio, sr, hop_length=512, win_length=1024, lpc_order=None, max_formant_hz=5500.0):
    if lpc_order is None:
        lpc_order = int(sr / 1000) + 2
    if len(audio) < win_length:
        return None
    frames = librosa.util.frame(audio, frame_length=win_length, hop_length=hop_length).T
    window = np.hamming(win_length).astype(np.float32)
    values = np.full((frames.shape[0], 3), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        frame = frame.astype(np.float32) * window
        if not np.any(frame):
            continue
        try:
            a = librosa.lpc(frame, order=lpc_order)
        except Exception:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        if roots.size == 0:
            continue
        angs = np.angle(roots)
        freqs = angs * (sr / (2 * np.pi))
        bw = -0.5 * (sr / np.pi) * np.log(np.abs(roots))
        mask = (freqs > 90.0) & (freqs < max_formant_hz) & (bw > 0) & (bw < 4000)
        freqs = freqs[mask]
        if freqs.size == 0:
            continue
        freqs = np.sort(freqs)
        take = min(3, freqs.size)
        values[i, :take] = freqs[:take]
    return values

def extract_formants(audio, sr, hop_length=512):
    if HAS_PM:
        try:
            values = extract_formants_parselmouth(audio, sr, hop_length=hop_length)
            if values is not None:
                return values
        except Exception:
            pass
    return extract_formants_lpc(audio, sr, hop_length=hop_length)

if HAS_TQDM:
    pbar = tq.tqdm(wav_files, total=len(wav_files), desc="提取Formant", unit="file", dynamic_ncols=True)
    for idx, file in enumerate(pbar, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(Formant_out, out_name)
        if os.path.exists(out_path):
            pbar.set_postfix({"步骤": "跳过(已存在)", "文件": file})
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        pbar.set_postfix({"步骤": "提取Formant", "文件": file})
        values = extract_formants(audio, target_sr)
        if values is None:
            tq.tqdm.write(f"❌ 提取失败: {file}")
            continue
        np.savetxt(out_path, values, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        pbar.set_postfix({"步骤": f"完成({cost_s:.1f}s)", "文件": file})
        tq.tqdm.write(f"✅ [{idx}/{len(wav_files)}] {file} | Formant saved: {out_path}")
else:
    for idx, file in enumerate(wav_files, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(Formant_out, out_name)
        if os.path.exists(out_path):
            print(f"[{idx}/{len(wav_files)}] 跳过(已存在): {file}")
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        values = extract_formants(audio, target_sr)
        if values is None:
            print(f"[{idx}/{len(wav_files)}] 失败: {file}")
            continue
        np.savetxt(out_path, values, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        print(f"[{idx}/{len(wav_files)}] 完成({cost_s:.1f}s): {file}")

print("🎉 全部处理完成。")
