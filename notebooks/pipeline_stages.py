# Pipeline evolution on ONE recording: raw data -> feature vector

# Re-implements, on a single recording, exactly the steps of version1.ipynb
# (cut 1.5 s, stationary removal, gravity alignment Z-then-X, band-pass,
# 3 s windows with 30% overlap, features of one window) and draws one panel per step.
# It reads the RAW Sensor Logger folder, so it does not depend on the
# intermediate folders in ../preprocessing.

# Usage in the notebook: paste this cell after the windowing/feature cells and set
# RECORDING_DIR. Standalone: python pipeline_stages.py <recording_dir> [out_dir]

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, welch

RECORDING_DIR = "../raw_data/bumpy_stadio"   # <- change to any raw recording
OUT_DIR = "../figures"

# Same constants as the notebook
FS = 100
CUT_SEC = 1.5
ENERGY_THRESHOLD = 0.30        # stationary removal cell
ENERGY_WINDOW = 50
MIN_STATIONARY = 200
FILTER_CONFIG = {"acc": (10.0, 40.0, 4), "gyr": (5.0, 40.0, 4)}
WINDOW_SAMPLES = 300
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - 0.3))   # 210
AXES = ["x", "y", "z"]

C_SMOOTH, C_BUMPY = "#2a78d6", "#eb6834"
C_AXIS = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
C_REMOVED = "#d9d8d3"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def rotation_z_then_x(g):
    """Same decomposition as the notebook alignment cell: R @ g = (0, -|g|, 0)."""
    up = -g / np.linalg.norm(g)
    psi = np.arctan2(up[0], up[1])
    theta = np.arctan2(-up[2], np.hypot(up[0], up[1]))
    cp, sp, ct, st = np.cos(psi), np.sin(psi), np.cos(theta), np.sin(theta)
    Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return Rx @ Rz, np.degrees(theta), np.degrees(psi)


def bandpass(x, sensor):
    lo, hi, order = FILTER_CONFIG[sensor]
    coefficients = butter(order, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="ba")
    if not isinstance(coefficients, tuple) or len(coefficients) != 2:
        raise TypeError("Expected butter() to return numerator and denominator coefficients.")
    b, a = coefficients[0], coefficients[1]
    return filtfilt(b, a, x)


def run_stages(rec_dir, sign=1.0):
    """Returns a dict with the signal at every stage, plus sample counts."""
    raw = {s: pd.read_csv(os.path.join(rec_dir, f"{s}.csv"))
           for s in ["Accelerometer", "Gyroscope", "Gravity"]}
    n_raw = min(len(d) for d in raw.values())
    t_raw = raw["Accelerometer"]["seconds_elapsed"].to_numpy()[:n_raw]

    # 1) cut first/last 1.5 s
    c = int(CUT_SEC * FS)
    cut = {s: d.iloc[c:n_raw - c].reset_index(drop=True) for s, d in raw.items()}

    # 2) stationary removal (energy of |acc| below threshold for >= 2 s)
    acc = cut["Accelerometer"]
    mag2 = acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2
    energy = mag2.rolling(ENERGY_WINDOW).mean()
    stat = energy < ENERGY_THRESHOLD
    run_ids = (stat != stat.shift()).cumsum()
    stat = stat & (stat.groupby(run_ids).transform("size") >= MIN_STATIONARY)
    moving = {s: d[~stat.to_numpy()].reset_index(drop=True) for s, d in cut.items()}

    # 3) alignment with one constant rotation from the median gravity
    g_med = np.median(sign * moving["Gravity"][AXES].to_numpy(), axis=0)
    R, theta, psi = rotation_z_then_x(g_med)
    aligned = {}
    for s, d in moving.items():
        sg = sign if s in ("Accelerometer", "Gravity") else 1.0
        out = d.copy()
        out[AXES] = (sg * d[AXES].to_numpy()) @ R.T
        aligned[s] = out

    # 4) band-pass (the notebook also splits at gaps > 0.2 s first)
    t_al = aligned["Accelerometer"]["seconds_elapsed"].to_numpy()
    gaps = np.where(np.diff(t_al) > 0.2)[0] + 1
    bounds = [0, *gaps, len(t_al)]
    filt = {}
    for s, key in [("Accelerometer", "acc"), ("Gyroscope", "gyr")]:
        for ax in AXES:
            sig = aligned[s][ax].to_numpy()
            parts = [bandpass(sig[a:b], key) if b - a >= 50 else np.full(b - a, np.nan)
                     for a, b in zip(bounds[:-1], bounds[1:])]
            filt[f"{key}_{ax}"] = np.concatenate(parts)

    # 5) windows (start indices inside each continuous segment)
    win_starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for st in range(0, b - a - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            win_starts.append(a + st)

    return dict(raw=raw, t_raw=t_raw, n_raw=n_raw, cut_samples=c, stat=stat,
                energy=energy, moving=moving, aligned=aligned, filt=filt, t_al=t_al,
                win_starts=win_starts, theta=theta, psi=psi, g_med=g_med)


def plot_stages(S, name, out_dir=None):
    fig, axs = plt.subplots(5, 1, figsize=(11, 15))
    t = S["t_raw"]
    c = S["cut_samples"]

    # A: raw accelerometer, cut borders and stationary parts greyed
    ax = axs[0]
    for a in AXES:
        ax.plot(t, S["raw"]["Accelerometer"][a].to_numpy()[:S["n_raw"]], lw=0.6,
                color=C_AXIS[a], label=f"acc {a}")
    ax.axvspan(t[0], t[c], color=C_REMOVED, alpha=0.8, label="cut 1.5 s")
    ax.axvspan(t[-c], t[-1], color=C_REMOVED, alpha=0.8)
    stat_t = S["raw"]["Accelerometer"]["seconds_elapsed"].to_numpy()[c:S["n_raw"] - c][S["stat"].to_numpy()]
    if len(stat_t):
        ax.scatter(stat_t, np.full(len(stat_t), ax.get_ylim()[1] * 0.9), s=2,
                   color="#52514e", label="stationary (removed)")
    ax.set_title(f"1-2. Raw accelerometer, {S['n_raw']} samples: borders cut, "
                 f"{int(S['stat'].sum())} stationary samples removed")
    ax.set_ylabel("m/s²"); ax.legend(loc="upper right", ncol=5, fontsize=7)

    # B: gravity before/after alignment
    ax = axs[1]
    gb = S["moving"]["Gravity"]; ga = S["aligned"]["Gravity"]
    for a in AXES:
        ax.plot(gb["seconds_elapsed"], gb[a], lw=0.8, ls="--", color=C_AXIS[a], label=f"before {a}")
        ax.plot(ga["seconds_elapsed"], ga[a], lw=1.2, color=C_AXIS[a], label=f"after {a}")
    ax.set_title(f"3. Gravity alignment: pitch {S['theta']:.1f}°, roll in holder {S['psi']:.1f}° "
                 f"-> gravity on -y only")
    ax.set_ylabel("m/s²"); ax.legend(loc="center right", ncol=3, fontsize=7)

    # C: aligned vs band-passed acc_y, 5 s zoom
    ax = axs[2]
    t_al = S["t_al"]
    i0 = len(t_al) // 2; sl = slice(i0, i0 + 5 * FS)
    ax.plot(t_al[sl], S["aligned"]["Accelerometer"]["y"].to_numpy()[sl], lw=0.8,
            color="#a5a39a", label="aligned acc_y")
    ax.plot(t_al[sl], S["filt"]["acc_y"][sl], lw=1.0, color=C_AXIS["y"],
            label="band-passed 10-40 Hz")
    ax.set_title("4. Band-pass filter (5 s zoom): slow components (body motion, slope) removed")
    ax.set_ylabel("m/s²"); ax.legend(loc="upper right", fontsize=7)

    # D: windowing
    ax = axs[3]
    y = S["filt"]["acc_y"]
    ax.plot(t_al, y, lw=0.5, color=C_AXIS["y"])
    lo, hi = np.nanmin(y), np.nanmax(y)
    for k, st in enumerate(S["win_starts"]):
        x0 = t_al[st]; x1 = t_al[st + WINDOW_SAMPLES - 1]
        yb = lo + (k % 2) * 0.06 * (hi - lo)
        ax.add_patch(Rectangle((x0, yb - 0.04 * (hi - lo)), x1 - x0, 0.04 * (hi - lo),
                               color="#4a3aa7", alpha=0.6))
    ax.set_title(f"5. Windowing: {len(S['win_starts'])} windows of 3 s (300 samples), step 2.1 s (30% overlap)")
    ax.set_ylabel("acc_y m/s²")

    # E: one window -> PSD and band energies (the spectral features)
    ax = axs[4]
    k = int(np.argmax([np.nanstd(y[s:s + WINDOW_SAMPLES]) for s in S["win_starts"]]))
    w = y[S["win_starts"][k]:S["win_starts"][k] + WINDOW_SAMPLES]
    f, p = welch(w, fs=FS, nperseg=len(w) // 2)
    ax.plot(f, p, color=C_AXIS["y"], lw=1.5)
    for (b0, b1), col in zip([(10, 20), (20, 30), (30, 40)], ["#e87ba4", "#eda100", "#1baf7a"]):
        m = (f >= b0) & (f <= b1)
        ax.fill_between(f[m], p[m], color=col, alpha=0.35,
                        label=f"band_energy_{b0}_{b1}Hz = {p[m].sum():.3g}")
    rms = np.sqrt(np.mean(w ** 2))
    ax.set_title(f"6. Feature extraction on window #{k} (the roughest): Welch PSD of acc_y, "
                 f"rms = {rms:.2f}, std = {np.std(w):.2f}, ptp = {np.ptp(w):.2f}")
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD"); ax.legend(fontsize=7)

    for a in axs[:4]:
        a.set_xlabel("seconds_elapsed (s)")
    fig.suptitle(f"From raw data to features: {name}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"pipeline_stages_{name}.png"), dpi=150)
    return fig


def plot_funnel(stages, out_path=None):
    """
    Data-volume diagram of the whole pipeline. `stages` is a list of
    (step name, shape text, note). Build it from notebook variables, e.g.
      ("Windowing", f"{len(windows_data)} windows x 300 x 6", "3 s, 30% overlap")
    """
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.62 * len(stages)))
    ax.axis("off")
    n = len(stages)
    for i, (step, shape, note) in enumerate(stages):
        yc = n - 1 - i
        width = 0.9 - 0.45 * i / max(n - 1, 1)
        ax.add_patch(Rectangle((0.5 - width / 2, yc - 0.35), width, 0.7,
                               color="#2a78d6", alpha=0.15 + 0.6 * i / max(n - 1, 1)))
        ax.text(0.5, yc + 0.08, f"{step}:  {shape}", ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(0.5, yc - 0.2, note, ha="center", va="center", fontsize=8, color="#52514e")
        if i < n - 1:
            ax.annotate("", xy=(0.5, yc - 0.62), xytext=(0.5, yc - 0.38),
                        arrowprops=dict(arrowstyle="->", color="#52514e"))
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, n - 0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


if __name__ == "__main__":
    rec = sys.argv[1] if len(sys.argv) > 1 else RECORDING_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    S = run_stages(rec)
    name = os.path.basename(os.path.normpath(rec))
    plot_stages(S, name, out)
    print(f"{name}: raw {S['n_raw']} samples, after cut {S['n_raw'] - 2 * S['cut_samples']}, "
          f"stationary removed {int(S['stat'].sum())}, windows {len(S['win_starts'])}, "
          f"pitch {S['theta']:.1f} deg, roll {S['psi']:.1f} deg")
