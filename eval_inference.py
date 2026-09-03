from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

OUT_DIR = Path(r"./PATH")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_IDX = 4999
TAU_COL_FOR_ANALYSIS = "volt_ai2"
MIN_COUNT_FOR_TAU_PLOT = 400
SMOOTH_WINDOW = 5

# =========================================================
# Calibration coefficients
# =========================================================
c = {
    "volt_ai0": [0, 2.95036612411691, -6.76217599157229, 319.355923774047, -67.9063003860238, -827.321662708292, 791.566879134215, 26156.7494359199],
    "volt_ai1": [0, 3.27923723907693, 10.8722635868812, 178.132631764582, 75.5289690946718, -296.194087538138, 362.298533821201, 3890.41471987789],
    "volt_ai2": [0, 2.98878837902547, 3.43324598296056, 191.084696513307, -6.99509069090871, -828.969206611036, 109.972250385424, 3232.10987880775],
    "volt_ai3": [0, 4.3945944709364, 10.7388025829227, 510.879295153304, -87.2188553287849, -2844.15429338984, 3959.65002683634, 41867.3303587697],
    "volt_ai4": [0, 3.39814310768549, -2.70167304608119, 389.139377339273, -69.9701537228927, -2388.43985256715, 1314.57025970785, 25177.4639030281],
    "volt_ai5": [0, 4.33809944555263, 12.67928815432, 605.111792390368, -109.398831332285, -6454.7392710159, 374.98375164121, 83197.1907873874],
}

def parse_reference_voltages(filename: str | Path) -> pd.DataFrame:
    filename = Path(filename)
    txt = filename.read_text(encoding="utf-8")

    blocks = []
    depth = 0
    start = None

    for i, ch in enumerate(txt):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(txt[start:i + 1])
                start = None
    rows = []
    for k, block in enumerate(blocks):
        vals = re.findall(r"np\.float64\(([-+0-9.eE]+)\)", block)
        rows.append([float(v) for v in vals])

    lengths = [len(r) for r in rows]

    V = np.array(rows, dtype=float).T
    cols = [f"volt_ai{i}" for i in range(6)]
    return pd.DataFrame(V, columns=cols)


def read_inference_voltages(filename: str | Path) -> pd.DataFrame:
    sensor_cols = [f"volt_ai{i}" for i in range(6)]
    return pd.read_csv(filename, sep="\t", usecols=sensor_cols)

def read_inference_actions(filename: str | Path) -> pd.DataFrame:
    return pd.read_csv(filename, sep="\t", usecols=["action"])

def get_actions(filename: str | Path) -> pd.DataFrame:
    df = read_inference_actions(filename).copy()
    df["action"] = pd.to_numeric(df["action"], errors="raise").astype(int)

    invalid = set(df["action"].unique()) - {0, 1}

    return df.reset_index(drop=True)

def get_tau(voltages: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    tau = voltages.copy()
    for col in tau.columns:
        a0, a1, a2, a3, a4, a5, a6, a7 = coeffs[col]
        v = voltages[col]
        tau[col] = (
            a0
            + a1 * v
            + a2 * v**2
            + a3 * v**3
            + a4 * v**4
            + a5 * v**5
            + a6 * v**6
            + a7 * v**7
        )
    return tau

def get_tau_from_file(filename: str | Path, coeffs: dict) -> pd.DataFrame:
    voltages = read_inference_voltages(filename).copy().reset_index(drop=True)
    tau = get_tau(voltages, coeffs)
    return tau

def get_actions_and_tau(filename: str | Path, coeffs: dict, start_idx: int = 0) -> pd.DataFrame:
    actions = get_actions(filename)
    tau = get_tau_from_file(filename, coeffs)

    if start_idx > 0:
        actions = actions.iloc[start_idx:, :].reset_index(drop=True)
        tau = tau.iloc[start_idx:, :].reset_index(drop=True)

    df = pd.concat([actions, tau], axis=1)
    df.insert(0, "sample_idx", np.arange(len(df)))
    return df

def add_switch_off_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    actions = pd.to_numeric(out["action"], errors="raise").astype(int)

    run_lengths = np.zeros(len(actions), dtype=int)
    count = 0

    for i, a in enumerate(actions.to_numpy()):
        if a == 0:
            count += 1
        else:
            count = 0
        run_lengths[i] = count

    out["action"] = actions
    out["zero_run_length"] = run_lengths
    out["zero_run_bin"] = run_lengths
    return out


def get_max_consecutive_zeros(actions: pd.Series) -> int:
    actions = pd.to_numeric(actions, errors="raise").astype(int).to_numpy()
    max_run = 0
    count = 0
    for a in actions:
        if a == 0:
            count += 1
            max_run = max(max_run, count)
        else:
            count = 0
    return max_run


def summarize_tau_by_toff(df: pd.DataFrame, tau_col: str = TAU_COL_FOR_ANALYSIS) -> pd.DataFrame:
    summary = (
        df.groupby("zero_run_length")[tau_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
        )
        .reset_index()
        .rename(columns={"zero_run_length": "t_off"})
        .sort_values("t_off")
        .reset_index(drop=True)
    )
    return summary


def add_weighted_smoothing(
    summary: pd.DataFrame,
    value_col: str = "median",
    weight_col: str = "count",
    window: int = 5,
) -> pd.DataFrame:
    summary = summary.copy()

    vals = summary[value_col].to_numpy(dtype=float)
    weights = summary[weight_col].to_numpy(dtype=float)

    smooth = np.full(len(summary), np.nan)
    half = window // 2

    for i in range(len(summary)):
        lo = max(0, i - half)
        hi = min(len(summary), i + half + 1)

        v = vals[lo:hi]
        w = weights[lo:hi]

        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if mask.any():
            smooth[i] = np.average(v[mask], weights=w[mask])

    summary["smooth"] = smooth
    return summary


def get_fluctuation(dataframe: pd.DataFrame) -> list[float]:
    offsets = dataframe.mean().tolist()
    df_centered = dataframe - offsets
    rms = np.sqrt((df_centered ** 2).mean()).tolist()
    return rms


def plot_tau(tau_reference: pd.DataFrame, tau_inference: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.33, top=0.98)

    tau_ref = tau_reference[TAU_COL_FOR_ANALYSIS]
    tau_inf = tau_inference[TAU_COL_FOR_ANALYSIS]

    tau_ref_s = tau_ref.rolling(2000, min_periods=1, center=True, step=1).mean()
    tau_inf_s = tau_inf.rolling(2000, min_periods=1, center=True, step=1).mean()

    x_ref = np.arange(1, len(tau_ref_s) + 1)
    x_inf = np.arange(1, len(tau_inf_s) + 1)

    ax.plot(x_inf, tau_inf_s, color="black", linewidth=0.6, linestyle="-", label="inference", zorder=2)
    ax.plot(x_ref, tau_ref_s, color="grey", linewidth=0.6, linestyle="--", label="reference", zorder=3)

    ax.set_xlabel(r"sample")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_xticks([0, 50000, 100000])
    ax.set_ylabel(r"$\tau_{RS3}$ [Pa]")
    ax.grid(True, linestyle="--", linewidth=0.5,  zorder=0)

    fig.savefig(OUT_DIR / f"tau_sensor_{TAU_COL_FOR_ANALYSIS}_comparison.eps")
    fig.savefig(OUT_DIR / f"tau_sensor_{TAU_COL_FOR_ANALYSIS}_comparison.png", dpi=150)
    plt.close(fig)


def plot_dc(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 1.6))
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.33, top=0.90)

    x_df = np.arange(1, len(df) + 1)

    ax.plot(x_df, df, color="black", linewidth=0.6, linestyle="-")

    ax.set_xlabel(r"sample")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_xlim(0, 1000)
    ax.set_ylabel("DC [ ]")
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.savefig(OUT_DIR / "dc_comparison.eps", dpi=150)
    #fig.savefig(OUT_DIR / "dc_comparison.pdf")
    plt.close(fig)


def plot_toff_counts(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.33, top=0.98)
    """
    ax.bar(
        summary["t_off"],
        summary["count"],
        width=0.8,
        color="black",
        alpha=0.75,
    )
    """
    ax.set_xlabel(r"$t_{off}$ [steps]")
    ax.set_ylabel("count")
    ax.set_xlim(-0.5, 20.5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.savefig(OUT_DIR / f"toff_counts_{TAU_COL_FOR_ANALYSIS}.eps", dpi=150)
    #fig.savefig(OUT_DIR / "toff_counts.pdf")
    plt.close(fig)


def plot_tau_binned_by_toff(
    summary: pd.DataFrame,
    min_count: int=100,
    tau_label: str = r"$\tau_{RS0}$ [Pa]",
) -> None:

    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.33, top=0.98)
    """
    ax.bar(
        summary_supported["t_off"],
        summary_supported["median"],
        width=0.8,
        color="black",
        alpha=0.75,
    )
    """
    ax.set_xlabel(r"$t_{off}$ [steps]")
    ax.set_ylabel(tau_label)
    ax.set_xlim(-0.5, 20.5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.savefig(OUT_DIR / f"tau_binned_by_toff_{TAU_COL_FOR_ANALYSIS}.eps", dpi=150)
    plt.close(fig)


def plot_tau_approx_vs_toff(
    summary: pd.DataFrame,
    min_count: int = 50,
    tau_label: str = r"$\tau_{RS0}$ [Pa]",
) -> None:
    summary_supported = summary[summary["count"] >= min_count].copy()

    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.33, top=0.98)



    ax.plot(
        summary_supported["t_off"],
        summary_supported["smooth"],
        color="black",
        linewidth=0.8,
        linestyle="-",
        label="smoothed trend",
        zorder=10,
    )
    custom_greys = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_greys",
        ["0.75", "0.0"]  # low count -> light grey, high count -> black
    )
    sc = ax.scatter(
        summary_supported["t_off"],
        summary_supported["median"],
        c=summary_supported["count"],
        cmap=custom_greys,
        s=16,
        edgecolors="none",
        alpha=0.8,
        label="median bins",
        zorder=2,
    )
    """
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("count")
    """

    ax.set_xlabel(r"$t_{off}$ [steps]")
    ax.set_ylabel(tau_label)
    ax.set_xlim(-0.5, 20.5)
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.savefig(OUT_DIR / f"tau_approx_vs_toff_{TAU_COL_FOR_ANALYSIS}.eps", dpi=150)
    plt.close(fig)

def main() -> None:
    ref_file = Path(
        r"PATH.txt"
    )
    inf_file = Path(
        r"PATH.txt"
    )

    voltages_reference = parse_reference_voltages(ref_file).iloc[START_IDX:, :].reset_index(drop=True)
    tau_reference = get_tau(voltages_reference, c)

    tau_inference = get_tau_from_file(inf_file, c).iloc[START_IDX:, :].reset_index(drop=True)

    actions_and_tau = get_actions_and_tau(inf_file, c, start_idx=START_IDX)
    actions_and_tau = add_switch_off_bins(actions_and_tau)

    dc = actions_and_tau[["action"]].rolling(8, min_periods=8, center=True, step=1).mean()
    dc = dc.iloc[7:, :].reset_index(drop=True)

    summary_tau_toff = summarize_tau_by_toff(actions_and_tau, tau_col=TAU_COL_FOR_ANALYSIS)
    summary_tau_toff = add_weighted_smoothing(
        summary_tau_toff,
        value_col="median",
        weight_col="count",
        window=SMOOTH_WINDOW,
    )

    print("\nRaw maximum consecutive zeros:")
    print(get_max_consecutive_zeros(actions_and_tau["action"]))

    print("\nRare bins (count < MIN_COUNT_FOR_TAU_PLOT):")
    print(summary_tau_toff[summary_tau_toff["count"] < MIN_COUNT_FOR_TAU_PLOT])

    print("\nBins used for tau plots:")
    print(summary_tau_toff[summary_tau_toff["count"] >= MIN_COUNT_FOR_TAU_PLOT])

    mean_ref = tau_reference.mean().tolist()
    fluctuation_ref = get_fluctuation(tau_reference)
    print("Mean_tau, reference:", mean_ref)
    print("Fluctuation width, reference:", fluctuation_ref)

    mean_inf = tau_inference.mean().tolist()
    fluctuation_inf = get_fluctuation(tau_inference)
    print("Mean_tau, inference:", mean_inf)
    print("Fluctuation width, inference:", fluctuation_inf)

    plot_dc(dc)
    plot_tau(tau_reference, tau_inference)

    plot_toff_counts(summary_tau_toff)
    plot_tau_binned_by_toff(
        summary_tau_toff,
        min_count=MIN_COUNT_FOR_TAU_PLOT,
        tau_label=r"$\tau_{RS3}$ [Pa]",
    )
    plot_tau_approx_vs_toff(
        summary_tau_toff,
        min_count=MIN_COUNT_FOR_TAU_PLOT,
        tau_label=r"$\tau_{RS3}$ [Pa]",
    )


if __name__ == "__main__":
    main()