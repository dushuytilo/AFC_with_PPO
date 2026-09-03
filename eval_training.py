from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy.printing.pretty.pretty_symbology import line_width

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

YLABELS = {
    "actor_loss": r"actor loss",
    "critic_loss": r"critic loss",
    "MEAN_GAMMA_AT_REFERENCE": r"mean $\gamma^{+}$",
    "MEAN DUTY CYCLE": r"mean DC",
}

PLOT_FOLDER = Path.cwd() / "plots" / "horizon_variations"
LOG_FOLDER = Path.cwd() / "logs" / "rl_afc" / "runs"

METRICS = {
    "TRAIN_EPOCH": ["MEAN_GAMMA_AT_REFERENCE", "MEAN DUTY CYCLE"],
    "frame_idx": ["actor_loss", "critic_loss"],
    "state_t": ["dist_prob"]
}

LOG_FILES = ["ppo_epoch_data.txt", "update_data.txt", "batch_data.txt"]

SMOOTH_WINDOW = 1000


def state_to_pa(state):
    tau = pd.Series(state, copy=False)

    return (
        4.3946 * tau
        + 10.7388 * tau**2
        + 510.8793 * tau**3
        - 87.2189 * tau**4
        - 2.8442e3 * tau**5
        + 3.9597e3 * tau**6
        + 4.1867e4 * tau**7
    )


def mean_centered_intervals(state_series: pd.Series) -> dict[str, tuple[float, float] | float]:
    s = state_series.dropna()
    mu = s.mean()
    abs_dev = (s - mu).abs()

    d50 = abs_dev.quantile(0.50)
    d90 = abs_dev.quantile(0.90)

    return {
        "mean": mu,
        "50": (mu - d50, mu + d50),
        "90": (mu - d90, mu + d90),
    }

def read_run_csv(run_folder: str, metrics: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    run_path = Path(run_folder)

    df0 = pd.read_csv(run_path / LOG_FILES[0], sep="\t")
    df0.columns = [str(c).strip() for c in df0.columns]
    df_epoch = df0[["TRAIN_EPOCH"] + metrics["TRAIN_EPOCH"]].copy()

    df_frame = None
    for lf in LOG_FILES:
        df = pd.read_csv(run_path / lf, sep="\t")
        df.columns = [str(c).strip() for c in df.columns]

        present = [m for m in metrics["frame_idx"] if m in df.columns]
        if "frame_idx" not in df.columns or not present:
            continue

        part = df[["frame_idx"] + present].copy()
        df_frame = part if df_frame is None else pd.merge(df_frame, part, on="frame_idx", how="outer")

    if df_frame is not None and not df_frame.empty:
        df_frame = df_frame.sort_values("frame_idx").reset_index(drop=True)

    dfb = pd.read_csv(run_path / LOG_FILES[2], sep="\t")
    dfb.columns = [str(c).strip() for c in dfb.columns]

    dfb_last = dfb[dfb["train_epoch"] == dfb["train_epoch"].max()].copy()
    df_policy = dfb_last[["state_t"] + metrics["state_t"]].copy()
    df_policy["state_pa"] = state_to_pa(df_policy["state_t"])

    df_epoch = df_epoch[df_epoch["TRAIN_EPOCH"] <= 500].copy()

    if df_frame is not None and not df_frame.empty:
        df_frame = df_frame[df_frame["frame_idx"] <= 50000].copy()

    dfb = dfb[dfb["train_epoch"] <= 500].copy()
    last_epoch = dfb["train_epoch"].max()
    dfb_last = dfb[dfb["train_epoch"] == last_epoch].copy()

    epochs_to_plot = [100, 300, 500]
    df_policy = dfb[dfb["train_epoch"].isin(epochs_to_plot)].copy()
    df_policy = df_policy[["train_epoch", "state_t", "dist_prob"]].copy()
    df_policy["state_pa"] = state_to_pa(df_policy["state_t"])

    return df_epoch, df_frame, df_policy


def get_outdir(counter_key: str) -> Path:
    outdir = PLOT_FOLDER / counter_key
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def list_run_folders(runs_root: str | Path) -> list[Path]:
    runs_root = Path(runs_root)
    return [p for p in runs_root.iterdir() if p.is_dir()]


def plot_losses(run_folder: Path, df_frame: pd.DataFrame):
    metrics = ["actor_loss", "critic_loss"]
    loss_smooth_window = 1000

    fig, axes = plt.subplots(2, 1, figsize=(3, 3.2), sharex=False)
    axes[0].set_position([0.18, 0.65, 0.75, 0.30])
    axes[1].set_position([0.18, 0.14, 0.75, 0.30])

    for ax, metric in zip(axes, metrics):
        tmp = df_frame[["frame_idx", metric]].dropna().copy()
        tmp = tmp.sort_values("frame_idx")

        tmp[f"{metric}_smooth"] = (
            tmp[metric]
            .rolling(loss_smooth_window, min_periods=1, center=True)
            .mean()
        )

        ax.plot(
            tmp["frame_idx"],
            tmp[metric],
            color="0.75",
            linewidth=0.4,
            label="raw",
            zorder=1,
        )

        ax.plot(
            tmp["frame_idx"],
            tmp[f"{metric}_smooth"],
            color="black",
            linewidth=0.9,
            label="rolling mean",
            zorder=2,
        )

        ax.set_xlim(0, 50000)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

        ax.set_xlabel(r"Update")
        ax.set_ylabel(YLABELS.get(metric, metric))
        ax.grid(False)

    return fig


def plot_control_metrics(run_folder: Path, df_epoch: pd.DataFrame):
    metrics = ["MEAN_GAMMA_AT_REFERENCE", "MEAN DUTY CYCLE"]

    fig, axes = plt.subplots(2, 1, figsize=(3, 3.2), sharex=False)
    axes[0].set_position([0.18, 0.65, 0.75, 0.30])
    axes[1].set_position([0.18, 0.14, 0.75, 0.30])

    for ax, metric in zip(axes, metrics):
        tmp = df_epoch[["TRAIN_EPOCH", metric]].dropna().copy()
        tmp = tmp.sort_values("TRAIN_EPOCH")

        ax.plot(tmp["TRAIN_EPOCH"], tmp[metric], color="black", linewidth=1.0)
        ax.set_xlim(0, 500)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"Episode")
        ax.set_ylabel(YLABELS.get(metric, metric))
        ax.grid(True, linestyle = "--", linewidth=0.5)

    return fig


def plot_prob_open_over_states(run_folder: Path, df_policy: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.0, 1.6))
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.33, top=0.90)

    df_100 = df_policy[df_policy["train_epoch"] == 100].copy()
    df_300 = df_policy[df_policy["train_epoch"] == 300].copy()
    df_500 = df_policy[df_policy["train_epoch"] == 500].copy()

    if df_500.empty:
        raise ValueError("No data found for train_epoch == 500.")

    bands = mean_centered_intervals(df_500["state_pa"])
    x90_lo, x90_hi = bands["90"]
    x50_lo, x50_hi = bands["50"]
    mu = bands["mean"]

    ax.axvspan(
        x90_lo,
        x90_hi,
        facecolor="none",
        edgecolor="0.55",
        hatch="//////",
        linewidth=0.1,
        zorder=0,
    )

    ax.axvspan(
        x50_lo,
        x50_hi,
        facecolor="none",
        edgecolor="0.55",
        hatch="//////////////",
        linewidth=0.1,
        zorder=1,
    )

    ax.axvline(
        mu,
        linestyle="--",
        linewidth=1,
        color="black",
        zorder=2,
    )

    handles = []

    if not df_100.empty:
        h100 = ax.scatter(
            df_100["state_pa"],
            df_100["dist_prob"],
            s=1,
            color="tab:blue",
            label="Ep. 100",
            zorder=3,
        )
        handles.append(h100)

    if not df_300.empty:
        h300 = ax.scatter(
            df_300["state_pa"],
            df_300["dist_prob"],
            s=1,
            color="gold",
            label="Ep. 300",
            zorder=4,
        )
        handles.append(h300)

    h500 = ax.scatter(
        df_500["state_pa"],
        df_500["dist_prob"],
        s=1,
        color="black",
        label="Ep. 500",
        zorder=5,
    )
    handles.append(h500)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(r"state [Pa]")
    ax.set_ylabel(r"$p_{open}$")
    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        bbox_transform=ax.transAxes,
        markerscale=4,
        handletextpad=0.10,
        borderaxespad=0.0,
    )

    return fig


def main():
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    for run_folder in list_run_folders(LOG_FOLDER):
        df_epoch, df_frame, df_policy = read_run_csv(run_folder, METRICS)

        fig_losses = plot_losses(run_folder, df_frame)
        out_losses = PLOT_FOLDER / f"{run_folder.name}_losses.eps"
        fig_losses.savefig(out_losses, dpi=150)
        plt.close(fig_losses)
        
        fig_control = plot_control_metrics(run_folder, df_epoch)
        out_control = PLOT_FOLDER / f"{run_folder.name}_gamma_dc.eps"
        fig_control.savefig(out_control, dpi=150 )
        plt.close(fig_control)

        fig_policy = plot_prob_open_over_states(run_folder, df_policy)
        out_policy = PLOT_FOLDER / f"{run_folder.name}_prob_open_vs_state.eps"
        fig_policy.savefig(out_policy, dpi=150)
        plt.close(fig_policy)

        print(f"Saved: {out_losses}")
        print(f"Saved: {out_control}")
        print(f"Saved: {out_policy}")


if __name__ == "__main__":
    main()