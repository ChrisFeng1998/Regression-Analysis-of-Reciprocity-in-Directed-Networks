#!/usr/bin/env python3
"""Replot existing studentized errors with exact pointwise normal envelopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jasa_qq_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy import stats


SETTINGS = (
    {"a": 0.2, "b": 0.1, "directory": "0.2_0.1", "figure": "Figure_4"},
    {"a": 0.2, "b": -0.1, "directory": "0.2_-0.1", "figure": "Figure_S16"},
    {"a": 0.3, "b": 0.5, "directory": "0.3_0.5", "figure": "Figure_S20"},
)
SIZES = (50, 100, 150, 200)
PARAMETERS = (("rho", r"$\rho_n$"), ("gamma1", r"$\gamma_1$"), ("gamma2", r"$\gamma_2$"))
COVERAGE = 0.95
AXIS_LIMIT = 4.5
ENVELOPE_COLOR = "#d9dde2"
REFERENCE_COLOR = "#b13c30"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normal_envelope(sample_size: int) -> tuple[np.ndarray, ...]:
    """Normal-order-statistic intervals; x uses SciPy/Filliben positions.

    Under an iid N(0,1) reference, Phi(Z_(r)) is Beta(r, B+1-r).
    These are marginal 95% intervals at each rank, not a simultaneous band.
    The interval endpoints depend on B, never on network size n.
    """
    if sample_size < 2:
        raise ValueError("At least two independent simulation replicates are required.")
    ranks = np.arange(1, sample_size + 1)
    positions = np.empty(sample_size, dtype=float)
    positions[0] = 1 - 0.5 ** (1 / sample_size)
    positions[-1] = 0.5 ** (1 / sample_size)
    positions[1:-1] = (ranks[1:-1] - 0.3175) / (sample_size + 0.365)
    x = stats.norm.ppf(positions)
    tail = (1 - COVERAGE) / 2
    lower = stats.norm.ppf(stats.beta.ppf(tail, ranks, sample_size + 1 - ranks))
    upper = stats.norm.ppf(stats.beta.ppf(1 - tail, ranks, sample_size + 1 - ranks))
    if not np.all(np.isfinite(np.concatenate((x, lower, upper)))):
        raise ValueError("Reference envelope contains nonfinite values.")
    if not (np.all(lower < upper) and np.all(np.diff(lower) > 0) and np.all(np.diff(upper) > 0)):
        raise ValueError("Reference envelope is not ordered correctly.")
    if not np.allclose(lower, -upper[::-1], atol=1e-10, rtol=0):
        raise ValueError("Reference envelope failed the normal symmetry check.")
    return ranks, x, lower, upper

def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

def legend_handles() -> list:
    return [
        Patch(facecolor=ENVELOPE_COLOR, edgecolor="none", label="95% pointwise envelope"),
        Line2D([], [], color="black", marker=".", linestyle="none", markersize=4, label="Studentized errors"),
        Line2D([], [], color=REFERENCE_COLOR, linewidth=1, label=r"$y=x$"),
    ]

def draw_panel(ax, panel: dict, compact: bool = False) -> None:
    x, y = panel["x"], panel["ordered"]
    ax.set_axisbelow(True)
    ax.grid(color="#eceef0", linewidth=0.45)
    ax.fill_between(x, panel["lower"], panel["upper"], color=ENVELOPE_COLOR, linewidth=0, zorder=2)
    ax.plot([-AXIS_LIMIT, AXIS_LIMIT], [-AXIS_LIMIT, AXIS_LIMIT], color=REFERENCE_COLOR, linewidth=0.9, zorder=3)
    ax.scatter(x, y, s=2.2 if compact else 4, color="black", alpha=0.82, linewidths=0, zorder=4)
    ax.set(xlim=(-AXIS_LIMIT, AXIS_LIMIT), ylim=(-AXIS_LIMIT, AXIS_LIMIT))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2.5)
    for spine in ax.spines.values():
        spine.set_color("#505050")
    if compact:
        ax.text(0.055, 0.94, panel["parameter_label"], transform=ax.transAxes, ha="left", va="top", fontsize=10.5)

def save_pair(fig, destination: Path, title: str, dpi: int) -> list[str]:
    # Setting labels contain decimal points, so append rather than replace a suffix.
    pdf_path = Path(str(destination) + ".pdf")
    png_path = Path(str(destination) + ".png")
    fig.savefig(pdf_path, metadata={
        "Title": title,
        "Subject": "Studentized errors with 95% pointwise iid standard-normal order-statistic reference envelopes",
        "Creator": "generate_qq_plots.py (Matplotlib)",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return [str(pdf_path), str(png_path)]

def load_panels(input_dir: Path) -> tuple[dict, list, dict]:
    panels, manifests = {}, []
    input_checks = {"checked_files": 0, "max_studentization_difference": 0.0, "all_finite": True, "all_B_1000": True}
    for setting in SETTINGS:
        for n in SIZES:
            path = input_dir / setting["directory"] / f"{n}.csv.gz"
            if not path.exists():path=path.with_suffix("")  # Also accept ordinary .csv inputs.
            before = sha256(path)
            frame = pd.read_csv(path)
            values = frame.to_numpy(dtype=float)
            if values.shape[1] != 15:
                raise ValueError(f"Expected exactly 15 columns: {path}; got {values.shape}.")
            if not np.isfinite(values).all():
                raise ValueError(f"Missing/nonfinite input values: {path}.")
            truth = np.array([setting["b"] * np.log(n) + 0.5, 1.0, 1.0])
            widths = values[:, 6:9] - values[:, 3:6]
            if not np.all(widths > 0):
                raise ValueError(f"Nonpositive CI width: {path}.")
            reconstructed = (values[:, :3] - truth) * (2 * 1.96) / widths
            stored_z = values[:, 12:15]
            difference = float(np.max(np.abs(reconstructed - stored_z)))
            if not np.allclose(reconstructed, stored_z, atol=1e-10, rtol=1e-10):
                raise ValueError(f"Saved studentized errors do not match estimates and CIs: {path}.")
            B = len(frame)
            ranks, x, lower, upper = normal_envelope(B)
            scipy_x, _ = stats.probplot(stored_z[:, 0], dist="norm", fit=False)
            if not np.allclose(x, scipy_x, atol=1e-12, rtol=0):
                raise ValueError("Plotting positions do not agree with scipy.stats.probplot.")
            for j, (parameter, label) in enumerate(PARAMETERS):
                z = stored_z[:, j].copy()
                ordered = np.sort(z)
                if np.max(np.abs(np.concatenate((ordered, lower, upper)))) >= AXIS_LIMIT:
                    raise ValueError(f"Plot limits would clip data or reference envelope: {path}, {parameter}.")
                panels[(setting["directory"], n, parameter)] = {
                    **setting, "n": n, "B": B, "parameter": parameter,
                    "parameter_label": label, "source": str(path),
                    "ranks": ranks, "x": x, "z": z, "ordered": ordered,
                    "lower": lower, "upper": upper,
                }
            manifests.append({
                "source_relative_to_output": os.path.relpath(path, input_dir),
                "source_absolute_path": str(path), "sha256": before,
                "rows": B, "columns": values.shape[1],
                "studentized_columns_zero_based": [12, 13, 14],
                "parameter_order": [p[0] for p in PARAMETERS],
                "max_studentization_difference": difference,
            })
            input_checks["checked_files"] += 1
            input_checks["max_studentization_difference"] = max(input_checks["max_studentization_difference"], difference)
            input_checks["all_B_1000"] &= B == 1000
    return panels, manifests, input_checks


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=root/"outputs/visualization/qq")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    single_dir,combined_dir=args.output_dir/"individual",args.output_dir/"combined"
    single_dir.mkdir(parents=True,exist_ok=True);combined_dir.mkdir(parents=True,exist_ok=True)
    panels,_,checks=load_panels(args.input_dir)
    configure_style()
    diagnostics=[]
    for panel in panels.values():
        fig,ax=plt.subplots(figsize=(4.2,4.4))
        fig.subplots_adjust(left=.17,right=.96,bottom=.20,top=.84)
        draw_panel(ax,panel)
        ax.set_xlabel("Standard normal quantiles",labelpad=5)
        ax.set_ylabel("Ordered studentized errors",labelpad=5)
        fig.suptitle(f"{panel['parameter_label']},  $n={panel['n']}$",y=.975,fontsize=12)
        fig.legend(handles=legend_handles(),loc="lower center",bbox_to_anchor=(.5,.025),ncol=3,
                   frameon=False,fontsize=7.1,handlelength=1.4,columnspacing=1.1)
        stem=single_dir/f"{panel['directory']}_{panel['n']}_{panel['parameter']}"
        save_pair(fig,stem,f"{panel['figure']}: {panel['parameter']}, n={panel['n']}",args.dpi)
        outside=(panel['ordered']<panel['lower'])|(panel['ordered']>panel['upper'])
        diagnostics.append(dict(setting=panel['directory'],n=panel['n'],parameter=panel['parameter'],B=panel['B'],
            mean_studentized_error=panel['z'].mean(),sd_studentized_error_ddof1=panel['z'].std(ddof=1),
            outside_pointwise_envelope=int(outside.sum())))
    for setting in SETTINGS:
        fig,axes=plt.subplots(3,4,figsize=(8.,7.),sharex=True,sharey=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.88,wspace=.10,hspace=.12)
        for row,(parameter,_) in enumerate(PARAMETERS):
            for col,n in enumerate(SIZES):
                ax=axes[row,col]
                draw_panel(ax,panels[(setting['directory'],n,parameter)],compact=True)
                if row==0:ax.set_title(f"$n={n}$",pad=6,fontsize=10)
                ax.tick_params(labelbottom=row==2,labelleft=col==0)
        fig.supxlabel("Standard normal quantiles",x=.535,y=.075,fontsize=10)
        fig.supylabel("Ordered studentized errors",x=.018,y=.51,fontsize=10)
        fig.legend(handles=legend_handles(),loc="lower center",bbox_to_anchor=(.53,.012),ncol=3,
                   frameon=False,fontsize=8,handlelength=1.6,columnspacing=2.)
        stem=combined_dir/f"{setting['figure']}_QQ_pointwise_95_no_header"
        save_pair(fig,stem,f"{setting['figure']}: a={setting['a']}, b={setting['b']}",args.dpi)
    pd.DataFrame(diagnostics).to_csv(args.output_dir/"qq_diagnostics.csv",index=False)
    print(f"Generated 39 QQ PDFs and PNGs in {args.output_dir}; 12 saved-data studentization checks passed.")


if __name__=="__main__":main()
