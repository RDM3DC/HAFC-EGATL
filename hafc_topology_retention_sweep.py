"""Sweep HAFC topology retention across size, disorder, and QWZ coupling.

This script answers three practical questions:

1. Does Bott quantization survive longer on larger lattices?
2. How sensitive is long-time topology retention to initial bond disorder?
3. Does the pi_f top-edge signature show an early precursor before the Bott plateau is lost?

Outputs:
    - CSV table: hafc_topology_retention_sweep.csv
    - Optional plot: hafc_topology_retention_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver.egatl import (
    EGATLParams,
    EntropyParams,
    RulerParams,
    run_recovery_protocol,
    qwz_topology_series,
    top_edge_signature_series,
)


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _first_true_time(t: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values.astype(float, copy=True)

    out = np.empty(values.size, dtype=float)
    prefix = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        out[idx] = (prefix[idx + 1] - prefix[start]) / count
    return out


def _safe_scale(values: np.ndarray, reference: float) -> float:
    if not values.size:
        return float("nan")
    return float(max(np.std(values), 0.05 * abs(reference), 1e-6))


def _rolling_drop_precursor(
    t: np.ndarray,
    signal: np.ndarray,
    baseline_mask: np.ndarray,
    window: int,
    z_threshold: float,
) -> Dict[str, float | np.ndarray]:
    smooth = _rolling_mean(signal, max(1, window))
    baseline = smooth[baseline_mask]
    baseline_mean = _safe_mean(baseline)
    baseline_scale = _safe_scale(baseline, baseline_mean)
    if math.isnan(baseline_mean) or math.isnan(baseline_scale):
        drop_z = np.full_like(smooth, np.nan, dtype=float)
        precursor_time = float("nan")
        max_drop_z = float("nan")
    else:
        drop_z = (baseline_mean - smooth) / baseline_scale
        candidate_mask = (~baseline_mask) & (drop_z >= z_threshold)
        precursor_time = _first_true_time(t, candidate_mask)
        max_drop_z = float(np.nanmax(drop_z))

    return {
        "smooth": smooth,
        "baseline_mean": baseline_mean,
        "baseline_scale": baseline_scale,
        "drop_z": drop_z,
        "precursor_time": precursor_time,
        "max_drop_z": max_drop_z,
    }


def _run_case(
    nx: int,
    ny: int,
    T: float,
    dt: float,
    damage_time: float,
    mass: float,
    seed: int,
    init_phase_noise: float,
    qzw_pi_gain: float,
    qzw_entropy_gain: float,
    sample_stride: int,
    precursor_window: int,
    precursor_z_threshold: float,
) -> Dict[str, float]:
    eg = EGATLParams(
        alpha0=2.5,
        S_c=0.3,
        dS=0.5,
        mu0=0.25,
        S0=2.0,
        lambda_s=0.05,
        g_min=0.01,
        g_max=5.0,
        g_imag_max=3.0,
        budget_re=120.0,
        init_phase_noise=init_phase_noise,
        qzw_pi_gain=qzw_pi_gain,
        qzw_entropy_gain=qzw_entropy_gain,
    )
    ent = EntropyParams(
        S_init=1.5,
        S_eq=1.0,
        gamma=0.20,
        kappa_slip=0.10,
        Tij=1.0,
    )
    ruler = RulerParams(
        pi0=math.pi,
        pi_init=math.pi,
        alpha_pi=0.15,
        mu_pi=0.15,
        pi_min=0.5,
        pi_max=12.0,
    )

    lattice, out = run_recovery_protocol(
        nx=nx,
        ny=ny,
        T=T,
        dt=dt,
        seed=seed,
        damage_time=damage_time,
        mass=mass,
        damage_factor=1e-4,
        eg=eg,
        ent=ent,
        ruler=ruler,
    )

    t = out["t"][::sample_stride]
    g = out["g"][::sample_stride]
    pi_a = out["pi_a"][::sample_stride]
    S = out["S"][::sample_stride]

    topo = qwz_topology_series(
        g,
        lattice,
        pi_a_hist=pi_a,
        S_hist=S,
        mass0=mass,
        pi_gain=qzw_pi_gain,
        entropy_gain=qzw_entropy_gain,
    )
    sig_top = top_edge_signature_series(g, lattice, pi_a)

    pre_mask = t < damage_time
    post_mask = t >= damage_time
    baseline_horizon = min(float(t[-1]), damage_time)
    early_pre_mask = t <= max(dt * sample_stride, baseline_horizon * 0.25)
    if not np.any(early_pre_mask):
        early_pre_mask = pre_mask
    if not np.any(pre_mask):
        pre_mask = np.ones_like(t, dtype=bool)

    bott_ref = int(round(_safe_mean(topo["bott"][early_pre_mask])))
    bott_loss_mask = np.abs(topo["bott"] - bott_ref) > 0.25
    bott_loss_time = _first_true_time(t, bott_loss_mask)
    bott_loss_observed = 0.0 if math.isnan(bott_loss_time) else 1.0
    retention_time = float(bott_loss_time) if bott_loss_observed else float(t[-1])

    sig_top_pre = _safe_mean(sig_top[early_pre_mask])
    sig_top_threshold = 0.5 * sig_top_pre if not math.isnan(sig_top_pre) else float("nan")
    if math.isnan(sig_top_threshold):
        sig_top_half_time = float("nan")
    else:
        sig_top_half_time = _first_true_time(t, sig_top <= sig_top_threshold)

    precursor = _rolling_drop_precursor(
        t,
        sig_top,
        early_pre_mask,
        window=max(1, precursor_window),
        z_threshold=precursor_z_threshold,
    )
    sig_top_precursor_time = float(precursor["precursor_time"])
    precursor_margin = (
        float(retention_time - sig_top_precursor_time)
        if not math.isnan(sig_top_precursor_time)
        else float("nan")
    )
    precursor_lead_time = (
        precursor_margin
        if bott_loss_observed and not math.isnan(sig_top_precursor_time)
        else float("nan")
    )
    half_margin = (
        float(retention_time - sig_top_half_time)
        if not math.isnan(sig_top_half_time)
        else float("nan")
    )
    lead_time = (
        half_margin
        if bott_loss_observed and not math.isnan(sig_top_half_time)
        else float("nan")
    )

    row = {
        "nx": nx,
        "ny": ny,
        "T": T,
        "dt": dt,
        "damage_time": damage_time,
        "mass": mass,
        "seed": seed,
        "init_phase_noise": init_phase_noise,
        "qzw_pi_gain": qzw_pi_gain,
        "qzw_entropy_gain": qzw_entropy_gain,
        "bott_ref": float(bott_ref),
        "bott_loss_observed": bott_loss_observed,
        "retained_to_end": 1.0 - bott_loss_observed,
        "retention_time": retention_time,
        "bott_pre_mean": _safe_mean(topo["bott"][pre_mask]),
        "bott_post_mean": _safe_mean(topo["bott"][post_mask]),
        "qzw_chern_pre_mean": _safe_mean(topo["chern"][pre_mask]),
        "qzw_chern_post_mean": _safe_mean(topo["chern"][post_mask]),
        "gap_pre_min": float(np.min(topo["gap"][pre_mask])) if np.any(pre_mask) else float("nan"),
        "gap_post_min": float(np.min(topo["gap"][post_mask])) if np.any(post_mask) else float("nan"),
        "u_eff_pre_mean": _safe_mean(topo["u_eff"][pre_mask]),
        "u_eff_post_mean": _safe_mean(topo["u_eff"][post_mask]),
        "sig_top_pre_mean": sig_top_pre,
        "sig_top_post_mean": _safe_mean(sig_top[post_mask]),
        "sig_top_baseline_mean": float(precursor["baseline_mean"]),
        "sig_top_baseline_scale": float(precursor["baseline_scale"]),
        "sig_top_drop_z_max": float(precursor["max_drop_z"]),
        "sig_top_precursor_time": sig_top_precursor_time,
        "precursor_margin": precursor_margin,
        "precursor_lead_time": precursor_lead_time,
        "sig_top_half_time": sig_top_half_time,
        "half_margin": half_margin,
        "bott_loss_time": float(bott_loss_time) if bott_loss_observed else float("nan"),
        "lead_time": lead_time,
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "gmres_fails": float(out["gmres_fails"]),
    }
    return row


def _write_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(rows: List[Dict[str, float]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not rows:
        return

    sizes = sorted({int(row["nx"]) for row in rows})
    gains = sorted({float(row["qzw_pi_gain"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for nx in sizes:
        for gain in gains:
            subset = [row for row in rows if int(row["nx"]) == nx and float(row["qzw_pi_gain"]) == gain]
            if not subset:
                continue
            subset.sort(key=lambda row: float(row["init_phase_noise"]))
            x = [float(row["init_phase_noise"]) for row in subset]
            y_loss = [float(row["retention_time"]) for row in subset]
            y_precursor = [float(row["precursor_margin"]) for row in subset]
            y_half = [float(row["half_margin"]) for row in subset]
            label = f"n={nx}, pi_gain={gain:.2f}"
            axes[0].plot(x, y_loss, marker="o", label=label)
            axes[1].plot(x, y_precursor, marker="o", label=label)
            axes[2].plot(x, y_half, marker="o", label=label)

    axes[0].set_title("Bott Retention Time")
    axes[0].set_xlabel("Initial phase noise")
    axes[0].set_ylabel("Retention boundary time")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("pi_f Precursor Margin")
    axes[1].set_xlabel("Initial phase noise")
    axes[1].set_ylabel("Retention boundary - precursor time")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.5)

    axes[2].set_title("pi_f Half-Threshold Margin")
    axes[2].set_xlabel("Initial phase noise")
    axes[2].set_ylabel("Retention boundary - top-edge half time")
    axes[2].grid(alpha=0.3)
    axes[2].axhline(0.0, color="black", lw=0.8, alpha=0.5)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="HAFC topology retention sweep")
    parser.add_argument("--sizes", default="6,8", help="Comma-separated lattice sizes")
    parser.add_argument("--T", type=float, default=18.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--damage-time", type=float, default=9.0)
    parser.add_argument("--mass", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase-noise", default="0.00,0.02,0.04")
    parser.add_argument("--qzw-pi-gains", default="0.25,0.45")
    parser.add_argument("--qzw-entropy-gain", type=float, default=0.10)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--precursor-window", type=int, default=5)
    parser.add_argument("--precursor-z-threshold", type=float, default=2.5)
    parser.add_argument("--no-damage", action="store_true")
    parser.add_argument("--csv", default="hafc_topology_retention_sweep.csv")
    parser.add_argument("--plot", default="hafc_topology_retention_sweep.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    sizes = _parse_int_list(args.sizes)
    phase_noise_values = _parse_float_list(args.phase_noise)
    qzw_pi_gains = _parse_float_list(args.qzw_pi_gains)

    rows: List[Dict[str, float]] = []
    total = len(sizes) * len(phase_noise_values) * len(qzw_pi_gains)
    counter = 0
    for size in sizes:
        for phase_noise in phase_noise_values:
            for qzw_pi_gain in qzw_pi_gains:
                counter += 1
                damage_time = args.damage_time if not args.no_damage else args.T + args.dt
                print(
                    f"[{counter}/{total}] n={size}, noise={phase_noise:.3f}, "
                    f"pi_gain={qzw_pi_gain:.3f}, damage_time={damage_time:.2f}"
                )
                row = _run_case(
                    nx=size,
                    ny=size,
                    T=args.T,
                    dt=args.dt,
                    damage_time=damage_time,
                    mass=args.mass,
                    seed=args.seed,
                    init_phase_noise=phase_noise,
                    qzw_pi_gain=qzw_pi_gain,
                    qzw_entropy_gain=args.qzw_entropy_gain,
                    sample_stride=max(1, args.sample_stride),
                    precursor_window=max(1, args.precursor_window),
                    precursor_z_threshold=args.precursor_z_threshold,
                )
                rows.append(row)

    _write_csv(rows, args.csv)
    if not args.no_plot:
        _plot_results(rows, args.plot)

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float(row["retained_to_end"]),
            float(row["retention_time"]),
            float(row["gap_post_min"] if not math.isnan(float(row["gap_post_min"])) else row["gap_pre_min"]),
            float(row["precursor_margin"] if not math.isnan(float(row["precursor_margin"])) else -1.0),
        ),
        reverse=True,
    )
    print("\nTop retention configs")
    for row in rows_sorted[: min(6, len(rows_sorted))]:
        retention_label = (
            f"retained_to_end ({row['retention_time']:.2f})"
            if float(row["retained_to_end"]) > 0.5
            else f"loss @ {row['bott_loss_time']:.2f}"
        )
        gap_ref = row["gap_post_min"] if not math.isnan(float(row["gap_post_min"])) else row["gap_pre_min"]
        print(
            f"  n={int(row['nx'])}, noise={row['init_phase_noise']:.3f}, "
            f"pi_gain={row['qzw_pi_gain']:.3f}, retention={retention_label}, "
            f"gap_ref={gap_ref:.4f}, precursor_margin={row['precursor_margin']:.2f}, "
            f"half_margin={row['half_margin']:.2f}, drop_z_max={row['sig_top_drop_z_max']:.2f}"
        )

    positive_precursor = [
        row for row in rows
        if not math.isnan(float(row["precursor_lead_time"])) and float(row["precursor_lead_time"]) > 0.0
    ]
    if positive_precursor:
        best_lead = max(positive_precursor, key=lambda row: float(row["precursor_lead_time"]))
        print(
            "\nBest pi_f precursor: "
            f"n={int(best_lead['nx'])}, noise={best_lead['init_phase_noise']:.3f}, "
            f"pi_gain={best_lead['qzw_pi_gain']:.3f}, precursor_lead={best_lead['precursor_lead_time']:.2f}, "
            f"drop_z_max={best_lead['sig_top_drop_z_max']:.2f}"
        )
    else:
        print("\nNo positive rolling-z pi_f precursor was detected at the current threshold.")

    print(f"\nCSV saved to: {args.csv}")
    if not args.no_plot:
        print(f"Plot saved to: {args.plot}")


if __name__ == "__main__":
    main()