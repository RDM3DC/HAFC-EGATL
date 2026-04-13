"""HAFC-EGATL Interactive Web App.

An interactive Streamlit interface for the Hybrid Adaptive Field Computer
built on an Entropy-Gated Adaptive Topological Lattice (EGATL).

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Ensure solver package is importable when running inside the repo directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver.egatl import (
    EGATLParams,
    EntropyParams,
    RulerParams,
    QWZLattice,
    build_qwz_lattice,
    run_recovery_protocol,
    summarize_recovery,
    effective_transfer,
    top_edge_fraction,
    proxy_chern_series,
    boundary_signature_series,
    top_edge_signature_series,
    plaquette_signature_series,
    qwz_topology_series,
    plaquette_signatures,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HAFC-EGATL Interactive Computer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_PRESET: Dict[str, Any] = {
    "nx": 6,
    "ny": 6,
    "T": 30.0,
    "dt": 0.1,
    "seed": 0,
    "apply_damage": True,
    "damage_time": 12.0,
    "damage_scenario": "central_strip",
    "mass": -1.0,
    "alpha0": 2.5,
    "qzw_pi_gain": 0.45,
    "qzw_entropy_gain": 0.10,
    "init_phase_noise": 0.02,
    "S_init": 1.5,
    "gamma": 0.20,
    "alpha_pi": 0.15,
}

BALANCED_RECOVERY_PRESET: Dict[str, Any] = {
    **DEFAULT_PRESET,
    "mass": -2.5,
    "alpha0": 4.0,
    "qzw_entropy_gain": 0.0,
    "alpha_pi": 0.08,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": DEFAULT_PRESET,
    "Balanced Recovery": BALANCED_RECOVERY_PRESET,
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "Default": "Reference configuration from the original UI.",
    "Balanced Recovery": "Robust 20-seed setting with near-unity transfer recovery and zero Bott drift.",
}

DAMAGE_SCENARIO_LABELS: Dict[str, str] = {
    "central_strip": "Central Strip",
    "center_cross": "Center Cross",
    "center_block": "Center Block",
    "top_edge": "Top Edge Cut",
    "source_corner": "Source Corner Hit",
    "random_bonds": "Random 30% Bonds",
}

DAMAGE_SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "central_strip": "Vertical band through the middle of the lattice. This matches the original built-in damage event.",
    "center_cross": "Both the central row and central column are hit at once.",
    "center_block": "A compact square-like region around the center is damaged.",
    "top_edge": "The top boundary edge is cut to stress edge transport directly.",
    "source_corner": "Bonds near the source corner are damaged to test recovery from a localized ingress hit.",
    "random_bonds": "A seed-dependent random 30% subset of bonds is knocked down.",
}


def _ensure_control_state() -> None:
    if "controls_initialized" not in st.session_state:
        for key, value in DEFAULT_PRESET.items():
            st.session_state[key] = value
        st.session_state["preset_name"] = "Default"
        st.session_state["controls_initialized"] = True
    else:
        for key, value in DEFAULT_PRESET.items():
            st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset_name", "Default")


def _apply_preset(preset_name: str) -> None:
    for key, value in PRESETS[preset_name].items():
        st.session_state[key] = value


def _consume_pending_preset() -> None:
    pending_preset = st.session_state.pop("pending_preset", None)
    if pending_preset in PRESETS:
        _apply_preset(pending_preset)
    pending_damage_scenario = st.session_state.pop("pending_damage_scenario", None)
    if pending_damage_scenario in DAMAGE_SCENARIO_LABELS:
        st.session_state["damage_scenario"] = pending_damage_scenario


def _clamp_numeric_value(value: Any, min_value: Any, max_value: Any, cast) -> Any:
    numeric_value = cast(value)
    if numeric_value < min_value:
        numeric_value = min_value
    if numeric_value > max_value:
        numeric_value = max_value
    return cast(numeric_value)


def _sync_numeric_control(key: str, source_key: str, cast, min_value: Any, max_value: Any) -> None:
    value = _clamp_numeric_value(st.session_state[source_key], min_value, max_value, cast)
    st.session_state[key] = value
    st.session_state[f"{key}__slider"] = value
    st.session_state[f"{key}__input"] = value


def _slider_with_input(
    label: str,
    *,
    key: str,
    min_value: Any,
    max_value: Any,
    step: Any,
    cast,
    display_format: Optional[str] = None,
    disabled: bool = False,
    default: Optional[Any] = None,
) -> Any:
    initial_value = min_value if default is None else default
    value = _clamp_numeric_value(st.session_state.get(key, initial_value), min_value, max_value, cast)
    st.session_state[key] = value
    st.session_state[f"{key}__slider"] = value
    st.session_state[f"{key}__input"] = value

    slider_kwargs: Dict[str, Any] = {}
    if display_format is not None:
        slider_kwargs["format"] = display_format

    slider_col, input_col = st.columns([4, 1.35])
    with slider_col:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=f"{key}__slider",
            disabled=disabled,
            on_change=_sync_numeric_control,
            args=(key, f"{key}__slider", cast, min_value, max_value),
            **slider_kwargs,
        )

    input_kwargs: Dict[str, Any] = {
        "label_visibility": "collapsed",
        "min_value": min_value,
        "max_value": max_value,
        "step": step,
        "key": f"{key}__input",
        "disabled": disabled,
        "on_change": _sync_numeric_control,
        "args": (key, f"{key}__input", cast, min_value, max_value),
    }
    if display_format is not None:
        input_kwargs["format"] = display_format

    with input_col:
        st.number_input(f"{label} value", **input_kwargs)

    return cast(st.session_state[key])


def _build_run_kwargs(
    *,
    nx: int,
    ny: int,
    T: float,
    dt: float,
    seed: int,
    damage_time: float,
    damage_scenario: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "nx": int(nx),
        "ny": int(ny),
        "T": float(T),
        "dt": float(dt),
        "seed": int(seed),
        "damage_time": float(damage_time),
        "damage_scenario": str(damage_scenario),
        "mass": float(config["mass"]),
        "alpha0": float(config["alpha0"]),
        "qzw_pi_gain": float(config["qzw_pi_gain"]),
        "qzw_entropy_gain": float(config["qzw_entropy_gain"]),
        "init_phase_noise": float(config["init_phase_noise"]),
        "S_init": float(config["S_init"]),
        "gamma": float(config["gamma"]),
        "alpha_pi": float(config["alpha_pi"]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

_ensure_control_state()
_consume_pending_preset()

with st.sidebar:
    st.title("⚙️ HAFC-EGATL Controls")
    st.caption("Hybrid Adaptive Field Computer  |  Entropy-Gated Topological Lattice")
    st.caption("Drag the sliders or type exact values in the boxes on the right.")

    preset_name = st.selectbox("Preset", list(PRESETS), key="preset_name")
    st.caption(PRESET_DESCRIPTIONS[preset_name])
    if st.button("Apply preset", width="stretch"):
        _apply_preset(preset_name)
        st.rerun()

    st.subheader("Lattice & Time")
    nx = _slider_with_input("Lattice width (nx)", key="nx", min_value=3, max_value=10, step=1, cast=int)
    ny = _slider_with_input("Lattice height (ny)", key="ny", min_value=3, max_value=10, step=1, cast=int)
    T = _slider_with_input(
        "Simulation time T",
        key="T",
        min_value=5.0,
        max_value=60.0,
        step=1.0,
        cast=float,
        display_format="%.1f",
    )
    dt = _slider_with_input(
        "Time step dt",
        key="dt",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        cast=float,
        display_format="%.3f",
    )
    seed = st.number_input("Random seed", 0, 9999, key="seed", step=1)

    st.subheader("Damage & Recovery")
    max_damage_time = max(0.5, float(T) - 0.5)
    if st.session_state["damage_time"] > max_damage_time:
        st.session_state["damage_time"] = max_damage_time

    apply_damage = st.checkbox("Apply damage event", key="apply_damage")
    damage_time = _slider_with_input(
        "Damage time t_d",
        key="damage_time",
        min_value=0.5,
        max_value=max_damage_time,
        step=0.5,
        cast=float,
        display_format="%.2f",
        disabled=not apply_damage,
    )
    damage_scenario = st.selectbox(
        "Damage scenario",
        options=list(DAMAGE_SCENARIO_LABELS),
        format_func=lambda key: DAMAGE_SCENARIO_LABELS[key],
        key="damage_scenario",
        disabled=not apply_damage,
    )
    st.caption(DAMAGE_SCENARIO_DESCRIPTIONS[damage_scenario])
    if not apply_damage:
        damage_time = T + dt  # effectively never

    st.subheader("Physics Parameters")
    mass = _slider_with_input("QWZ mass M", key="mass", min_value=-3.0, max_value=1.0, step=0.1, cast=float, display_format="%.2f")
    alpha0 = _slider_with_input("ARP gain α₀", key="alpha0", min_value=0.5, max_value=6.0, step=0.1, cast=float, display_format="%.2f")
    qzw_pi_gain = _slider_with_input("QWZ π gain", key="qzw_pi_gain", min_value=0.0, max_value=1.0, step=0.05, cast=float, display_format="%.2f")
    qzw_entropy_gain = _slider_with_input("QWZ entropy gain", key="qzw_entropy_gain", min_value=0.0, max_value=0.5, step=0.01, cast=float, display_format="%.2f")
    init_phase_noise = _slider_with_input("Init phase noise", key="init_phase_noise", min_value=0.0, max_value=0.2, step=0.005, cast=float, display_format="%.3f")

    st.subheader("Entropy & Ruler")
    S_init = _slider_with_input("Initial entropy S₀", key="S_init", min_value=0.5, max_value=4.0, step=0.1, cast=float, display_format="%.2f")
    gamma = _slider_with_input("Entropy relaxation γ", key="gamma", min_value=0.01, max_value=1.0, step=0.01, cast=float, display_format="%.2f")
    alpha_pi = _slider_with_input("Ruler drive α_π", key="alpha_pi", min_value=0.01, max_value=0.5, step=0.01, cast=float, display_format="%.2f")

    run_btn = st.button("▶ Run Simulation", type="primary", width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HAFC-EGATL Interactive Computer")
st.markdown(
    "**Hybrid Adaptive Field Computer** on an *Entropy-Gated Adaptive Topological Lattice*. "
    "Configure the parameters in the sidebar and click **▶ Run Simulation**."
)

quick_cols = st.columns(3)
with quick_cols[0]:
    st.caption("Best first run")
    if st.button("Run Canonical Demo", key="quickstart_demo", width="stretch"):
        st.session_state["pending_preset"] = "Balanced Recovery"
        st.session_state["pending_damage_scenario"] = "central_strip"
        st.session_state["autorun_main"] = True
        st.rerun()
with quick_cols[1]:
    st.caption("See why the preset wins")
    if st.button("Compare Presets", key="quickstart_compare", width="stretch"):
        st.session_state["autorun_compare"] = True
        st.rerun()
with quick_cols[2]:
    st.caption("Check repeatability")
    if st.button("Stress Test Current Setup", key="quickstart_sweep", width="stretch"):
        st.session_state["autorun_sweep"] = True
        st.rerun()

with st.expander("How to read the dashboard"):
    st.markdown(
        "- **Y_eff recovery**: how well end-to-end transfer returns after damage. Values near `1.0` mean the pathway recovers.\n"
        "- **Sig boundary / top-edge recovery**: how well the edge-localized Hall-like structure comes back.\n"
        "- **Bott Δ**: drift in the real-space topological invariant. Values near `0` are the target.\n"
        "- **GMRES fails**: numerical stress indicator. `0` is what you want."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run and cache simulation
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_sim(
    nx: int, ny: int, T: float, dt: float, seed: int,
    damage_time: float, damage_scenario: str, mass: float,
    alpha0: float, qzw_pi_gain: float, qzw_entropy_gain: float,
    init_phase_noise: float, S_init: float, gamma: float, alpha_pi: float,
) -> tuple:
    eg = EGATLParams(
        alpha0=alpha0, S_c=0.3, dS=0.5, mu0=0.25, S0=2.0,
        lambda_s=0.05, g_min=0.01, g_max=5.0,
        g_imag_max=3.0, budget_re=float(nx * ny * 3.5),
        init_phase_noise=init_phase_noise,
        qzw_pi_gain=qzw_pi_gain,
        qzw_entropy_gain=qzw_entropy_gain,
    )
    ent = EntropyParams(
        S_init=S_init, S_eq=1.0, gamma=gamma,
        kappa_slip=0.10, Tij=1.0,
    )
    ruler = RulerParams(
        pi0=math.pi, pi_init=math.pi,
        alpha_pi=alpha_pi, mu_pi=0.15,
        pi_min=0.5, pi_max=12.0,
    )
    lattice, out = run_recovery_protocol(
        nx=nx, ny=ny, T=T, dt=dt, seed=seed,
        damage_time=damage_time, mass=mass,
        damage_factor=1e-4,
        damage_scenario=damage_scenario,
        eg=eg, ent=ent, ruler=ruler,
    )
    return lattice, out


@st.cache_data(show_spinner=False)
def _compute_topo(
    g_bytes: bytes,
    nx: int, ny: int,
    pi_a_bytes: bytes, S_bytes: bytes,
    mass: float, qzw_pi_gain: float, qzw_entropy_gain: float,
    n_bonds: int, K: int,
) -> Dict[str, Any]:
    g = np.frombuffer(g_bytes, dtype=np.complex128).reshape(K, n_bonds)
    pi_a = np.frombuffer(pi_a_bytes, dtype=np.float64)
    S_arr = np.frombuffer(S_bytes, dtype=np.float64)
    lat = build_qwz_lattice(nx, ny, mass)
    # Down-sample for Bott (expensive) but keep full for others
    stride = max(1, K // 60)
    qwz = qwz_topology_series(
        g[::stride], lat, pi_a[::stride], S_arr[::stride],
        mass0=mass, pi_gain=qzw_pi_gain, entropy_gain=qzw_entropy_gain,
    )
    chern = proxy_chern_series(g, lat)
    sig_bnd = boundary_signature_series(g, lat, pi_a)
    sig_top = top_edge_signature_series(g, lat, pi_a)
    sig_plaq = plaquette_signature_series(g, lat, pi_a)
    return {
        "qwz": qwz,
        "chern": chern,
        "sig_bnd": sig_bnd,
        "sig_top": sig_top,
        "sig_plaq": sig_plaq,
        "stride": stride,
    }


def _package_run_outputs(
    lattice: QWZLattice,
    out: Dict[str, Any],
    damage_time: float,
    mass: float,
    qzw_pi_gain: float,
    qzw_entropy_gain: float,
) -> Dict[str, Any]:
    topo = _compute_topo(
        g_bytes=out["g"].tobytes(),
        nx=lattice.nx,
        ny=lattice.ny,
        pi_a_bytes=out["pi_a"].tobytes(),
        S_bytes=out["S"].tobytes(),
        mass=mass,
        qzw_pi_gain=qzw_pi_gain,
        qzw_entropy_gain=qzw_entropy_gain,
        n_bonds=lattice.n_bonds,
        K=len(out["t"]),
    )
    summary = summarize_recovery(
        out,
        lattice,
        damage_time,
        mass0=mass,
        pi_gain=qzw_pi_gain,
        entropy_gain=qzw_entropy_gain,
    )
    y_eff = np.array([
        effective_transfer(out["phi"][k], lattice.source_cell, lattice.sink_cell)
        for k in range(len(out["t"]))
    ])
    return {
        "t": out["t"],
        "S": out["S"],
        "pi_a": out["pi_a"],
        "Y_eff": y_eff,
        "sig_bnd": boundary_signature_series(out["g"], lattice, out["pi_a"]),
        "sig_top": top_edge_signature_series(out["g"], lattice, out["pi_a"]),
        "bott_t": out["t"][::topo["stride"]],
        "bott": topo["qwz"]["bott"],
        "gap_t": out["t"][::topo["stride"]],
        "gap": topo["qwz"]["gap"],
        "summary": summary,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _best_effort_git_commit(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _build_run_report(
    lattice: QWZLattice,
    out: Dict[str, Any],
    run_params: Dict[str, Any],
    topo_data: Dict[str, Any],
    summary: Dict[str, Any],
    compare_results: Optional[Dict[str, Any]] = None,
    robustness_rows: Optional[list[Dict[str, Any]]] = None,
    robustness_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": "hafc-egatl.run-report",
        "schema_version": "0.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "app": "app.py",
            "solver": "solver.egatl.run_recovery_protocol",
            "git_commit": _best_effort_git_commit(os.path.dirname(os.path.abspath(__file__))),
        },
        "platform": {
            "python": sys.version.split()[0],
            "platform": os.name,
        },
        "config": {
            "lattice": {
                "nx": int(run_params["nx"]),
                "ny": int(run_params["ny"]),
            },
            "timing": {
                "T": float(run_params["T"]),
                "dt": float(run_params["dt"]),
                "steps": int(len(out["t"])),
            },
            "seed": int(run_params["seed"]),
            "damage": {
                "enabled": bool(run_params["has_damage"]),
                "time": float(run_params["damage_time"]),
                "scenario": str(run_params["damage_scenario"]),
                "factor": 1e-4,
            },
            "physics": {
                "mass": float(run_params["mass"]),
                "alpha0": float(run_params["alpha0"]),
                "qzw_pi_gain": float(run_params["qzw_pi_gain"]),
                "qzw_entropy_gain": float(run_params["qzw_entropy_gain"]),
                "init_phase_noise": float(run_params["init_phase_noise"]),
                "S_init": float(run_params["S_init"]),
                "gamma": float(run_params["gamma"]),
                "alpha_pi": float(run_params["alpha_pi"]),
            },
        },
        "lattice": {
            "nodes": int(lattice.n_nodes),
            "bonds": int(lattice.n_bonds),
            "plaquettes": int(len(lattice.plaquettes)),
            "source_cell": int(lattice.source_cell),
            "sink_cell": int(lattice.sink_cell),
        },
        "numerics": {
            "integrator": "explicit_euler",
            "topology_stride": int(topo_data["stride"]),
            "gmres_fails": int(summary["gmres_fails"]),
        },
        "observables": {
            key: _json_scalar(value)
            for key, value in summary.items()
        },
        "final_state": {
            "S": float(out["S"][-1]),
            "pi_a": float(out["pi_a"][-1]),
            "pi_a_over_pi": float(out["pi_a"][-1] / math.pi),
        },
    }

    if compare_results is not None:
        report["compare"] = {
            "has_damage": bool(compare_results["has_damage"]),
            "damage_time": float(compare_results["damage_time"]),
            "shared": {
                "nx": int(compare_results["shared"]["nx"]),
                "ny": int(compare_results["shared"]["ny"]),
                "T": float(compare_results["shared"]["T"]),
                "dt": float(compare_results["shared"]["dt"]),
                "seed": int(compare_results["shared"]["seed"]),
                "damage_scenario": str(compare_results["shared"]["damage_scenario"]),
            },
            "presets": {
                preset_name: {
                    key: _json_scalar(value)
                    for key, value in preset_data["summary"].items()
                }
                for preset_name, preset_data in compare_results["results"].items()
            },
        }

    if robustness_rows:
        rows = [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in robustness_rows
        ]
        report["robustness"] = {
            "meta": {
                key: _json_scalar(value)
                for key, value in (robustness_meta or {}).items()
            },
            "aggregate": {
                "median_transfer_recovery": float(np.median([row["transfer_recovery"] for row in rows])),
                "worst_transfer_recovery": float(min(row["transfer_recovery"] for row in rows)),
                "max_abs_bott_delta": float(max(abs(row["bott_delta"]) for row in rows)),
                "total_gmres_fails": int(sum(int(row["gmres_fails"]) for row in rows)),
            },
            "rows": rows,
        }

    return report


@st.cache_data(show_spinner=False)
def _run_preset_comparison(
    nx: int,
    ny: int,
    T: float,
    dt: float,
    seed: int,
    apply_damage: bool,
    damage_time: float,
    damage_scenario: str,
) -> Dict[str, Any]:
    resolved_damage_time = float(damage_time if apply_damage else T + dt)
    results: Dict[str, Dict[str, Any]] = {}

    for preset_name, preset_config in PRESETS.items():
        lattice, out = _run_sim(
            **_build_run_kwargs(
                nx=nx,
                ny=ny,
                T=T,
                dt=dt,
                seed=seed,
                damage_time=resolved_damage_time,
                damage_scenario=damage_scenario,
                config=preset_config,
            )
        )
        results[preset_name] = _package_run_outputs(
            lattice,
            out,
            resolved_damage_time,
            float(preset_config["mass"]),
            float(preset_config["qzw_pi_gain"]),
            float(preset_config["qzw_entropy_gain"]),
        )

    return {
        "damage_time": resolved_damage_time,
        "has_damage": bool(apply_damage),
        "shared": {
            "nx": int(nx),
            "ny": int(ny),
            "T": float(T),
            "dt": float(dt),
            "seed": int(seed),
            "damage_scenario": str(damage_scenario),
        },
        "results": results,
    }


@st.cache_data(show_spinner=False)
def _run_robustness_sweep(
    nx: int,
    ny: int,
    T: float,
    dt: float,
    seed_start: int,
    seed_count: int,
    apply_damage: bool,
    damage_time: float,
    damage_scenario: str,
    mass: float,
    alpha0: float,
    qzw_pi_gain: float,
    qzw_entropy_gain: float,
    init_phase_noise: float,
    S_init: float,
    gamma: float,
    alpha_pi: float,
) -> list[Dict[str, float]]:
    resolved_damage_time = float(damage_time if apply_damage else T + dt)
    config = {
        "mass": mass,
        "alpha0": alpha0,
        "qzw_pi_gain": qzw_pi_gain,
        "qzw_entropy_gain": qzw_entropy_gain,
        "init_phase_noise": init_phase_noise,
        "S_init": S_init,
        "gamma": gamma,
        "alpha_pi": alpha_pi,
    }
    rows: list[Dict[str, float]] = []

    for offset in range(int(seed_count)):
        seed_value = int(seed_start) + offset
        lattice, out = _run_sim(
            **_build_run_kwargs(
                nx=nx,
                ny=ny,
                T=T,
                dt=dt,
                seed=seed_value,
                damage_time=resolved_damage_time,
                damage_scenario=damage_scenario,
                config=config,
            )
        )
        summary = summarize_recovery(
            out,
            lattice,
            resolved_damage_time,
            mass0=float(mass),
            pi_gain=float(qzw_pi_gain),
            entropy_gain=float(qzw_entropy_gain),
        )
        rows.append(
            {
                "seed": float(seed_value),
                "transfer_recovery": float(summary["transfer_recovery"]),
                "sig_boundary_recovery": float(summary["sig_boundary_recovery"]),
                "sig_top_edge_recovery": float(summary["sig_top_edge_recovery"]),
                "bott_delta": float(summary["bott_delta"]),
                "gap_post": float(summary["qwz_gap_post"]),
                "gmres_fails": float(summary["gmres_fails"]),
            }
        )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers (matplotlib → st.pyplot)
# ─────────────────────────────────────────────────────────────────────────────

def _vline_kwargs(color: str = "red") -> dict:
    return dict(color=color, linestyle="--", alpha=0.55, linewidth=1.4)


def _plot_dynamics(t, out, lattice, damage_time, has_damage):
    """Plot entropy, pi_a and Y_eff dynamics."""
    import matplotlib.pyplot as plt

    K = len(t)
    Yeff = np.array([
        effective_transfer(out["phi"][k], lattice.source_cell, lattice.sink_cell)
        for k in range(K)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    fig.suptitle("System Dynamics", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(t, out["S"], color="sienna", lw=1.5, label="Entropy S")
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs())
    ax.set_xlabel("Time"); ax.set_ylabel("S"); ax.set_title("Entropy"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, out["pi_a"] / math.pi, color="purple", lw=1.5)
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs())
    ax.set_xlabel("Time"); ax.set_ylabel("π_a / π"); ax.set_title("Adaptive π ruler"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, Yeff, color="steelblue", lw=1.5)
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs(), label="damage")
    ax.set_xlabel("Time"); ax.set_ylabel("Y_eff"); ax.set_title("Effective Transfer"); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_topology(t, topo_data, damage_time, has_damage):
    """Plot Bott, Chern, gap, signatures."""
    import matplotlib.pyplot as plt

    qwz = topo_data["qwz"]
    stride = topo_data["stride"]
    t_s = t[::stride]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    fig.suptitle("Topological Readout", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(t_s, qwz["bott"], color="black", lw=1.8, label="Bott index", zorder=3)
    ax.plot(t_s, qwz["chern"], color="teal", lw=1.2, ls="--", label="QWZ Chern")
    ax.plot(t, topo_data["chern"], color="orange", lw=1.0, alpha=0.6, label="Chern proxy")
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs())
    ax.set_xlabel("Time"); ax.set_ylabel("Index"); ax.set_title("Bott & Chern")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_s, qwz["gap"], color="darkred", lw=1.5, label="Spectral gap")
    ax.plot(t_s, qwz["u_eff"], color="slateblue", lw=1.2, ls="--", label="u_eff")
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs())
    ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.set_title("Gap & u_eff")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, topo_data["sig_bnd"], color="royalblue", lw=1.4, label="Boundary sig.")
    ax.plot(t, topo_data["sig_top"], color="crimson", lw=1.4, label="Top-edge sig.")
    ax.plot(t, topo_data["sig_plaq"], color="forestgreen", lw=1.2, alpha=0.8, label="Plaquette sig.")
    if has_damage:
        ax.axvline(damage_time, **_vline_kwargs())
    ax.set_xlabel("Time"); ax.set_ylabel("Signature"); ax.set_title("π_f Loop Signatures")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_lattice(lattice, g_snapshot, title: str = "Conductance network"):
    """Draw lattice with bond widths ∝ |g_re| and colour ∝ |g_im|."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    g_re = np.abs(np.real(g_snapshot))
    g_im = np.abs(np.imag(g_snapshot))
    g_re_max = max(float(g_re.max()), 1e-6)
    g_im_max = max(float(g_im.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title, fontsize=10)
    cmap = plt.get_cmap("plasma")

    for e, (i, j) in enumerate(lattice.bonds):
        xi, yi = lattice.node_x[i], lattice.node_y[i]
        xj, yj = lattice.node_x[j], lattice.node_y[j]
        lw = 0.5 + 4.5 * (g_re[e] / g_re_max)
        col = cmap(g_im[e] / g_im_max)
        ax.plot([xi, xj], [yi, yj], color=col, lw=lw, alpha=0.85)

    ax.scatter(lattice.node_x, lattice.node_y, s=30, color="k", zorder=5)
    ax.scatter(
        [lattice.node_x[lattice.source_cell]], [lattice.node_y[lattice.source_cell]],
        s=120, color="lime", zorder=6, label="Source",
    )
    ax.scatter(
        [lattice.node_x[lattice.sink_cell]], [lattice.node_y[lattice.sink_cell]],
        s=120, color="red", zorder=6, label="Sink",
    )
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=7, loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, g_im_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.7, label="|g_im| (Hall)")
    plt.tight_layout()
    return fig


def _plot_plaquettes(lattice, g_snapshot, pi_a_val, title: str):
    """Heat-map of plaquette signatures."""
    import matplotlib.pyplot as plt

    sigs = plaquette_signatures(g_snapshot, lattice, float(pi_a_val))
    npx, npy = lattice.nx - 1, lattice.ny - 1
    grid = sigs.reshape(npy, npx)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, origin="lower", cmap="inferno",
                   aspect="equal", extent=[0, npx, 0, npy])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Plaquette sig.")
    plt.tight_layout()
    return fig


def _plot_compare_dashboard(compare_data: Dict[str, Any]):
    import matplotlib.pyplot as plt

    palette = {
        "Default": "#4c566a",
        "Balanced Recovery": "#2e8b57",
    }
    has_damage = bool(compare_data["has_damage"])
    damage_time = float(compare_data["damage_time"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Preset Comparison", fontsize=13, fontweight="bold")

    for name, run_data in compare_data["results"].items():
        color = palette.get(name, None)
        axes[0, 0].plot(run_data["t"], run_data["Y_eff"], lw=1.8, color=color, label=name)
        axes[0, 1].plot(run_data["t"], run_data["S"], lw=1.8, color=color, label=name)
        axes[1, 0].plot(run_data["t"], run_data["pi_a"] / math.pi, lw=1.8, color=color, label=name)
        axes[1, 1].plot(run_data["bott_t"], run_data["bott"], lw=1.8, color=color, label=name)

    for ax in axes.ravel():
        if has_damage:
            ax.axvline(damage_time, **_vline_kwargs())
        ax.grid(alpha=0.3)

    axes[0, 0].set_title("Effective Transfer")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Y_eff")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Entropy")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("S")

    axes[1, 0].set_title("Adaptive π Ruler")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("π_a / π")

    axes[1, 1].set_title("Bott Index")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Bott")

    plt.tight_layout()
    return fig


def _plot_robustness_dashboard(df):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Robustness Sweep", fontsize=13, fontweight="bold")

    seeds = df["seed"].to_numpy()

    axes[0, 0].plot(seeds, df["transfer_recovery"], marker="o", color="steelblue", lw=1.4)
    axes[0, 0].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Transfer Recovery")
    axes[0, 0].set_xlabel("Seed")
    axes[0, 0].set_ylabel("Recovery")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(seeds, df["sig_boundary_recovery"], marker="o", color="royalblue", lw=1.4, label="Boundary")
    axes[0, 1].plot(seeds, df["sig_top_edge_recovery"], marker="o", color="crimson", lw=1.4, label="Top edge")
    axes[0, 1].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Signature Recovery")
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Recovery")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(seeds, df["bott_delta"], marker="o", color="darkgreen", lw=1.4)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Bott Drift")
    axes[1, 0].set_xlabel("Seed")
    axes[1, 0].set_ylabel("Δ Bott")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(seeds, df["gap_post"], marker="o", color="darkred", lw=1.4)
    axes[1, 1].set_title("Post-settle Spectral Gap")
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Gap")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

auto_run_main = bool(st.session_state.pop("autorun_main", False))

if run_btn or auto_run_main:
    with st.spinner("Running HAFC-EGATL simulation…"):
        t0 = time.time()
        lattice, out = _run_sim(
            nx=nx, ny=ny, T=T, dt=dt, seed=int(seed),
            damage_time=damage_time, damage_scenario=damage_scenario, mass=mass,
            alpha0=alpha0, qzw_pi_gain=qzw_pi_gain,
            qzw_entropy_gain=qzw_entropy_gain,
            init_phase_noise=init_phase_noise,
            S_init=S_init, gamma=gamma, alpha_pi=alpha_pi,
        )
    st.success(f"Simulation complete in {time.time()-t0:.1f} s  |  "
               f"{len(out['t'])} steps  ·  {lattice.n_bonds} bonds  ·  "
               f"{len(lattice.plaquettes)} plaquettes")
    st.session_state["lattice"] = lattice
    st.session_state["out"] = out
    st.session_state["run_params"] = {
        "nx": int(nx),
        "ny": int(ny),
        "T": float(T),
        "dt": float(dt),
        "seed": int(seed),
        "damage_time": damage_time,
        "has_damage": apply_damage,
        "damage_scenario": damage_scenario,
        "mass": mass,
        "alpha0": alpha0,
        "qzw_pi_gain": qzw_pi_gain,
        "qzw_entropy_gain": qzw_entropy_gain,
        "init_phase_noise": init_phase_noise,
        "S_init": S_init,
        "gamma": gamma,
        "alpha_pi": alpha_pi,
    }

if "out" not in st.session_state or "run_params" not in st.session_state:
    st.info("👈 Configure parameters and click **▶ Run Simulation** to get started.")
    st.stop()

# ── Retrieve session data ────────────────────────────────────────────────────
lattice: QWZLattice = st.session_state["lattice"]
out: Dict[str, Any] = st.session_state["out"]
run_params: Dict[str, Any] = st.session_state["run_params"]
damage_time_sess: float = float(run_params["damage_time"])
has_damage: bool = bool(run_params["has_damage"])
run_mass: float = float(run_params["mass"])
run_qzw_pi_gain: float = float(run_params["qzw_pi_gain"])
run_qzw_entropy_gain: float = float(run_params["qzw_entropy_gain"])
t = out["t"]
K = len(t)

# ── Compute topology (cached) ────────────────────────────────────────────────
with st.spinner("Computing topological invariants…"):
    topo_data = _compute_topo(
        g_bytes=out["g"].tobytes(),
        nx=lattice.nx, ny=lattice.ny,
        pi_a_bytes=out["pi_a"].tobytes(),
        S_bytes=out["S"].tobytes(),
        mass=run_mass,
        qzw_pi_gain=run_qzw_pi_gain,
        qzw_entropy_gain=run_qzw_entropy_gain,
        n_bonds=lattice.n_bonds, K=K,
    )

# ── Summary metrics row ──────────────────────────────────────────────────────
summ = summarize_recovery(
    out,
    lattice,
    damage_time_sess,
    mass0=run_mass,
    pi_gain=run_qzw_pi_gain,
    entropy_gain=run_qzw_entropy_gain,
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Y_eff recovery", f"{summ['transfer_recovery']:.3f}",
            delta=f"{summ['transfer_post']-summ['transfer_pre']:.3f}")
col2.metric("Sig boundary rec.", f"{summ['sig_boundary_recovery']:.3f}")
col3.metric("Sig top-edge rec.", f"{summ['sig_top_edge_recovery']:.3f}")
col4.metric("Bott Δ (post–pre)", f"{summ['bott_delta']:+.2f}")
col5.metric("GMRES fails", f"{int(summ['gmres_fails'])}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dyn, tab_topo, tab_net, tab_plaq, tab_compare, tab_robust, tab_data = st.tabs([
    "📈 Dynamics", "⚛️ Topology", "🔗 Network", "⬛ Plaquettes", "🆚 Compare", "🛡️ Robustness", "📋 Data",
])

with tab_dyn:
    st.subheader("System Dynamics")
    fig_dyn = _plot_dynamics(t, out, lattice, damage_time_sess, has_damage)
    st.pyplot(fig_dyn, width="stretch")

    with st.expander("Pre / Post comparison table"):
        import pandas as pd
        rows = [
            ("Y_eff transfer", summ["transfer_pre"], summ["transfer_post"], summ["transfer_recovery"]),
            ("Boundary current", summ["boundary_pre"], summ["boundary_post"], None),
            ("Top-edge current", summ["top_edge_pre"], summ["top_edge_post"], None),
            ("Sig boundary", summ["sig_boundary_pre"], summ["sig_boundary_post"], summ["sig_boundary_recovery"]),
            ("Sig top-edge", summ["sig_top_edge_pre"], summ["sig_top_edge_post"], summ["sig_top_edge_recovery"]),
            ("Sig plaquette", summ["sig_plaquette_pre"], summ["sig_plaquette_post"], summ["sig_plaquette_recovery"]),
            ("Chern proxy", summ["chern_pre"], summ["chern_post"], None),
            ("QWZ Chern", summ["qwz_chern_pre"], summ["qwz_chern_post"], None),
            ("Bott index", summ["bott_pre"], summ["bott_post"], None),
            ("Spectral gap", summ["qwz_gap_pre"], summ["qwz_gap_post"], None),
            ("u_eff", summ["qwz_u_eff_pre"], summ["qwz_u_eff_post"], None),
            ("Entropy S", None, summ["final_S"], None),
            ("π_a / π", None, summ["final_pi_a"] / math.pi, None),
        ]
        df = pd.DataFrame(
            [(r, f"{v:.4f}" if v is not None else "–", f"{w:.4f}" if w is not None else "–",
              f"{x:.4f}" if x is not None else "–")
             for r, v, w, x in rows],
            columns=["Metric", "Pre", "Post", "Recovery"],
        )
        st.dataframe(df, width="stretch", hide_index=True)

with tab_topo:
    st.subheader("Topological Readout")
    fig_topo = _plot_topology(t, topo_data, damage_time_sess, has_damage)
    st.pyplot(fig_topo, width="stretch")

    with st.expander("Topological summary"):
        st.markdown(f"""
| Invariant | Pre-damage | Post-settle | Δ |
|-----------|-----------|------------|---|
| **Bott index** | {summ['bott_pre']:.3f} | {summ['bott_post']:.3f} | {summ['bott_delta']:+.3f} |
| **QWZ Chern** | {summ['qwz_chern_pre']:.3f} | {summ['qwz_chern_post']:.3f} | {summ['qwz_chern_delta']:+.3f} |
| **Chern proxy** | {summ['chern_pre']:.3f} | {summ['chern_post']:.3f} | {summ['chern_delta']:+.3f} |
| **Spectral gap** | {summ['qwz_gap_pre']:.4f} | {summ['qwz_gap_post']:.4f} | — |
| **Bott int. error** | {summ['bott_integer_error_pre']:.4f} | {summ['bott_integer_error_post']:.4f} | — |
        """)

with tab_net:
    st.subheader("Conductance Network Snapshots")

    n_snaps = 4
    snap_times = np.linspace(0.0, float(t[-1]), n_snaps)
    snap_cols = st.columns(n_snaps)
    for ci, ts in enumerate(snap_times):
        k = int(np.argmin(np.abs(t - ts)))
        label = f"t = {t[k]:.1f}"
        if has_damage and abs(t[k] - damage_time_sess) < 2 * (t[-1] / len(t)):
            label += " ⚡"
        fig_n = _plot_lattice(lattice, out["g"][k], title=label)
        snap_cols[ci].pyplot(fig_n, width="stretch")

    st.caption(
        "Bond width ∝ |g_re| (real conductance).  "
        "Bond colour (dark→bright) ∝ |g_im| (Hall/imaginary conductance).  "
        "🟢 Source  🔴 Sink"
    )

    st.subheader("Time-selected snapshot")
    t_step = float(t[1] - t[0]) if K > 1 else 0.1
    t_select = _slider_with_input(
        "Select time",
        key="network_t_select",
        min_value=float(t[0]),
        max_value=float(t[-1]),
        step=t_step,
        cast=float,
        display_format="%.2f",
        default=float(t[K // 2]),
    )
    k_sel = int(np.argmin(np.abs(t - t_select)))
    c1, c2 = st.columns([3, 2])
    with c1:
        fig_sel = _plot_lattice(lattice, out["g"][k_sel],
                                title=f"Network at t = {t[k_sel]:.2f}")
        st.pyplot(fig_sel, width="stretch")
    with c2:
        g_snap = out["g"][k_sel]
        st.markdown(f"**t = {t[k_sel]:.2f}**")
        st.markdown(f"- Mean |g_re|: `{float(np.abs(np.real(g_snap)).mean()):.4f}`")
        st.markdown(f"- Mean |g_im|: `{float(np.abs(np.imag(g_snap)).mean()):.4f}`")
        st.markdown(f"- Entropy S: `{out['S'][k_sel]:.4f}`")
        st.markdown(f"- π_a / π: `{out['pi_a'][k_sel]/math.pi:.4f}`")

with tab_plaq:
    st.subheader("Plaquette Signatures (π_f loop flux)")

    dmg_k = max(0, min(K - 1, int(damage_time_sess / (t[-1] / max(K - 1, 1)))))
    pre_k = max(0, dmg_k - 1)
    post_k = min(K - 1, dmg_k + int(0.3 * K))

    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        fig_p = _plot_plaquettes(lattice, out["g"][pre_k], out["pi_a"][pre_k],
                                  f"Pre-damage (t={t[pre_k]:.1f})")
        st.pyplot(fig_p, width="stretch")
    with cp2:
        fig_p = _plot_plaquettes(lattice, out["g"][min(K - 1, dmg_k + 2)],
                                  out["pi_a"][min(K - 1, dmg_k + 2)],
                                  f"Just after damage (t={t[min(K-1, dmg_k+2)]:.1f})")
        st.pyplot(fig_p, width="stretch")
    with cp3:
        fig_p = _plot_plaquettes(lattice, out["g"][post_k], out["pi_a"][post_k],
                                  f"Post-settle (t={t[post_k]:.1f})")
        st.pyplot(fig_p, width="stretch")

    st.caption(
        "Each cell is a plaquette of 4 bonds. Brightness = π_f loop signature "
        "(phase winding × π_a). Topological order = bright and uniform."
    )

with tab_compare:
    st.subheader("Compare Shipped Presets")
    st.caption(
        "Runs `Default` and `Balanced Recovery` with the same lattice, timing, damage, and seed so the difference is visible rather than implied."
    )

    compare_btn = st.button("Run preset comparison", key="run_preset_comparison", width="stretch")
    auto_run_compare = bool(st.session_state.pop("autorun_compare", False))
    if compare_btn or auto_run_compare:
        with st.spinner("Running preset comparison…"):
            st.session_state["compare_results"] = _run_preset_comparison(
                nx=int(nx),
                ny=int(ny),
                T=float(T),
                dt=float(dt),
                seed=int(seed),
                apply_damage=bool(apply_damage),
                damage_time=float(damage_time),
                damage_scenario=str(damage_scenario),
            )

    if "compare_results" not in st.session_state:
        st.info("Run the comparison to generate side-by-side evidence for the shipped presets.")
    else:
        import pandas as pd

        compare_results = st.session_state["compare_results"]
        shared_meta = compare_results["shared"]
        st.caption(
            f"Last comparison: nx={shared_meta['nx']}, ny={shared_meta['ny']}, T={shared_meta['T']:.1f}, dt={shared_meta['dt']:.2f}, seed={shared_meta['seed']}, scenario={DAMAGE_SCENARIO_LABELS[shared_meta['damage_scenario']]}"
        )

        metric_rows = [
            ("Transfer recovery", "transfer_recovery"),
            ("Boundary signature recovery", "sig_boundary_recovery"),
            ("Top-edge signature recovery", "sig_top_edge_recovery"),
            ("Bott drift", "bott_delta"),
            ("Post spectral gap", "qwz_gap_post"),
            ("GMRES fails", "gmres_fails"),
        ]
        default_summary = compare_results["results"]["Default"]["summary"]
        balanced_summary = compare_results["results"]["Balanced Recovery"]["summary"]
        df_compare = pd.DataFrame(
            [
                {
                    "Metric": label,
                    "Default": float(default_summary[key]),
                    "Balanced Recovery": float(balanced_summary[key]),
                    "Delta": float(balanced_summary[key] - default_summary[key]),
                }
                for label, key in metric_rows
            ]
        )
        st.dataframe(df_compare.round(6), width="stretch", hide_index=True)

        fig_compare = _plot_compare_dashboard(compare_results)
        st.pyplot(fig_compare, width="stretch")

with tab_robust:
    st.subheader("Robustness Sweep")
    st.caption("Runs the current sidebar configuration over a seed range and reports how much the recovery metrics move.")

    robust_cols = st.columns([1, 1, 2])
    with robust_cols[0]:
        seed_count = _slider_with_input(
            "Seeds",
            key="robust_seed_count",
            min_value=1,
            max_value=50,
            step=1,
            cast=int,
            default=10,
        )
    with robust_cols[1]:
        seed_start = st.number_input(
            "Start seed",
            min_value=0,
            max_value=9999,
            value=int(seed),
            step=1,
            key="robust_seed_start",
        )
    with robust_cols[2]:
        run_sweep_btn = st.button("Run robustness sweep", key="run_robustness_sweep", width="stretch")

    auto_run_sweep = bool(st.session_state.pop("autorun_sweep", False))
    if run_sweep_btn or auto_run_sweep:
        with st.spinner("Running robustness sweep…"):
            st.session_state["robustness_rows"] = _run_robustness_sweep(
                nx=int(nx),
                ny=int(ny),
                T=float(T),
                dt=float(dt),
                seed_start=int(seed_start),
                seed_count=int(seed_count),
                apply_damage=bool(apply_damage),
                damage_time=float(damage_time),
                damage_scenario=str(damage_scenario),
                mass=float(mass),
                alpha0=float(alpha0),
                qzw_pi_gain=float(qzw_pi_gain),
                qzw_entropy_gain=float(qzw_entropy_gain),
                init_phase_noise=float(init_phase_noise),
                S_init=float(S_init),
                gamma=float(gamma),
                alpha_pi=float(alpha_pi),
            )
            st.session_state["robustness_meta"] = {
                "seed_start": int(seed_start),
                "seed_count": int(seed_count),
                "damage_scenario": str(damage_scenario),
                "mass": float(mass),
                "alpha0": float(alpha0),
                "alpha_pi": float(alpha_pi),
            }

    if "robustness_rows" not in st.session_state:
        st.info("Run a sweep to see how stable the current configuration is across seeds.")
    else:
        import pandas as pd

        df_robust = pd.DataFrame(st.session_state["robustness_rows"])
        robust_meta = st.session_state.get("robustness_meta", {})
        start_seed = int(robust_meta.get("seed_start", 0))
        count_seed = int(robust_meta.get("seed_count", 0))
        st.caption(
            f"Last sweep: seeds {start_seed}–{start_seed + max(count_seed - 1, 0)}, scenario={DAMAGE_SCENARIO_LABELS[robust_meta.get('damage_scenario', 'central_strip')]}, mass={robust_meta.get('mass', 0.0):.2f}, alpha0={robust_meta.get('alpha0', 0.0):.2f}, alpha_pi={robust_meta.get('alpha_pi', 0.0):.2f}"
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Median Y_eff rec.", f"{df_robust['transfer_recovery'].median():.4f}")
        r2.metric("Worst Y_eff rec.", f"{df_robust['transfer_recovery'].min():.4f}")
        r3.metric("Max |Bott Δ|", f"{df_robust['bott_delta'].abs().max():.4f}")
        r4.metric("Total GMRES fails", f"{int(df_robust['gmres_fails'].sum())}")

        fig_robust = _plot_robustness_dashboard(df_robust)
        st.pyplot(fig_robust, width="stretch")
        st.dataframe(df_robust.round(6), width="stretch", hide_index=True)

with tab_data:
    st.subheader("Raw simulation data")
    import pandas as pd

    stride_d = max(1, K // 200)
    df_raw = pd.DataFrame({
        "t": t[::stride_d],
        "S": out["S"][::stride_d],
        "pi_a": out["pi_a"][::stride_d],
        "mean_g_re": np.abs(np.real(out["g"][::stride_d])).mean(axis=1),
        "mean_g_im": np.abs(np.imag(out["g"][::stride_d])).mean(axis=1),
        "bott": np.interp(t[::stride_d], t[::topo_data["stride"]], topo_data["qwz"]["bott"]),
        "chern_proxy": topo_data["chern"][::stride_d],
        "sig_bnd": topo_data["sig_bnd"][::stride_d],
        "sig_top": topo_data["sig_top"][::stride_d],
    })
    st.dataframe(df_raw.round(6), width="stretch", hide_index=True)

    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="hafc_egatl_results.csv",
        mime="text/csv",
    )

    run_report = _build_run_report(
        lattice,
        out,
        run_params,
        topo_data,
        summ,
        compare_results=st.session_state.get("compare_results"),
        robustness_rows=st.session_state.get("robustness_rows"),
        robustness_meta=st.session_state.get("robustness_meta"),
    )
    report_json = json.dumps(run_report, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download Run Report (JSON)",
        data=report_json,
        file_name="hafc_egatl_run_report.json",
        mime="application/json",
    )

    with st.expander("Run report preview"):
        st.code(report_json.decode("utf-8"), language="json")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HAFC-EGATL — Hybrid Adaptive Field Computer on an Entropy-Gated Adaptive "
    "Topological Lattice.  Topological invariants: Bott index (real-space, half-filling), "
    "Chern proxy (plaquette Berry phase), π_f loop signatures."
)
