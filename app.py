"""HAFC-EGATL Interactive Web App.

An interactive Streamlit interface for the Hybrid Adaptive Field Computer
built on an Entropy-Gated Adaptive Topological Lattice (EGATL).

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import math
import os
import sys
import time
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

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ HAFC-EGATL Controls")
    st.caption("Hybrid Adaptive Field Computer  |  Entropy-Gated Topological Lattice")

    st.subheader("Lattice & Time")
    nx = st.slider("Lattice width (nx)", 3, 10, 6)
    ny = st.slider("Lattice height (ny)", 3, 10, 6)
    T = st.slider("Simulation time T", 5.0, 60.0, 30.0, step=1.0)
    dt = st.select_slider("Time step dt", options=[0.02, 0.05, 0.1, 0.2], value=0.1)
    seed = st.number_input("Random seed", 0, 9999, 0, step=1)

    st.subheader("Damage & Recovery")
    apply_damage = st.checkbox("Apply damage event", value=True)
    damage_time = st.slider(
        "Damage time t_d",
        0.5,
        float(T) - 0.5,
        min(12.0, float(T) * 0.4),
        step=0.5,
        disabled=not apply_damage,
    )
    if not apply_damage:
        damage_time = T + dt  # effectively never

    st.subheader("Physics Parameters")
    mass = st.slider("QWZ mass M", -3.0, 1.0, -1.0, step=0.1)
    alpha0 = st.slider("ARP gain α₀", 0.5, 6.0, 2.5, step=0.1)
    qzw_pi_gain = st.slider("QWZ π gain", 0.0, 1.0, 0.45, step=0.05)
    qzw_entropy_gain = st.slider("QWZ entropy gain", 0.0, 0.5, 0.10, step=0.01)
    init_phase_noise = st.slider("Init phase noise", 0.0, 0.2, 0.02, step=0.005)

    st.subheader("Entropy & Ruler")
    S_init = st.slider("Initial entropy S₀", 0.5, 4.0, 1.5, step=0.1)
    gamma = st.slider("Entropy relaxation γ", 0.01, 1.0, 0.20, step=0.01)
    alpha_pi = st.slider("Ruler drive α_π", 0.01, 0.5, 0.15, step=0.01)

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 HAFC-EGATL Interactive Computer")
st.markdown(
    "**Hybrid Adaptive Field Computer** on an *Entropy-Gated Adaptive Topological Lattice*. "
    "Configure the parameters in the sidebar and click **▶ Run Simulation**."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run and cache simulation
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_sim(
    nx: int, ny: int, T: float, dt: float, seed: int,
    damage_time: float, mass: float,
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
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    g_re = np.abs(np.real(g_snapshot))
    g_im = np.abs(np.imag(g_snapshot))
    g_re_max = max(float(g_re.max()), 1e-6)
    g_im_max = max(float(g_im.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title, fontsize=10)
    cmap = cm.get_cmap("plasma")

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


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running HAFC-EGATL simulation…"):
        t0 = time.time()
        lattice, out = _run_sim(
            nx=nx, ny=ny, T=T, dt=dt, seed=int(seed),
            damage_time=damage_time, mass=mass,
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
    st.session_state["damage_time"] = damage_time
    st.session_state["has_damage"] = apply_damage
    st.session_state["mass"] = mass
    st.session_state["qzw_pi_gain"] = qzw_pi_gain
    st.session_state["qzw_entropy_gain"] = qzw_entropy_gain

if "out" not in st.session_state:
    st.info("👈 Configure parameters and click **▶ Run Simulation** to get started.")
    st.stop()

# ── Retrieve session data ────────────────────────────────────────────────────
lattice: QWZLattice = st.session_state["lattice"]
out: Dict[str, Any] = st.session_state["out"]
damage_time_sess: float = st.session_state["damage_time"]
has_damage: bool = st.session_state["has_damage"]
t = out["t"]
K = len(t)

# ── Compute topology (cached) ────────────────────────────────────────────────
with st.spinner("Computing topological invariants…"):
    topo_data = _compute_topo(
        g_bytes=out["g"].tobytes(),
        nx=lattice.nx, ny=lattice.ny,
        pi_a_bytes=out["pi_a"].tobytes(),
        S_bytes=out["S"].tobytes(),
        mass=st.session_state["mass"],
        qzw_pi_gain=st.session_state["qzw_pi_gain"],
        qzw_entropy_gain=st.session_state["qzw_entropy_gain"],
        n_bonds=lattice.n_bonds, K=K,
    )

# ── Summary metrics row ──────────────────────────────────────────────────────
summ = summarize_recovery(out, lattice, damage_time_sess)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Y_eff recovery", f"{summ['transfer_recovery']:.3f}",
            delta=f"{summ['transfer_post']-summ['transfer_pre']:.3f}")
col2.metric("Sig boundary rec.", f"{summ['sig_boundary_recovery']:.3f}")
col3.metric("Sig top-edge rec.", f"{summ['sig_top_edge_recovery']:.3f}")
col4.metric("Bott Δ (post–pre)", f"{summ['bott_delta']:+.2f}")
col5.metric("GMRES fails", f"{int(summ['gmres_fails'])}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dyn, tab_topo, tab_net, tab_plaq, tab_data = st.tabs([
    "📈 Dynamics", "⚛️ Topology", "🔗 Network", "⬛ Plaquettes", "📋 Data",
])

with tab_dyn:
    st.subheader("System Dynamics")
    fig_dyn = _plot_dynamics(t, out, lattice, damage_time_sess, has_damage)
    st.pyplot(fig_dyn, use_container_width=True)

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
            ("QWZ Chern", summ["qzw_chern_pre"], summ["qzw_chern_post"], None),
            ("Bott index", summ["bott_pre"], summ["bott_post"], None),
            ("Spectral gap", summ["qzw_gap_pre"], summ["qzw_gap_post"], None),
            ("u_eff", summ["qzw_u_eff_pre"], summ["qzw_u_eff_post"], None),
            ("Entropy S", None, summ["final_S"], None),
            ("π_a / π", None, summ["final_pi_a"] / math.pi, None),
        ]
        df = pd.DataFrame(
            [(r, f"{v:.4f}" if v is not None else "–", f"{w:.4f}" if w is not None else "–",
              f"{x:.4f}" if x is not None else "–")
             for r, v, w, x in rows],
            columns=["Metric", "Pre", "Post", "Recovery"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_topo:
    st.subheader("Topological Readout")
    fig_topo = _plot_topology(t, topo_data, damage_time_sess, has_damage)
    st.pyplot(fig_topo, use_container_width=True)

    with st.expander("Topological summary"):
        st.markdown(f"""
| Invariant | Pre-damage | Post-settle | Δ |
|-----------|-----------|------------|---|
| **Bott index** | {summ['bott_pre']:.3f} | {summ['bott_post']:.3f} | {summ['bott_delta']:+.3f} |
| **QWZ Chern** | {summ['qzw_chern_pre']:.3f} | {summ['qzw_chern_post']:.3f} | {summ['qzw_chern_delta']:+.3f} |
| **Chern proxy** | {summ['chern_pre']:.3f} | {summ['chern_post']:.3f} | {summ['chern_delta']:+.3f} |
| **Spectral gap** | {summ['qzw_gap_pre']:.4f} | {summ['qzw_gap_post']:.4f} | — |
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
        snap_cols[ci].pyplot(fig_n, use_container_width=True)

    st.caption(
        "Bond width ∝ |g_re| (real conductance).  "
        "Bond colour (dark→bright) ∝ |g_im| (Hall/imaginary conductance).  "
        "🟢 Source  🔴 Sink"
    )

    st.subheader("Time-selected snapshot")
    t_select = st.slider(
        "Select time", float(t[0]), float(t[-1]), float(t[K // 2]), step=float(t[1] - t[0])
    )
    k_sel = int(np.argmin(np.abs(t - t_select)))
    c1, c2 = st.columns([3, 2])
    with c1:
        fig_sel = _plot_lattice(lattice, out["g"][k_sel],
                                title=f"Network at t = {t[k_sel]:.2f}")
        st.pyplot(fig_sel, use_container_width=True)
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
        st.pyplot(fig_p, use_container_width=True)
    with cp2:
        fig_p = _plot_plaquettes(lattice, out["g"][min(K - 1, dmg_k + 2)],
                                  out["pi_a"][min(K - 1, dmg_k + 2)],
                                  f"Just after damage (t={t[min(K-1, dmg_k+2)]:.1f})")
        st.pyplot(fig_p, use_container_width=True)
    with cp3:
        fig_p = _plot_plaquettes(lattice, out["g"][post_k], out["pi_a"][post_k],
                                  f"Post-settle (t={t[post_k]:.1f})")
        st.pyplot(fig_p, use_container_width=True)

    st.caption(
        "Each cell is a plaquette of 4 bonds. Brightness = π_f loop signature "
        "(phase winding × π_a). Topological order = bright and uniform."
    )

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
    st.dataframe(df_raw.round(6), use_container_width=True, hide_index=True)

    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="hafc_egatl_results.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HAFC-EGATL — Hybrid Adaptive Field Computer on an Entropy-Gated Adaptive "
    "Topological Lattice.  Topological invariants: Bott index (real-space, half-filling), "
    "Chern proxy (plaquette Berry phase), π_f loop signatures."
)
