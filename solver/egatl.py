"""Entropy-Gated Adaptive Topological Lattice (EGATL) solver.

Implements the full HAFC simulation engine:
  - QWZ admittance lattice construction
  - EGATL dynamics (entropy-gated ARP, adaptive pi ruler)
  - Complex-admittance network solver (GMRES / direct)
  - Topological readout: Bott index, Chern proxy, spectral gap
  - Loop / plaquette / edge signatures
  - Damage-recovery protocol and ablation comparison
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres as sp_gmres, spilu, spsolve

# ============================================================ #
#  Parameter dataclasses                                        #
# ============================================================ #

@dataclass
class EGATLParams:
    """Entropy-Gated Adaptive Topological Lattice hyper-parameters."""
    alpha0: float = 2.5        # base ARP reinforcement gain
    S_c: float = 0.3           # entropy threshold (sigmoid centre)
    dS: float = 0.5            # entropy sigmoid width
    mu0: float = 0.25          # base conductance decay rate
    S0: float = 2.0            # entropy normalisation reference
    lambda_s: float = 0.05     # imaginary-part update rate
    g_min: float = 0.01        # minimum real conductance
    g_max: float = 5.0         # maximum real conductance
    g_imag_max: float = 3.0    # maximum |imaginary| conductance
    budget_re: float = 120.0   # global real-conductance budget
    init_phase_noise: float = 0.02  # initial phase noise on g_im
    qzw_pi_gain: float = 0.45  # QWZ pi_a → u_eff gain
    qzw_entropy_gain: float = 0.10  # QWZ S → u_eff gain


@dataclass
class EntropyParams:
    """Entropy-state dynamics parameters."""
    S_init: float = 1.5    # initial entropy
    S_eq: float = 1.0      # equilibrium entropy
    gamma: float = 0.20    # relaxation rate toward S_eq
    kappa_slip: float = 0.10  # activity-slip coupling to entropy
    Tij: float = 1.0       # bond current threshold


@dataclass
class RulerParams:
    """Adaptive pi ruler (pi_a) parameters."""
    pi0: float = math.pi      # reference / rest value
    pi_init: float = math.pi  # initial pi_a
    alpha_pi: float = 0.15    # entropy → pi_a drive
    mu_pi: float = 0.15       # pi_a decay toward pi0
    pi_min: float = 0.5       # hard lower bound
    pi_max: float = 12.0      # hard upper bound


# ============================================================ #
#  Lattice structure                                            #
# ============================================================ #

@dataclass
class QWZLattice:
    nx: int
    ny: int
    n_nodes: int
    n_bonds: int
    bonds: List[Tuple[int, int]]        # (node_i, node_j) for each bond
    bond_dir: np.ndarray                # 0=horizontal, 1=vertical
    plaquettes: List[List[int]]         # list of 4-bond-index quads
    source_cell: int
    sink_cell: int
    boundary_bond_mask: np.ndarray      # bool shape (n_bonds,)
    top_edge_bond_mask: np.ndarray      # bool shape (n_bonds,)
    node_x: np.ndarray                  # x coord of each node
    node_y: np.ndarray                  # y coord of each node
    bond_src: np.ndarray                # source node index for each bond
    bond_dst: np.ndarray                # destination node index for each bond
    incidence: csr_matrix               # node-edge incidence matrix, shape (n_nodes, n_bonds)


def build_qwz_lattice(nx: int, ny: int, mass: float = -1.0) -> "QWZLattice":
    """Build a 2-D QWZ admittance lattice with nearest-neighbour bonds."""
    n_nodes = nx * ny

    def idx(x: int, y: int) -> int:
        return x + y * nx

    bonds: List[Tuple[int, int]] = []
    bond_dir_list: List[int] = []

    # Horizontal bonds (x-direction, dir=0)
    for y in range(ny):
        for x in range(nx - 1):
            bonds.append((idx(x, y), idx(x + 1, y)))
            bond_dir_list.append(0)

    # Vertical bonds (y-direction, dir=1)
    for y in range(ny - 1):
        for x in range(nx):
            bonds.append((idx(x, y), idx(x, y + 1)))
            bond_dir_list.append(1)

    n_bonds = len(bonds)
    bond_dir = np.array(bond_dir_list, dtype=int)

    # Build plaquette index maps
    bond_idx_h: Dict[Tuple[int, int], int] = {}
    bond_idx_v: Dict[Tuple[int, int], int] = {}
    bi = 0
    for y in range(ny):
        for x in range(nx - 1):
            bond_idx_h[(x, y)] = bi
            bi += 1
    for y in range(ny - 1):
        for x in range(nx):
            bond_idx_v[(x, y)] = bi
            bi += 1

    # Plaquettes: (x,y)...(x+1,y+1) square, listed CCW: bottom, right, top, left
    plaquettes: List[List[int]] = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            plaquettes.append([
                bond_idx_h[(x, y)],      # bottom
                bond_idx_v[(x + 1, y)],  # right
                bond_idx_h[(x, y + 1)],  # top
                bond_idx_v[(x, y)],      # left
            ])

    source_cell = idx(0, 0)
    sink_cell = idx(nx - 1, ny - 1)

    boundary_bond_mask = np.zeros(n_bonds, dtype=bool)
    top_edge_bond_mask = np.zeros(n_bonds, dtype=bool)

    bi = 0
    for y in range(ny):
        for x in range(nx - 1):
            if y == 0 or y == ny - 1:
                boundary_bond_mask[bi] = True
            if y == ny - 1:
                top_edge_bond_mask[bi] = True
            bi += 1
    for y in range(ny - 1):
        for x in range(nx):
            if x == 0 or x == nx - 1:
                boundary_bond_mask[bi] = True
            bi += 1

    node_x = np.array([idx % nx for idx in range(n_nodes)], dtype=float)
    node_y = np.array([idx // nx for idx in range(n_nodes)], dtype=float)
    bond_src = np.array([i for i, _ in bonds], dtype=int)
    bond_dst = np.array([j for _, j in bonds], dtype=int)

    incidence_rows = np.concatenate([bond_src, bond_dst])
    incidence_cols = np.concatenate([np.arange(n_bonds), np.arange(n_bonds)])
    incidence_data = np.concatenate([
        np.ones(n_bonds, dtype=complex),
        -np.ones(n_bonds, dtype=complex),
    ])
    incidence = csr_matrix(
        (incidence_data, (incidence_rows, incidence_cols)),
        shape=(n_nodes, n_bonds),
    )

    return QWZLattice(
        nx=nx, ny=ny, n_nodes=n_nodes, n_bonds=n_bonds,
        bonds=bonds, bond_dir=bond_dir,
        plaquettes=plaquettes,
        source_cell=source_cell, sink_cell=sink_cell,
        boundary_bond_mask=boundary_bond_mask,
        top_edge_bond_mask=top_edge_bond_mask,
        node_x=node_x, node_y=node_y,
        bond_src=bond_src, bond_dst=bond_dst,
        incidence=incidence,
    )


# ============================================================ #
#  Helper maths                                                 #
# ============================================================ #

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _alpha_eff(S: float, eg: EGATLParams) -> float:
    return eg.alpha0 * _sigmoid((S - eg.S_c) / eg.dS)


def _mu_eff(S: float, eg: EGATLParams) -> float:
    return eg.mu0 * (S / eg.S0)


def _project_real_budget(values: np.ndarray, eg: EGATLParams) -> np.ndarray:
    """Project real conductances onto box constraints plus total budget."""
    clipped = np.clip(values, eg.g_min, eg.g_max)
    if eg.budget_re <= 0 or float(clipped.sum()) <= eg.budget_re:
        return clipped

    min_feasible = eg.g_min * clipped.size
    if eg.budget_re <= min_feasible:
        return np.full_like(clipped, eg.g_min)

    lo = float(np.min(values - eg.g_max))
    hi = float(np.max(values - eg.g_min))
    projected = clipped
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        projected = np.clip(values - mid, eg.g_min, eg.g_max)
        if float(projected.sum()) > eg.budget_re:
            lo = mid
        else:
            hi = mid
    return np.clip(values - hi, eg.g_min, eg.g_max)


# ============================================================ #
#  Network solver                                               #
# ============================================================ #

def _build_laplacian(g: np.ndarray, lattice: QWZLattice) -> np.ndarray:
    g_diag = diags(g, offsets=0, shape=(lattice.n_bonds, lattice.n_bonds), format="csr")
    return (lattice.incidence @ g_diag @ lattice.incidence.conjugate().transpose()).tocsr()


def _make_preconditioner(matrix: csr_matrix) -> Optional[LinearOperator]:
    try:
        ilu = spilu(matrix.tocsc(), drop_tol=1e-4, fill_factor=10.0)
        return LinearOperator(matrix.shape, matvec=ilu.solve, dtype=matrix.dtype)
    except Exception:
        return None


def _grounded_solve(
    matrix: csr_matrix,
    rhs: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Solve the grounded sparse system with direct solve first, then GMRES fallback."""
    dim = matrix.shape[0]
    if dim <= 128:
        try:
            phi_red = np.asarray(spsolve(matrix, rhs), dtype=complex)
            if np.all(np.isfinite(phi_red)):
                return phi_red, 0
        except Exception:
            pass

    gmres_failed = 0
    preconditioner = _make_preconditioner(matrix)
    try:
        phi_red, info = sp_gmres(
            matrix,
            rhs,
            x0=x0,
            M=preconditioner,
            rtol=1e-8,
            atol=1e-10,
            restart=min(80, max(20, dim)),
            maxiter=1000,
        )
        phi_red = np.asarray(phi_red, dtype=complex)
        if info == 0 and np.all(np.isfinite(phi_red)):
            return phi_red, 0
        gmres_failed = 1
    except Exception:
        gmres_failed = 1

    try:
        phi_red = np.asarray(spsolve(matrix, rhs), dtype=complex)
        if np.all(np.isfinite(phi_red)):
            return phi_red, gmres_failed
    except Exception:
        pass

    return np.zeros(dim, dtype=complex), 1


def _solve_network(
    g: np.ndarray,
    lattice: QWZLattice,
    prev_phi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve complex admittance network.

    Returns
    -------
    phi     : (n_nodes,) complex voltage array
    I_bond  : (n_bonds,) complex current array
    I_abs   : (n_bonds,) real array, raw current magnitude
    I_norm  : (n_bonds,) real array, |I| normalised to [0,1]
    gmres_fail : 0 or 1
    """
    N = lattice.n_nodes
    sink = lattice.sink_cell
    source = lattice.source_cell

    L = _build_laplacian(g, lattice)

    b = np.zeros(N, dtype=complex)
    b[source] += 1.0
    b[sink] -= 1.0

    # Ground at sink: remove sink row/col
    keep = np.flatnonzero(np.arange(N) != sink)
    L_red = L[keep][:, keep].tocsr()
    b_red = b[keep]
    x0_red = None
    if prev_phi is not None and prev_phi.shape == (N,):
        x0_red = np.asarray(prev_phi[keep], dtype=complex)

    phi_red, gmres_fail = _grounded_solve(L_red, b_red, x0=x0_red)

    phi = np.zeros(N, dtype=complex)
    phi[keep] = phi_red

    # Bond currents
    edge_drop = np.asarray(lattice.incidence.conjugate().transpose() @ phi).reshape(-1)
    I_bond = np.asarray(g * edge_drop, dtype=complex)

    I_abs = np.abs(I_bond)
    peak = float(I_abs.max()) if I_abs.size > 0 else 1.0
    I_norm = I_abs / max(peak, 1e-12)

    return phi, I_bond, I_abs, I_norm, gmres_fail


# ============================================================ #
#  EGATL one-step update                                       #
# ============================================================ #

def _egatl_step(
    g: np.ndarray,
    I_bond: np.ndarray,
    S: float,
    pi_a: float,
    dt: float,
    eg: EGATLParams,
    ent: EntropyParams,
    ruler: RulerParams,
) -> Tuple[np.ndarray, float, float]:
    """Advance conductances, entropy, and pi_a by one time step dt."""
    I_abs = np.abs(I_bond)
    g_re = np.real(g)
    g_im = np.imag(g)

    # Entropy-gated ARP for real conductance
    alpha = _alpha_eff(S, eg)
    mu = _mu_eff(S, eg)

    if mu > 1e-12:
        decay = math.exp(-mu * dt)
        g_re_tentative = g_re * decay + (alpha * I_abs / mu) * (1.0 - decay)
    else:
        g_re_tentative = g_re + dt * alpha * I_abs
    g_re_new = _project_real_budget(g_re_tentative, eg)

    # Imaginary part: pi_a-gated Hall conductance
    phase_g = np.angle(g)
    target_im = I_abs * np.sin(phase_g + pi_a * 0.5)
    dg_im = eg.lambda_s * (target_im - g_im)
    g_im_new = np.clip(g_im + dt * dg_im, -eg.g_imag_max, eg.g_imag_max)

    g_new = g_re_new + 1j * g_im_new

    # Entropy dynamics
    activity = float(np.mean(I_abs))
    # slip = fraction of bonds below threshold
    slip = float(np.mean(np.maximum(0.0, ent.Tij - I_abs)))
    entropy_drive = ent.kappa_slip * slip * activity
    if ent.gamma > 1e-12:
        decay_S = math.exp(-ent.gamma * dt)
        S_target = ent.S_eq + entropy_drive / ent.gamma
        S_new = decay_S * S + (1.0 - decay_S) * S_target
    else:
        S_new = S + dt * entropy_drive
    S_new = max(1e-3, S_new)

    # Adaptive pi ruler
    if ruler.mu_pi > 1e-12:
        decay_pi = math.exp(-ruler.mu_pi * dt)
        pi_dev = pi_a - ruler.pi0
        pi_dev_new = decay_pi * pi_dev + (ruler.alpha_pi * S / ruler.mu_pi) * (1.0 - decay_pi)
        pi_candidate = ruler.pi0 + pi_dev_new
    else:
        pi_candidate = pi_a + dt * ruler.alpha_pi * S
    pi_a_new = float(np.clip(pi_candidate, ruler.pi_min, ruler.pi_max))

    return g_new, S_new, pi_a_new


# ============================================================ #
#  Recovery protocol                                            #
# ============================================================ #

def _damage_mask(
    lattice: "QWZLattice",
    damage_scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a boolean mask for the selected damage scenario."""
    scenario = damage_scenario or "central_strip"
    mask = np.zeros(lattice.n_bonds, dtype=bool)

    center_x = 0.5 * float(lattice.nx - 1)
    center_y = 0.5 * float(lattice.ny - 1)

    for edge_index, (node_i, node_j) in enumerate(lattice.bonds):
        xi, yi = float(lattice.node_x[node_i]), float(lattice.node_y[node_i])
        xj, yj = float(lattice.node_x[node_j]), float(lattice.node_y[node_j])
        x_mid = 0.5 * (xi + xj)
        y_mid = 0.5 * (yi + yj)

        if scenario == "central_strip":
            if abs(x_mid - center_x) <= 0.75:
                mask[edge_index] = True
        elif scenario == "center_cross":
            if abs(x_mid - center_x) <= 0.75 or abs(y_mid - center_y) <= 0.75:
                mask[edge_index] = True
        elif scenario == "center_block":
            if abs(x_mid - center_x) <= 1.0 and abs(y_mid - center_y) <= 1.0:
                mask[edge_index] = True
        elif scenario == "top_edge":
            if yi == lattice.ny - 1 or yj == lattice.ny - 1:
                mask[edge_index] = True
        elif scenario == "source_corner":
            if max(xi, xj) <= 1.0 and max(yi, yj) <= 1.0:
                mask[edge_index] = True
        elif scenario == "random_bonds":
            continue
        else:
            raise ValueError(f"Unknown damage scenario: {damage_scenario}")

    if scenario == "random_bonds":
        n_drop = max(1, int(round(0.30 * lattice.n_bonds)))
        dropped = rng.choice(lattice.n_bonds, size=n_drop, replace=False)
        mask[dropped] = True

    if not np.any(mask):
        mask[:] = True

    return mask


def _apply_damage_event(
    g: np.ndarray,
    lattice: "QWZLattice",
    damage_scenario: str,
    damage_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the selected damage scenario by scaling targeted bonds."""
    damaged = np.array(g, copy=True)
    mask = _damage_mask(lattice, damage_scenario, rng)
    damaged[mask] *= damage_factor
    return damaged

def run_recovery_protocol(
    nx: int,
    ny: int,
    T: float,
    dt: float,
    seed: int,
    damage_time: float,
    mass: float,
    damage_factor: float,
    damage_scenario: str,
    eg: EGATLParams,
    ent: EntropyParams,
    ruler: RulerParams,
) -> Tuple["QWZLattice", Dict[str, Any]]:
    """Run the full HAFC damage-recovery protocol.

    Returns
    -------
    lattice : QWZLattice
    out     : dict with arrays t, g, phi, I_abs, I_norm, pi_a, S, gmres_fails
    """
    rng = np.random.default_rng(seed)
    lattice = build_qwz_lattice(nx, ny, mass)

    base_steps = int(math.floor(T / dt))
    t_arr = np.arange(base_steps + 1, dtype=float) * dt
    if t_arr[-1] < T - 1e-12:
        t_arr = np.concatenate([t_arr, np.array([float(T)], dtype=float)])
    K = len(t_arr)

    # Initial conductances: real=1, small imaginary noise
    g = np.ones(lattice.n_bonds, dtype=complex)
    g = g + 1j * eg.init_phase_noise * rng.standard_normal(lattice.n_bonds)
    g = np.clip(np.real(g), eg.g_min, eg.g_max) + 1j * np.imag(g)

    S = ent.S_init
    pi_a = ruler.pi_init
    gmres_fails = 0

    g_hist = np.zeros((K, lattice.n_bonds), dtype=complex)
    phi_hist = np.zeros((K, lattice.n_nodes), dtype=complex)
    I_abs_hist = np.zeros((K, lattice.n_bonds), dtype=float)
    I_norm_hist = np.zeros((K, lattice.n_bonds), dtype=float)
    pi_a_hist = np.zeros(K, dtype=float)
    S_hist = np.zeros(K, dtype=float)

    damage_applied = False
    prev_phi: Optional[np.ndarray] = None

    for k, t_now in enumerate(t_arr):
        # Apply damage
        if not damage_applied and t_now >= damage_time:
            g = _apply_damage_event(
                g,
                lattice,
                damage_scenario=damage_scenario,
                damage_factor=damage_factor,
                rng=rng,
            )
            damage_applied = True

        phi, I_bond, I_abs, I_norm, gf = _solve_network(g, lattice, prev_phi=prev_phi)
        gmres_fails += gf
        prev_phi = phi

        g_hist[k] = g
        phi_hist[k] = phi
        I_abs_hist[k] = I_abs
        I_norm_hist[k] = I_norm
        pi_a_hist[k] = pi_a
        S_hist[k] = S

        if k < K - 1:
            dt_step = float(t_arr[k + 1] - t_now)
            g, S, pi_a = _egatl_step(g, I_bond, S, pi_a, dt_step, eg, ent, ruler)

    out: Dict[str, Any] = {
        "t": t_arr,
        "g": g_hist,
        "phi": phi_hist,
        "I_abs": I_abs_hist,
        "I_norm": I_norm_hist,
        "pi_a": pi_a_hist,
        "S": S_hist,
        "gmres_fails": gmres_fails,
    }
    return lattice, out


# ============================================================ #
#  Observables                                                  #
# ============================================================ #

def effective_transfer(phi: np.ndarray, source: int, sink: int) -> float:
    """Effective admittance Y = 1 / |phi_source - phi_sink|."""
    delta = abs(phi[source] - phi[sink])
    return 1.0 / max(delta, 1e-12)


def boundary_current_fraction(I_abs: np.ndarray, lattice: "QWZLattice") -> float:
    """Fraction of raw current magnitude carried by boundary bonds."""
    all_I = float(np.sum(I_abs))
    bnd_I = float(np.sum(I_abs[lattice.boundary_bond_mask]))
    return bnd_I / max(all_I, 1e-12)


def top_edge_fraction(I_abs: np.ndarray, lattice: "QWZLattice") -> float:
    top_I = float(np.sum(I_abs[lattice.top_edge_bond_mask]))
    all_I = float(np.sum(I_abs))
    return top_I / max(all_I, 1e-12)


def proxy_chern_series(g: np.ndarray, lattice: "QWZLattice") -> np.ndarray:
    """Proxy Chern number from plaquette flux (Berry-phase analog)."""
    K = g.shape[0]
    chern = np.zeros(K)
    for k in range(K):
        total = 0.0
        for plaq in lattice.plaquettes:
            g0, g1, g2, g3 = g[k, plaq[0]], g[k, plaq[1]], g[k, plaq[2]], g[k, plaq[3]]
            # Phase of product around plaquette (CCW)
            prod = g0 * g1 * np.conj(g2) * np.conj(g3)
            if abs(prod) > 1e-30:
                total += float(np.angle(prod))
        chern[k] = total / (2 * math.pi)
    return chern


# ============================================================ #
#  Plaquette / loop signatures                                  #
# ============================================================ #

def plaquette_signatures(
    g_snapshot: np.ndarray,
    lattice: "QWZLattice",
    pi_a: float,
) -> np.ndarray:
    """Compute pi_f loop signature for each plaquette.

    Signature S_p = |Im(g_p)| mean, gated by pi_a.
    """
    g = np.asarray(g_snapshot)
    result = np.zeros(len(lattice.plaquettes))
    for pi_idx, plaq_bonds in enumerate(lattice.plaquettes):
        g_plaq = g[plaq_bonds]
        # Phase winding around plaquette
        prod = np.prod(g_plaq[:2]) * np.prod(np.conj(g_plaq[2:]))
        angle = float(np.angle(prod)) if abs(prod) > 1e-30 else 0.0
        sig = abs(angle) / (math.pi + 1e-12) * pi_a
        result[pi_idx] = sig
    return result


def boundary_signature_series(
    g: np.ndarray,
    lattice: "QWZLattice",
    pi_a: np.ndarray,
) -> np.ndarray:
    K = g.shape[0]
    sig = np.zeros(K)
    bnd_idx = np.where(lattice.boundary_bond_mask)[0]
    for k in range(K):
        g_bnd = g[k, bnd_idx]
        sig[k] = float(np.mean(np.abs(np.imag(g_bnd)))) * float(pi_a[k])
    return sig


def top_edge_signature_series(
    g: np.ndarray,
    lattice: "QWZLattice",
    pi_a: np.ndarray,
) -> np.ndarray:
    K = g.shape[0]
    sig = np.zeros(K)
    top_idx = np.where(lattice.top_edge_bond_mask)[0]
    for k in range(K):
        g_top = g[k, top_idx]
        sig[k] = float(np.mean(np.abs(np.imag(g_top)))) * float(pi_a[k])
    return sig


def plaquette_signature_series(
    g: np.ndarray,
    lattice: "QWZLattice",
    pi_a: np.ndarray,
) -> np.ndarray:
    K = g.shape[0]
    sig = np.zeros(K)
    for k in range(K):
        sigs = plaquette_signatures(g[k], lattice, float(pi_a[k]))
        sig[k] = float(np.mean(sigs))
    return sig


# ============================================================ #
#  QWZ topology: Bott index + Chern + gap                       #
# ============================================================ #

def _build_qwz_hamiltonian(
    g: np.ndarray,
    lattice: "QWZLattice",
    u_eff: float,
) -> np.ndarray:
    """Build 2N×2N real-space QWZ Hamiltonian from conductance snapshot.

    Orbital ordering: node r → (2r = spin-up, 2r+1 = spin-down).
    On-site: u_eff σ_z.
    x-bond (i→j): (σ_z − i σ_x)/2 scaled by g_re, with g_im via σ_x.
    y-bond (i→j): (σ_z − i σ_y)/2 scaled by g_re, with g_im via σ_y.
    """
    N = lattice.n_nodes
    H = np.zeros((2 * N, 2 * N), dtype=complex)

    # On-site term: u_eff σ_z
    for i in range(N):
        H[2 * i, 2 * i] += u_eff
        H[2 * i + 1, 2 * i + 1] -= u_eff

    # Bond hopping
    for e, (i, j) in enumerate(lattice.bonds):
        g_re = float(np.real(g[e]))
        g_im = float(np.imag(g[e]))
        d = lattice.bond_dir[e]

        if d == 0:  # x-direction: σ_z/2 hopping, -iσ_x/2 Hall
            t_uu = g_re / 2.0
            t_dd = -g_re / 2.0
            t_ud = 1j * g_im / 2.0
            t_du = 1j * g_im / 2.0
        else:       # y-direction: σ_z/2 hopping, -iσ_y/2 Hall
            t_uu = g_re / 2.0
            t_dd = -g_re / 2.0
            t_ud = g_im / 2.0
            t_du = -g_im / 2.0

        # i→j block
        H[2 * i, 2 * j] += t_uu
        H[2 * i + 1, 2 * j + 1] += t_dd
        H[2 * i, 2 * j + 1] += t_ud
        H[2 * i + 1, 2 * j] += t_du
        # j→i Hermitian conjugate
        H[2 * j, 2 * i] += t_uu.conjugate()
        H[2 * j + 1, 2 * i + 1] += t_dd.conjugate()
        H[2 * j, 2 * i + 1] += t_du.conjugate()
        H[2 * j + 1, 2 * i] += t_ud.conjugate()

    return H


def _bott_index(
    H: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    Lx: float,
    Ly: float,
) -> float:
    """Compute Bott index from 2N×2N Hamiltonian at half-filling."""
    dim = H.shape[0]
    N = dim // 2

    try:
        E, U = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return 0.0

    # Projector onto lower N states (half-filling)
    occ = U[:, :N]
    P = occ @ occ.conj().T

    # Normalised position phases (each node repeated for 2 orbitals)
    px = np.repeat(node_x / max(Lx, 1.0) * 2 * math.pi, 2)
    py = np.repeat(node_y / max(Ly, 1.0) * 2 * math.pi, 2)

    I_dim = np.eye(dim, dtype=complex)
    Ux = np.diag(np.exp(1j * px))
    Uy = np.diag(np.exp(1j * py))

    A = P @ Ux @ P + (I_dim - P)
    B = P @ Uy @ P + (I_dim - P)
    C = A @ B @ A.conj().T @ B.conj().T

    try:
        eigvals = np.linalg.eigvals(C)
        # Sum of imaginary parts of log(eigenvalues)
        log_sum = float(np.sum(np.log(eigvals + 1e-30).imag))
        return float(round(log_sum / (2 * math.pi)))
    except Exception:
        return 0.0


def _spectral_gap(H: np.ndarray) -> float:
    """Spectral gap at half-filling (gap between N-th and (N+1)-th eigenvalue)."""
    N = H.shape[0] // 2
    try:
        E = np.linalg.eigvalsh(H)
        return float(E[N] - E[N - 1])
    except Exception:
        return 0.0


def qwz_topology_series(
    g: np.ndarray,
    lattice: "QWZLattice",
    pi_a_hist: np.ndarray,
    S_hist: np.ndarray,
    mass0: float,
    pi_gain: float,
    entropy_gain: float,
) -> Dict[str, np.ndarray]:
    """Compute Bott, Chern, gap, u_eff for every time step.

    Returns dict with arrays: bott, chern, gap, u_eff (each shape (K,)).
    """
    K = g.shape[0]
    bott = np.zeros(K)
    chern = np.zeros(K)
    gap = np.zeros(K)
    u_eff_arr = np.zeros(K)

    Lx = float(lattice.nx - 1) if lattice.nx > 1 else 1.0
    Ly = float(lattice.ny - 1) if lattice.ny > 1 else 1.0

    for k in range(K):
        u_eff = mass0 + pi_gain * float(pi_a_hist[k]) + entropy_gain * float(S_hist[k])
        u_eff_arr[k] = u_eff

        H = _build_qwz_hamiltonian(g[k], lattice, u_eff)
        bott[k] = _bott_index(H, lattice.node_x, lattice.node_y, Lx, Ly)
        gap[k] = _spectral_gap(H)

        # Chern from plaquette flux
        total = 0.0
        for plaq in lattice.plaquettes:
            g0, g1, g2, g3 = g[k, plaq[0]], g[k, plaq[1]], g[k, plaq[2]], g[k, plaq[3]]
            prod = g0 * g1 * np.conj(g2) * np.conj(g3)
            if abs(prod) > 1e-30:
                total += float(np.angle(prod))
        chern[k] = total / (2 * math.pi)

    return {"bott": bott, "chern": chern, "gap": gap, "u_eff": u_eff_arr}


# ============================================================ #
#  Recovery summary                                             #
# ============================================================ #

def _window_mean(arr: np.ndarray, t: np.ndarray, t_lo: float, t_hi: float) -> float:
    mask = (t >= t_lo) & (t <= t_hi)
    if not np.any(mask):
        return float(np.mean(arr))
    return float(np.mean(arr[mask]))


def summarize_recovery(
    out: Dict[str, Any],
    lattice: "QWZLattice",
    damage_time: float,
    mass0: float = -1.0,
    pi_gain: float = 0.45,
    entropy_gain: float = 0.10,
) -> Dict[str, float]:
    """Summarise pre/post recovery metrics."""
    t = out["t"]
    T_end = float(t[-1])
    dt_wind = max((T_end - damage_time) * 0.1, 1.0)

    pre_lo = max(0.0, damage_time - 3.0)
    pre_hi = damage_time
    post_lo = damage_time + dt_wind
    post_hi = T_end

    K = len(t)

    Yeff = np.array([
        effective_transfer(out["phi"][k], lattice.source_cell, lattice.sink_cell)
        for k in range(K)
    ])
    Bfrac = np.array([
        boundary_current_fraction(out["I_abs"][k], lattice)
        for k in range(K)
    ])
    Tfrac = np.array([
        top_edge_fraction(out["I_abs"][k], lattice)
        for k in range(K)
    ])
    Sig_bnd = boundary_signature_series(out["g"], lattice, out["pi_a"])
    Sig_top = top_edge_signature_series(out["g"], lattice, out["pi_a"])
    Sig_plaq = plaquette_signature_series(out["g"], lattice, out["pi_a"])
    Chern_prx = proxy_chern_series(out["g"], lattice)

    # QWZ series (down-sampled for speed)
    stride = max(1, K // 80)
    g_s = out["g"][::stride]
    pi_s = out["pi_a"][::stride]
    S_s = out["S"][::stride]
    t_s = t[::stride]
    qwz_s = qwz_topology_series(
        g_s,
        lattice,
        pi_s,
        S_s,
        mass0=mass0,
        pi_gain=pi_gain,
        entropy_gain=entropy_gain,
    )

    def pm(arr: np.ndarray) -> float:
        return _window_mean(arr, t, pre_lo, pre_hi)

    def po(arr: np.ndarray) -> float:
        return _window_mean(arr, t, post_lo, post_hi)

    def pm_s(arr: np.ndarray) -> float:
        return _window_mean(arr, t_s, pre_lo, pre_hi)

    def po_s(arr: np.ndarray) -> float:
        return _window_mean(arr, t_s, post_lo, post_hi)

    y_pre = pm(Yeff)
    y_post = po(Yeff)

    sb_pre = pm(Sig_bnd)
    sb_post = po(Sig_bnd)
    st_pre = pm(Sig_top)
    st_post = po(Sig_top)
    sp_pre = pm(Sig_plaq)
    sp_post = po(Sig_plaq)

    chern_pre = pm(Chern_prx)
    chern_post = po(Chern_prx)

    bott_pre = pm_s(qwz_s["bott"])
    bott_post = po_s(qwz_s["bott"])
    qwz_chern_pre = pm_s(qwz_s["chern"])
    qwz_chern_post = po_s(qwz_s["chern"])
    gap_pre = pm_s(qwz_s["gap"])
    gap_post = po_s(qwz_s["gap"])
    u_eff_pre = pm_s(qwz_s["u_eff"])
    u_eff_post = po_s(qwz_s["u_eff"])

    bott_int_err_pre = abs(bott_pre - round(bott_pre))
    bott_int_err_post = abs(bott_post - round(bott_post))

    def _rec(pre: float, post: float) -> float:
        return post / max(abs(pre), 1e-12)

    return {
        "transfer_pre": y_pre,
        "transfer_post": y_post,
        "transfer_recovery": _rec(y_pre, y_post),
        "boundary_pre": pm(Bfrac),
        "boundary_post": po(Bfrac),
        "top_edge_pre": pm(Tfrac),
        "top_edge_post": po(Tfrac),
        "sig_boundary_pre": sb_pre,
        "sig_boundary_post": sb_post,
        "sig_boundary_recovery": _rec(sb_pre, sb_post),
        "sig_top_edge_pre": st_pre,
        "sig_top_edge_post": st_post,
        "sig_top_edge_recovery": _rec(st_pre, st_post),
        "sig_plaquette_pre": sp_pre,
        "sig_plaquette_post": sp_post,
        "sig_plaquette_recovery": _rec(sp_pre, sp_post),
        "chern_pre": chern_pre,
        "chern_post": chern_post,
        "chern_delta": chern_post - chern_pre,
        "qwz_chern_pre": qwz_chern_pre,
        "qwz_chern_post": qwz_chern_post,
        "qwz_chern_delta": qwz_chern_post - qwz_chern_pre,
        "bott_pre": bott_pre,
        "bott_post": bott_post,
        "bott_delta": bott_post - bott_pre,
        "bott_integer_error_pre": bott_int_err_pre,
        "bott_integer_error_post": bott_int_err_post,
        "qwz_u_eff_pre": u_eff_pre,
        "qwz_u_eff_post": u_eff_post,
        "qwz_gap_pre": gap_pre,
        "qwz_gap_post": gap_post,
        "final_S": float(out["S"][-1]),
        "final_pi_a": float(out["pi_a"][-1]),
        "gmres_fails": float(out["gmres_fails"]),
    }


# ============================================================ #
#  Ablation comparison                                          #
# ============================================================ #

def compare_ablations(
    nx: int,
    ny: int,
    T: float,
    dt: float,
    seed: int,
    damage_time: float,
    mass: float,
    init_phase_noise: float,
    qzw_pi_gain: float,
    qzw_entropy_gain: float,
    damage_scenario: str = "central_strip",
) -> Dict[str, Tuple[Any, Dict[str, Any], Dict[str, float]]]:
    """Run three ablation variants and return results dict.

    Variants
    --------
    full        : Standard EGATL (entropy-gated + adaptive pi_a)
    no_entropy  : Entropy held at S_eq (no gating)
    no_ruler    : pi_a held at pi0 (no adaptation)
    """
    base_eg = EGATLParams(
        alpha0=2.5, S_c=0.3, dS=0.5, mu0=0.25, S0=2.0,
        lambda_s=0.05, g_min=0.01, g_max=5.0,
        g_imag_max=3.0, budget_re=120.0,
        init_phase_noise=init_phase_noise,
        qzw_pi_gain=qzw_pi_gain,
        qzw_entropy_gain=qzw_entropy_gain,
    )
    base_ent = EntropyParams(S_init=1.5, S_eq=1.0, gamma=0.20,
                              kappa_slip=0.10, Tij=1.0)
    base_ruler = RulerParams(pi0=math.pi, pi_init=math.pi,
                              alpha_pi=0.15, mu_pi=0.15,
                              pi_min=0.5, pi_max=12.0)

    # Ablation 1: no entropy (gamma→∞ so S→S_eq instantly, kappa_slip=0)
    ent_frozen = EntropyParams(S_init=1.0, S_eq=1.0, gamma=999.0,
                               kappa_slip=0.0, Tij=1.0)

    # Ablation 2: no ruler (alpha_pi=0, so pi_a stays at pi0)
    ruler_frozen = RulerParams(pi0=math.pi, pi_init=math.pi,
                                alpha_pi=0.0, mu_pi=1.0,
                                pi_min=0.5, pi_max=12.0)

    results: Dict[str, Tuple[Any, Dict[str, Any], Dict[str, float]]] = {}

    for name, eg, ent, ruler in [
        ("full", base_eg, base_ent, base_ruler),
        ("no_entropy", base_eg, ent_frozen, base_ruler),
        ("no_ruler", base_eg, base_ent, ruler_frozen),
    ]:
        lat, out = run_recovery_protocol(
            nx=nx, ny=ny, T=T, dt=dt, seed=seed,
            damage_time=damage_time, mass=mass,
            damage_factor=1e-4,
            damage_scenario=damage_scenario,
            eg=eg, ent=ent, ruler=ruler,
        )
        summ = summarize_recovery(
            out,
            lat,
            damage_time,
            mass0=mass,
            pi_gain=qzw_pi_gain,
            entropy_gain=qzw_entropy_gain,
        )
        results[name] = (lat, out, summ)

    return results
