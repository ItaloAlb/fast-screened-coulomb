"""
Screened Coulomb interaction in a layered dielectric system.

Implements the Electrostatic Transfer Matrix (ETM) method from:
  Cavalcante et al., Phys. Rev. B 97, 125427 (2018)
  "Electrostatics of electron-hole interactions in van der Waals heterostructures"

Computes V_eh(rho) — the electron-hole interaction potential — for a stack of
dielectric slabs. The T-matrices at each interface are evaluated analytically
(no symbolic algebra needed), avoiding the numerical overflow that SymPy's
lambdify produces with raw exponentials.

Asymptotic subtraction is used to accelerate convergence of the Fourier-Bessel
integral: as k → ∞, ε(k) → ε_c (the local dielectric constant of the source
layer), so the subtracted integrand decays exponentially (~e^{-2k·d_min}).
"""

import warnings
import numpy as np
from scipy.integrate import quad
from scipy.special import j0
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Transfer-matrix core
# ---------------------------------------------------------------------------

def _safe_exp(x, max_arg=500.0):
    """exp(x) capped to avoid float64 overflow (max ~exp(709))."""
    if x > max_arg:
        return np.inf
    if x < -max_arg:
        return 0.0
    return np.exp(x)


def _T_matrix(k, dn, eps_lo, eps_hi):
    """
    Transfer matrix at a single interface (Eq. 5 of the paper).

    T_n = M_n^{-1} · M̄_n, evaluated analytically:

        T_n = [[ a,   b·e^{-2k·dn} ],
               [ b·e^{2k·dn},  a   ]]

    with  a = (ε_hi + ε_lo) / (2 ε_hi)
          b = (ε_hi - ε_lo) / (2 ε_hi)

    For large |2k·dn| the exponentials are safely capped; when the
    subtracted integrand is non-negligible the arguments stay moderate.
    """
    a = (eps_hi + eps_lo) / (2.0 * eps_hi)
    b = (eps_hi - eps_lo) / (2.0 * eps_hi)
    arg = 2.0 * k * dn
    em2kd = _safe_exp(-arg)
    e2kd = _safe_exp(arg)
    return np.array([[a, b * em2kd],
                     [b * e2kd, a]])


def _M_bar(k, dn, eps_lo):
    """M̄_n matrix (Eq. 5, left)."""
    ekd = np.exp(k * dn)
    emkd = 1.0 / ekd
    return np.array([[ekd, emkd],
                     [eps_lo * ekd, -eps_lo * emkd]])


def _M_inv(k, dn, eps_hi):
    """Analytic inverse of M_n (Eq. 5, right)."""
    emkd = np.exp(-k * dn)
    ekd = 1.0 / emkd
    return np.array([[0.5 * emkd, 0.5 * emkd / eps_hi],
                     [0.5 * ekd, -0.5 * ekd / eps_hi]])


def _compute_coefficients(epsilons, d, c, k, t):
    """
    Compute A_t(k), B_t(k) — the potential coefficients at layer t
    for a source charge in layer c.  Eqs. (4)–(7) of the PRB paper.

    Returns (A_t, B_t).
    """
    N = len(epsilons)
    eps_c = epsilons[c - 1]

    # --- global transfer matrix  M = T_{N-1} … T_1 ---
    M = np.eye(2)
    for n in range(N - 1, 0, -1):
        M = M @ _T_matrix(k, d[n - 1], epsilons[n - 1], epsilons[n])

    # --- M' = T_{N-1} … T_c · M_{c-1}^{-1} ---
    Mp = np.eye(2)
    for n in range(N - 1, c - 1, -1):
        Mp = Mp @ _T_matrix(k, d[n - 1], epsilons[n - 1], epsilons[n])
    Mp = Mp @ _M_inv(k, d[c - 2], epsilons[c - 1])

    # --- M'' = T_{N-1} … T_{c+1} · M_c^{-1} ---
    Mpp = np.eye(2)
    for n in range(N - 1, c, -1):
        Mpp = Mpp @ _T_matrix(k, d[n - 1], epsilons[n - 1], epsilons[n])
    Mpp = Mpp @ _M_inv(k, d[c - 1], epsilons[c])

    # --- A₁ from Eq. (7) ---
    term1 = (Mp[0, 0] + eps_c * Mp[0, 1]) * np.exp(k * d[c - 2])
    term2 = (Mpp[0, 0] - Mpp[0, 1] * eps_c) * np.exp(-k * d[c - 1])
    A1 = (term1 - term2) / M[0, 0]

    # --- propagate (A₁, B₁=0) → (A_t, B_t) via Eq. (4) ---
    coeffs = np.array([A1, 0.0])

    for n in range(1, t):
        delta1 = np.zeros(2)
        if n == c - 1:
            ek = np.exp(k * d[c - 2])
            delta1 = np.array([ek, eps_c * ek])

        delta2 = np.zeros(2)
        if n == c:
            emk = np.exp(-k * d[c - 1])
            delta2 = np.array([emk, -eps_c * emk])

        rhs = _M_bar(k, d[n - 1], epsilons[n - 1]) @ coeffs - delta1 + delta2

        # solve M_n · coeffs_new = rhs
        dn = d[n - 1]
        ekd = np.exp(k * dn)
        emkd = 1.0 / ekd
        eps_next = epsilons[n]
        M_mat = np.array([[ekd, emkd],
                          [eps_next * ekd, -eps_next * emkd]])
        coeffs = np.linalg.solve(M_mat, rhs)

    return coeffs[0], coeffs[1]


def _integrand_subtracted(k, epsilons, d, c, rho, t=None, z_t=0.0):
    """The subtracted integrand (1/ε − 1/ε_c)·J₀(kρ) for diagnostics."""
    k_safe = max(k, 1e-15)
    eps_c = epsilons[c - 1]
    ek = compute_epsilon_k(epsilons, d, c, k_safe, t, z_t)
    return (1.0 / ek - 1.0 / eps_c) * j0(k_safe * rho)


def optimal_parameters(epsilons, d, c, t=None, z_t=0.0, tol=1e-4,
                       tol_V=1e-3, safety=1.5):
    """
    Automatically determine safe integration parameters and the physical
    cutoff distance for a given dielectric stack configuration.

    Parameters
    ----------
    epsilons : list of float
        Dielectric constants (N layers, bottom to top).
    d : list of float
        Interface positions (N−1).  Origin at centre of source layer.
    c : int (1-based)
        Source-charge layer.
    t : int or None
        Observation layer.  None → t = c (direct exciton).
    z_t : float
        z-coordinate of observation point (only used when t ≠ c).
    tol : float
        Target relative accuracy on the k-integration envelope.
        Default 1e-4 → ~0.01 % on V(ρ).
    tol_V : float
        Potential threshold (Ry) below which V(ρ) is considered zero.
        Default 1e-3 Ry ≈ 14 meV.  Smaller → larger rho_max.
    safety : float
        Safety factor applied to the estimated k_max (> 1).

    Returns
    -------
    k_max : float
        Recommended upper integration limit (a₀⁻¹).
    n_quad : int
        Recommended quad `limit`.
    rho_max : float
        Physical cutoff (a₀) where |V(ρ)| < tol_V for all ρ > rho_max.
    info : dict
        Diagnostic fields: d_min, is_direct, asymptotic, k_envelope,
        eps_0, rho_max, tol_V.
    """
    if t is None:
        t = c
    is_direct = (t == c and abs(z_t) < 1e-15)
    eps_c = epsilons[c - 1]

    # --- find k where the envelope drops below tolerance ---
    k_grid = np.logspace(-1.5, 1.8, 200)   # 0.03 → 63

    if is_direct:
        target = tol / eps_c
        dev_vals = []
        for kval in k_grid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ek = compute_epsilon_k(epsilons, d, c, kval, t, z_t)
                dev_vals.append(abs(1.0 / ek - 1.0 / eps_c))
            except Exception:
                dev_vals.append(np.inf)
        dev_vals = np.array(dev_vals)
        asymptotic = eps_c
    else:
        target = tol / eps_c
        dev_vals = []
        for kval in k_grid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ek = compute_epsilon_k(epsilons, d, c, kval, t, z_t)
                dev_vals.append(1.0 / ek)
            except Exception:
                dev_vals.append(np.inf)
        dev_vals = np.array(dev_vals)
        asymptotic = np.inf

    # Find first k where deviation < target
    idx = np.where(dev_vals < target)[0]
    if len(idx) > 0:
        k_envelope = float(k_grid[idx[0]])
    else:
        k_envelope = float(k_grid[-1])

    k_max = k_envelope * safety

    # --- n_quad: fixed conservative value ---
    n_quad = 200

    # --- estimate rho_max: where V(rho) drops below tol_V ---
    # At large rho, V(rho) ~ 2 / (eps_0 * rho)  where eps_0 = eps(k→0).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            eps_0 = compute_epsilon_k(epsilons, d, c, 1e-4, t, z_t)
        except Exception:
            eps_0 = eps_c  # fallback
    rho_max = 2.0 / (eps_0 * tol_V) if eps_0 > 1e-15 else np.inf

    # --- extract d_min for diagnostics ---
    if c > 1 and c < len(epsilons):
        d_below = abs(d[c - 2])
    else:
        d_below = np.inf
    if c < len(epsilons):
        d_above = abs(d[c - 1])
    else:
        d_above = np.inf
    d_min = min(d_below, d_above)

    info = dict(
        d_min=d_min,
        is_direct=is_direct,
        asymptotic=asymptotic,
        k_envelope=k_envelope,
        target=target,
        eps_0=eps_0,
        rho_max=rho_max,
        tol_V=tol_V,
    )

    return k_max, n_quad, rho_max, info


def tolerance_timing(epsilons=None, d_array=None, c_source=None,
                     t_obs=None, z_t=None,
                     tol_vals=None, tol_V_vals=None,
                     n_warmup=3, n_repeat=5):
    """
    Measure how the single-ρ computation time depends on the two
    tolerances  *tol*  (k-integration accuracy) and  *tol_V*
    (potential cutoff).

    For each (tol, tol_V) pair the optimal parameters are computed and
    then V(ρ) is timed at the worst-case ρ = rho_max (fastest Bessel
    oscillations for that tol_V).  The result is a 2-D colormap of
    wall-clock time vs. tolerance.

    Defaults target the bilayer system from `main()`.
    """
    import time

    # --- defaults ---
    if epsilons is None:
        epsilons = [1.0, 14.0, 14.0, 1.0]
    if d_array is None:
        d_array = [-3.0, 3.0, 9.0]
    if c_source is None:
        c_source = 2
    if t_obs is None:
        t_obs = c_source
        z_t = 0.0
    if z_t is None:
        z_t = 0.0
    if tol_vals is None:
        tol_vals = np.logspace(-5, -1, 12)      # 1e-5 → 0.1
    if tol_V_vals is None:
        tol_V_vals = np.logspace(-6, -1, 12)     # 1e-6 → 0.1

    is_direct = (t_obs == c_source and abs(z_t) < 1e-15)

    times = np.zeros((len(tol_V_vals), len(tol_vals)))
    k_max_grid = np.zeros_like(times)
    rho_max_grid = np.zeros_like(times)

    for i, tol_V in enumerate(tol_V_vals):
        for j, tol in enumerate(tol_vals):
            km, nq, rhom, _ = optimal_parameters(
                epsilons, d_array, c_source, t=t_obs, z_t=z_t,
                tol=tol, tol_V=tol_V)

            k_max_grid[i, j] = km
            rho_max_grid[i, j] = rhom

            # Increase n_quad for large rho (fast Bessel oscillations)
            nq_actual = max(nq, int(np.ceil(km * rhom / 2.0)))

            # warmup (suppress quad warnings)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(n_warmup):
                    if is_direct:
                        V_eh_direct([rhom], epsilons, d_array, c_source,
                                    k_max=km, n_quad=nq_actual)
                    else:
                        V_eh([rhom], epsilons, d_array, c_source,
                             t=t_obs, z_t=z_t, k_max=km, n_quad=nq_actual)

            # timed
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t_start = time.perf_counter()
                for _ in range(n_repeat):
                    if is_direct:
                        V_eh_direct([rhom], epsilons, d_array, c_source,
                                    k_max=km, n_quad=nq_actual)
                    else:
                        V_eh([rhom], epsilons, d_array, c_source,
                             t=t_obs, z_t=z_t, k_max=km, n_quad=nq_actual)
                t_end = time.perf_counter()

            times[i, j] = (t_end - t_start) / n_repeat

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: time colormap
    im = axes[0].pcolormesh(tol_vals, tol_V_vals, times * 1000,
                            shading='gouraud', cmap='YlOrRd')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('tol  (k-integration)')
    axes[0].set_ylabel('tol_V  (potential cutoff, Ry)')
    axes[0].set_title('Time per ρ point (ms)')
    plt.colorbar(im, ax=axes[0], label='ms')

    # Panel 2: k_max colormap
    im2 = axes[1].pcolormesh(tol_vals, tol_V_vals, k_max_grid,
                             shading='gouraud', cmap='viridis')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('tol  (k-integration)')
    axes[1].set_ylabel('tol_V  (potential cutoff, Ry)')
    axes[1].set_title(r'Optimal $k_{\rm max}$ ($a_0^{-1}$)')
    plt.colorbar(im2, ax=axes[1])

    # Panel 3: rho_max colormap
    im3 = axes[2].pcolormesh(tol_vals, tol_V_vals, rho_max_grid * 0.529,
                             shading='gouraud', cmap='cividis')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('tol  (k-integration)')
    axes[2].set_ylabel('tol_V  (potential cutoff, Ry)')
    axes[2].set_title(r'Optimal $\rho_{\rm max}$ (Ang)')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # --- print key numbers ---
    print("=" * 60)
    print("Tolerance timing summary")
    print(f"  tol range:      {tol_vals[0]:.0e} -- {tol_vals[-1]:.0e}")
    print(f"  tol_V range:    {tol_V_vals[0]:.0e} -- {tol_V_vals[-1]:.0e} Ry")
    print(f"  time range:     {np.min(times)*1000:.4f} -- "
          f"{np.max(times)*1000:.2f} ms/point")
    print(f"  k_max range:    {np.min(k_max_grid):.2f} -- "
          f"{np.max(k_max_grid):.2f} a0^-1")
    print(f"  rho_max range:  {np.min(rho_max_grid)*0.529:.0f} -- "
          f"{np.max(rho_max_grid)*0.529:.0f} Ang")
    print("=" * 60)

    return times, k_max_grid, rho_max_grid, tol_vals, tol_V_vals


# ---------------------------------------------------------------------------
# Analytical-fit benchmarks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Expansion functions
# ---------------------------------------------------------------------------

def _fit_pade12(rho, a, b, c, d):
    """Padé [1/2] approximant: (a + b*rho) / (c*rho + d*rho^2)."""
    return (a + b * rho) / (c * rho + d * rho**2)


def _fit_chebyshev(rho, rho_min, rho_max, *coeffs):
    """Evaluate Chebyshev expansion on mapped rho -> x in [-1,1]."""
    coeffs = np.asarray(coeffs)
    x = (2.0 * rho - rho_min - rho_max) / (rho_max - rho_min)
    return np.polynomial.chebyshev.chebval(x, coeffs)


def _fit_legendre(rho, rho_min, rho_max, *coeffs):
    """Evaluate Legendre expansion on mapped rho -> x in [-1,1]."""
    coeffs = np.asarray(coeffs)
    x = (2.0 * rho - rho_min - rho_max) / (rho_max - rho_min)
    return np.polynomial.legendre.legval(x, coeffs)


def _fit_image_charges(rho, *params):
    """Image-charge expansion: V(rho) = sum_i q_i / sqrt(rho^2 + z_i^2).

    params = [q1, z1, q2, z2, ...]  — (q_i, z_i) pairs.
    """
    n_pairs = len(params) // 2
    result = np.zeros_like(rho, dtype=float)
    for i in range(n_pairs):
        q = params[2 * i]
        z = abs(params[2 * i + 1]) + 1e-15  # ensure z > 0
        result += q / np.sqrt(rho**2 + z**2)
    return result


def _make_pade_mn(m, n):
    """Factory: Pade [m/n] approximant with (m+1)+(n) free parameters."""
    def f(rho, *p):
        num = sum(p[i] * rho**i for i in range(m + 1))
        den = sum(p[m + 1 + i] * rho**(i + 1) for i in range(n))
        return num / den
    return f, m + n + 1


def fit_benchmark(epsilons=None, d_array=None, c_source=None,
                  t_obs=None, z_t=None,
                  tol=1e-4, tol_V=1e-2,
                  n_fit_pts=30, n_eval_pts=200,
                  rho_range=None,
                  targets=None,
                  show_plots=True, verbose=True):
    """
    Compare expansions by fixing the *target accuracy* and letting each
    method use however many parameters it needs to reach it.  Reports
    parameter count, evaluation time, and speedup vs the exact integral.

    Methods:  Pade [m/n], Chebyshev (fit rho*V), Legendre (fit rho*V),
              image charges, cubic B-spline (fit rho*V).

    Parameters
    ----------
    targets : list of float
        Target max errors (%).  Default: [10, 5, 2, 1, 0.5, 0.2].
    """
    import time
    from scipy.optimize import curve_fit
    from scipy.interpolate import make_lsq_spline

    if targets is None:
        targets = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]

    # --- defaults ---
    if epsilons is None:
        epsilons = [1.0, 14.0, 14.0, 1.0]
    if d_array is None:
        d_array = [-3.0, 3.0, 9.0]
    if c_source is None:
        c_source = 2
    if t_obs is None:
        t_obs = c_source; z_t = 0.0
    if z_t is None:
        z_t = 0.0
    is_direct = (t_obs == c_source and abs(z_t) < 1e-15)

    km, nq, rhom_opt, _ = optimal_parameters(
        epsilons, d_array, c_source, t=t_obs, z_t=z_t,
        tol=tol, tol_V=tol_V)

    if rho_range is None:
        rho_range = (0.1, rhom_opt)
    rho_min, rho_max = rho_range

    rho_fit = np.logspace(np.log10(rho_min), np.log10(rho_max), n_fit_pts)
    rho_eval = np.logspace(np.log10(rho_min), np.log10(rho_max), n_eval_pts)

    nq_act = max(nq, int(np.ceil(km * rho_max / 2.0)))
    V_kw = dict(k_max=km, n_quad=nq_act)
    if is_direct:
        V_fit = V_eh_direct(rho_fit, epsilons, d_array, c_source, **V_kw)
        V_exact = V_eh_direct(rho_eval, epsilons, d_array, c_source, **V_kw)
    else:
        V_fit = V_eh(rho_fit, epsilons, d_array, c_source,
                     t=t_obs, z_t=z_t, **V_kw)
        V_exact = V_eh(rho_eval, epsilons, d_array, c_source,
                       t=t_obs, z_t=z_t, **V_kw)

    rhoV_fit = rho_fit * V_fit
    x_fit = (2.0 * rho_fit - rho_min - rho_max) / (rho_max - rho_min)
    x_eval = (2.0 * rho_eval - rho_min - rho_max) / (rho_max - rho_min)

    # --- helper: find minimum params to reach target ---
    def _find_fit(name, param_sweep, error_fn_wrapper):
        """Sweep params ascending, return (n_params, max_err, t_us) at first hit."""
        for n_params in param_sweep:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    fit_result = error_fn_wrapper(n_params)
                    if fit_result is None:
                        continue
                    max_err, eval_fn = fit_result
                except Exception:
                    continue
            if max_err <= targets[0]:  # at least good enough for coarsest target
                # time it — evaluate whole array at once (vectorized)
                t0 = time.perf_counter()
                for _ in range(5000):
                    eval_fn(rho_eval)
                t_us = (time.perf_counter() - t0) / 5000 * 1e6 / len(rho_eval)
                return n_params, max_err, t_us, eval_fn
        return None, np.inf, np.inf, None

    # build all fits
    methods = {}

    # -- Pade --
    for m, n in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
        pade_func, np_ = _make_pade_mn(m, n)
        p0 = [1.0] * np_
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                popt, _ = curve_fit(pade_func, rho_fit, V_fit, p0=p0, maxfev=5000)
                V_f = pade_func(rho_eval, *popt)
                err = np.max(np.abs(V_f - V_exact) / V_exact * 100)
                methods[f'Pade [{m}/{n}]'] = (
                    np_, err,
                    lambda r, f=pade_func, p=popt: f(r, *p))
            except Exception:
                pass

    # -- Chebyshev (fit rho*V) --
    for deg in [3, 5, 7, 9, 11, 15, 20, 25, 30]:
        cc = np.polynomial.chebyshev.chebfit(x_fit, rhoV_fit, deg)
        V_ch = np.polynomial.chebyshev.chebval(x_eval, cc) / rho_eval
        err = np.max(np.abs(V_ch - V_exact) / V_exact * 100)
        methods[f'Chebyshev d={deg}'] = (
            deg + 1, err,
            lambda r, c=cc, rm=rho_min, rx=rho_max:
                np.polynomial.chebyshev.chebval(
                    (2.0*r - rm - rx)/(rx - rm), c) / r)

    # -- Legendre (fit rho*V) --
    for deg in [3, 5, 7, 9, 11, 15, 20, 25, 30]:
        cl = np.polynomial.legendre.legfit(x_fit, rhoV_fit, deg)
        V_lg = np.polynomial.legendre.legval(x_eval, cl) / rho_eval
        err = np.max(np.abs(V_lg - V_exact) / V_exact * 100)
        methods[f'Legendre d={deg}'] = (
            deg + 1, err,
            lambda r, c=cl, rm=rho_min, rx=rho_max:
                np.polynomial.legendre.legval(
                    (2.0*r - rm - rx)/(rx - rm), c) / r)

    # -- Image charges --
    for n_img in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        zs = np.logspace(np.log10(rho_min*0.5 + 1e-6),
                         np.log10(rho_max*2.0), n_img)
        A = np.array([[1.0/np.sqrt(r**2 + z**2) for z in zs] for r in rho_fit])
        q = np.linalg.lstsq(A, V_fit, rcond=None)[0]
        V_img = np.zeros_like(rho_eval)
        for qi, zi in zip(q, zs):
            V_img += qi / np.sqrt(rho_eval**2 + zi**2)
        err = np.max(np.abs(V_img - V_exact) / V_exact * 100)
        methods[f'Image charges n={n_img}'] = (
            n_img, err,
            lambda r, qs=q, zs=zs:
                np.sum(qs[:, None] / np.sqrt(r[None, :]**2 + zs[:, None]**2),
                       axis=0))

    # -- B-spline (fit rho*V) --
    for n_knots in [0, 1, 2, 3, 5, 8, 12, 20, 30]:
        deg3 = 3
        try:
            if n_knots == 0:
                t_bs = np.r_[[rho_fit[0]]*(deg3+1), [rho_fit[-1]]*(deg3+1)]
            else:
                pct = np.linspace(5, 95, n_knots)
                inner = np.percentile(rho_fit, pct)
                t_bs = np.r_[[rho_fit[0]]*(deg3+1), inner,
                              [rho_fit[-1]]*(deg3+1)]
            spl = make_lsq_spline(rho_fit, rhoV_fit, t_bs)
            V_bs = spl(rho_eval) / rho_eval
            err = np.max(np.abs(V_bs - V_exact) / V_exact * 100)
            methods[f'B-spline k={n_knots}'] = (
                len(spl.c), err,
                lambda r, s=spl: s(r) / r)
        except Exception:
            pass

    # --- for each target, pick the cheapest method that meets it ---
    # time exact integral
    t0 = time.perf_counter()
    for _ in range(50):
        if is_direct:
            V_eh_direct(rho_eval[:1], epsilons, d_array, c_source, **V_kw)
        else:
            V_eh(rho_eval[:1], epsilons, d_array, c_source,
                 t=t_obs, z_t=z_t, **V_kw)
    t_exact_ms = (time.perf_counter() - t0) / 50 * 1000

    # build result matrix: targets x methods
    method_names = ['Pade', 'Chebyshev', 'Legendre', 'Image chg', 'B-spline']
    all_results = {tgt: {} for tgt in targets}

    for tgt in targets:
        for mname in method_names:
            best_np, best_err, best_tus = np.inf, np.inf, np.inf
            for key, (np_, err, fn) in methods.items():
                if mname.lower() in key.lower().replace('image charges', 'image chg').replace('cubic b-spline', 'b-spline'):
                    if err <= tgt and np_ < best_np:
                        best_np, best_err, best_tus = np_, err, None
            # time the best one
            for key, (np_, err, fn) in methods.items():
                if mname.lower() in key.lower().replace('image charges', 'image chg').replace('cubic b-spline', 'b-spline'):
                    if np_ == best_np and err == best_err:
                        t0 = time.perf_counter()
                        for _ in range(5000):
                            fn(rho_eval)
                        best_tus = (time.perf_counter()-t0)/5000*1e6/len(rho_eval)
                        break
            if best_np < np.inf:
                all_results[tgt][mname] = (best_np, best_err, best_tus)

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'Pade': 'crimson', 'Chebyshev': 'steelblue', 'Legendre': 'forestgreen',
              'Image chg': 'darkorange', 'B-spline': 'purple'}

    for mname in method_names:
        pts = [(t, all_results[t][mname][0])
               for t in targets if mname in all_results[t]]
        if pts:
            ts, ps = zip(*pts)
            ax1.plot(ts, ps, 'o-', color=colors[mname], markersize=6,
                     linewidth=1.5, label=mname)

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.set_xlabel('Target max error (%)')
    ax1.set_ylabel('Parameters needed')
    ax1.set_title('Cost to reach accuracy target')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)

    for mname in method_names:
        pts = [(t, all_results[t][mname][2])
               for t in targets if mname in all_results[t]]
        if pts:
            ts, us = zip(*pts)
            ax2.loglog(ts, us, 's-', color=colors[mname], markersize=6,
                       linewidth=1.5, label=mname)
    # exact reference line
    ax2.axhline(t_exact_ms * 1000, color='black', linestyle=':', linewidth=1,
                label=f'Exact ({t_exact_ms:.0f} ms)')
    ax2.invert_xaxis()
    ax2.set_xlabel('Target max error (%)')
    ax2.set_ylabel('Eval time (us/point)')
    ax2.set_title('Evaluation speed vs accuracy target')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)

    if show_plots:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    if verbose:
        print("=" * 90)
        print("Fit benchmark — target-driven")
        print(f"  config: eps={epsilons}, d={d_array}, c={c_source}")
        print(f"  exact integral: {t_exact_ms:.2f} ms/point")
        print()
        for tgt in targets:
            print(f"  --- target: max error < {tgt:.1f}% ---")
            hdr = (f"  {'method':>20s}  {'params':>7s}  {'max err %':>10s}  "
                   f"{'t_eval':>10s}  {'speedup':>10s}")
            print(hdr)
            print("  " + "-" * 68)
            for mname in method_names:
                if mname in all_results[tgt]:
                    np_, err, tus = all_results[tgt][mname]
                    speedup = t_exact_ms * 1000 / tus
                    print(f"  {mname:>20s}  {np_:7d}  {err:10.4f}  "
                          f"{tus:8.2f} us  {speedup:9.0f}x")
            print()
        print("=" * 90)
    return all_results, rho_eval, V_exact


def bulk_fit_benchmark(targets=None, tol=1e-4, tol_V=1e-2,
                       logfile='fit_benchmark_results.log'):
    """
    Run fit_benchmark across many physical configurations and save
    aggregate statistics to a log file.

    Sweeps: d (Ang), eps_c, N=3 mono / N=4 buried / N=4 hetero,
            direct & indirect excitons.
    """
    import time as _time
    from datetime import datetime

    if targets is None:
        targets = [10.0, 5.0, 2.0, 1.0, 0.5]

    ang2a0 = 1.0 / 0.529177

    configs = []
    for d_ang in [3.0, 5.0, 7.0]:
        d_a0 = d_ang * ang2a0
        d_half = d_a0 / 2.0
        for eps_c in [5.0, 14.0]:
            # N=3 monolayer
            configs.append(('N3 mono', [1.0, eps_c, 1.0],
                           [-d_half, d_half], 2, None, 0.0))
            # N=4 buried
            configs.append(('N4 buried', [1.0, eps_c, eps_c, 1.0],
                           [-d_half, d_half, d_half + 6.0], 2, None, 0.0))
            # N=4 hetero
            configs.append(('N4 hetero', [1.0, eps_c, 5.0, 1.0],
                           [-d_half, d_half, d_half + 6.0], 2, None, 0.0))
            # indirect: bilayer (same eps on both TMD slabs)
            configs.append(('indirect', [1.0, eps_c, eps_c, 1.0],
                           [-d_a0/2, d_a0/2, 3*d_a0/2], 2, 3, d_a0))

    all_runs = []

    with open(logfile, 'w', encoding='utf-8') as log:
        log.write(f"# Bulk fit benchmark — {datetime.now()}\n")
        log.write(f"# tol={tol:.0e}  tol_V={tol_V:.0e}\n")
        log.write(f"# {len(configs)} configurations\n")
        log.write("#" + "=" * 79 + "\n\n")

        for idx, (name, eps, d_arr, c, t, zt) in enumerate(configs):
            t0 = _time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    results, _, _ = fit_benchmark(
                        epsilons=eps, d_array=d_arr,
                        c_source=c, t_obs=t, z_t=zt,
                        tol=tol, tol_V=tol_V,
                        targets=targets,
                        show_plots=False, verbose=False)
                except Exception as e:
                    log.write(f"[{idx+1}/{len(configs)}] {name}  FAILED: {e}\n\n")
                    continue
            elapsed = _time.perf_counter() - t0

            # Extract per-target stats
            log.write(f"[{idx+1}/{len(configs)}] {name}  "
                      f"eps={eps}  d={d_arr}  c={c}  t={t}  zt={zt}  "
                      f"({elapsed:.1f}s)\n")
            for tgt in targets:
                log.write(f"  target <{tgt:.1f}%:\n")
                for mname in results[tgt]:
                    np_, err, tus = results[tgt][mname]
                    log.write(f"    {mname:>20s}  params={np_:3d}  "
                              f"err={err:.4f}%  t_eval={tus:.3f} us/pt\n")
                    all_runs.append(dict(
                        config=name, d_ang=d_ang, eps_c=eps_c,
                        target=tgt, method=mname,
                        n_params=np_, max_err=err, t_eval_us=tus))
            log.write("\n")
            log.flush()

    # --- aggregate statistics ---
    print("=" * 78)
    print("Aggregate statistics across all configurations")
    print(f"  {len(configs)} configs x {len(targets)} targets = "
          f"{len(all_runs)} data points")
    print()

    method_names = list(dict.fromkeys(r['method'] for r in all_runs))
    for tgt in targets:
        print(f"  --- target < {tgt:.1f}% ---")
        print(f"  {'method':>20s}  {'reach %':>8s}  {'mean p':>7s}  "
              f"{'std p':>7s}  {'mean t(us)':>10s}  {'std t':>7s}")
        print("  " + "-" * 60)
        for mname in method_names:
            subset = [r for r in all_runs
                      if r['method'] == mname and r['target'] == tgt]
            if not subset:
                continue
            reach = len(subset) / len(configs) * 100
            np_vals = [r['n_params'] for r in subset]
            t_vals = [r['t_eval_us'] for r in subset]
            print(f"  {mname:>20s}  {reach:7.1f}%  "
                  f"{np.mean(np_vals):7.1f}  {np.std(np_vals):7.1f}  "
                  f"{np.mean(t_vals):9.3f}  {np.std(t_vals):7.3f}")
        print()

    print(f"  Results saved to: {logfile}")
    print("=" * 78)
    return all_runs


def compute_epsilon_k(epsilons, d, c, k, t=None, z_t=0.0):
    """
    Effective dielectric function ε_{t,c}(k) from Eq. (3) of the paper:

        ε_{t,c}(k) = ε_c / [A_t e^{k z_t} + B_t e^{-k z_t} + δ_{t,c}]

    Parameters
    ----------
    epsilons : list of float
        Dielectric constants (N layers, bottom to top).
    d : list of float
        Interface positions (N-1 values).  Origin at centre of source layer.
    c : int (1-based)
        Source-charge layer.
    k : float
        Wave-vector magnitude.
    t : int or None
        Observation layer (1-based).  Default None → t = c (direct exciton).
    z_t : float
        z-coordinate of the observation point relative to the origin (centre
        of layer c).  Only used when t is specified.
    """
    if t is None:
        t = c
    eps_c = epsilons[c - 1]
    delta_tc = 1.0 if t == c else 0.0

    At, Bt = _compute_coefficients(epsilons, d, c, k, t)
    denom = At * np.exp(k * z_t) + Bt * np.exp(-k * z_t) + delta_tc
    return eps_c / denom


# ---------------------------------------------------------------------------
# Real-space potential  (general & direct-exciton convenience wrapper)
# ---------------------------------------------------------------------------

def V_eh(rhos, epsilons, d, c, t=None, z_t=0.0, k_max=15.0, n_quad=200):
    """
    Electron-hole interaction potential V_eh(ρ) for arbitrary source (c)
    and observation (t) layers, Eq. (3) of the PRB paper:

        V(ρ) = 2 ∫₀^∞ J₀(kρ) / ε_{t,c}(k)  dk                    [in Ry]

    For *direct* excitons (t = c, z_t = 0) asymptotic subtraction is used
    (ε → ε_c as k → ∞).  For *indirect* excitons (t ≠ c) the integral is
    computed directly because ε(k) → ∞ as k → ∞, so 1/ε(k) → 0 naturally.
    """
    if t is None:
        t = c

    is_direct = (t == c and abs(z_t) < 1e-15)
    eps_c = epsilons[c - 1]

    V = np.zeros_like(rhos, dtype=float)

    for i, rho in enumerate(rhos):
        if rho < 1e-15:
            V[i] = np.inf
            continue

        if is_direct:
            # --- asymptotic subtraction: ε_inf = ε_c ---
            def integrand(k):
                k_safe = max(k, 1e-15)
                ek = compute_epsilon_k(epsilons, d, c, k_safe, t, z_t)
                return (1.0 / ek - 1.0 / eps_c) * j0(k_safe * rho)

            integral, _ = quad(integrand, 0.0, k_max,
                               limit=n_quad, epsabs=1e-10, epsrel=1e-8)
            V[i] = 2.0 * (integral + 1.0 / (eps_c * rho))
        else:
            # --- direct integration (ε → ∞ as k → ∞, no subtraction) ---
            def integrand(k):
                k_safe = max(k, 1e-15)
                ek = compute_epsilon_k(epsilons, d, c, k_safe, t, z_t)
                return j0(k_safe * rho) / ek

            integral, _ = quad(integrand, 0.0, k_max,
                               limit=n_quad, epsabs=1e-10, epsrel=1e-8)
            V[i] = 2.0 * integral

    return V


def V_eh_direct(rhos, epsilons, d, c, k_max=15.0, n_quad=200):
    """Convenience wrapper: direct exciton (t = c, z_t = 0)."""
    return V_eh(rhos, epsilons, d, c, t=c, z_t=0.0,
                k_max=k_max, n_quad=n_quad)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    # Dielectric environment (bottom → top)
    eps_array = [1.0, 14.0, 14.0, 1.0]

    # Interface positions; origin at centre of layer 2 (the source layer)
    # Layer 2: z ∈ [-3, 3] a₀  →  width 6 a₀ ≈ 3.15 Å (MoS₂ monolayer)
    # Layer 3: z ∈ [3, 9] a₀   →  width 6 a₀
    d_array = [-3.0, 3.0, 9.0]

    c_source = 2                       # charge in layer 2

    rho_array = np.linspace(0.1, 30.0, 50)

    V = V_eh_direct(rho_array, eps_array, d_array, c=c_source, k_max=15.0)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(rho_array, V, 'o-', color='crimson', markersize=4,
             label='ETM (this work)')
    plt.xlabel(r'In-plane distance $\rho$ ($a_0$)')
    plt.ylabel(r'$V_{eh}$ (Rydberg)')
    plt.title('Screened electron–hole interaction — direct exciton, bilayer')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("rho (a0):")
    print(np.array2string(rho_array, precision=3, max_line_width=80))
    print("\nV_eh (Ry):")
    print(np.array2string(V, precision=6, max_line_width=80))


# ---------------------------------------------------------------------------
# Bilayer effective R-K  (from the moire-exciton paper)
# ---------------------------------------------------------------------------

def epsilon_bilayer_exact(k, eps_e, eps_t, d):
    """
    Exact effective dielectric function for a bilayer heterostructure:
    two slabs of equal thickness *d* and dielectric constant ε_t, touching
    at z = 0, surrounded by a medium ε_e.  Eq. (14) of the moire paper.

    The source charge is at the centre of the bottom slab (z = −d/2) and
    the observation point is at the centre of the top slab (z = +d/2).

    Parameters
    ----------
    k : float or array
        Wave-vector magnitude (a₀⁻¹).
    eps_e : float
        Environment dielectric constant.
    eps_t : float
        TMD slab dielectric constant.
    d : float
        Single-slab thickness (half the total bilayer thickness), in a₀.

    Returns
    -------
    eps : float or array
        ε_eff(k).
    """
    ekd = np.exp(k * d)
    emkd = 1.0 / ekd

    num = ekd**2 * (eps_e + eps_t)**2 - emkd**2 * (eps_e - eps_t)**2
    den = emkd * (ekd * (eps_e + eps_t) - (eps_e - eps_t))**2

    return eps_t * num / den


def V_bilayer_eff_RK(rhos, eps_e, eps_t, d, alpha):
    """
    Effective Rytova-Keldysh potential for an interlayer (indirect) exciton
    in a TMD bilayer, Eq. (15) of the moire paper:

        V(ρ) = 4 / (ε_e ρ₀) [H₀(√(ρ²+d²)/ρ₀) − Y₀(√(ρ²+d²)/ρ₀)]

    with  ρ₀ = α · ε_t · d / (2 ε_e).

    NOTE: Eq. (15) in the paper writes the prefactor as 2e²/(π ε_e ρ₀).
    Converting to Rydberg units (e² = 2 Ry·a₀ in Gaussian, and the
    Fourier–Bessel integral contributes an extra factor of π/2 relative
    to the closed Struve–Bessel form) yields the factor 4/(ε_e ρ₀).

    Parameters
    ----------
    rhos : array
        In-plane distances (a₀).
    eps_e : float
        Environment dielectric constant.
    eps_t : float
        Slab dielectric constant.
    d : float
        Single-slab thickness (a₀).
    alpha : float
        Fitting parameter.  α = 1.8 (vacuum), 1.5 (hBN), 0.9 (sapphire).

    Returns
    -------
    V : ndarray
        Interaction potential magnitude |V| in Rydberg.
    """
    from scipy.special import struve, y0

    rho_0 = alpha * eps_t * d / (2.0 * eps_e)

    V = np.zeros_like(rhos, dtype=float)
    for i, rho in enumerate(rhos):
        r_eff = np.sqrt(rho**2 + d**2)
        x = r_eff / rho_0
        # prefactor = 2e²/(π ε_e ρ₀) → in Ry: 4/(ε_e ρ₀)
        # (the π from the Fourier–Bessel integral cancels the π in the
        #  paper's denominator)
        V[i] = 4.0 / (eps_e * rho_0) * (struve(0, x) - y0(x))

    return V

def epsilon_N3_analytic(k, eps_1, eps_2, eps_3, d_total):
    """
    Exact effective dielectric function for N = 3 (one finite slab between
    two semi-infinite media).  Eq. (14) of the paper.

    Parameters
    ----------
    k : float or array
        Wave-vector magnitude.
    eps_1, eps_2, eps_3 : float
        Dielectric constants (bottom, slab, top).
    d_total : float
        Total slab thickness  d₂ − d₁  (in a₀).

    Returns
    -------
    eps : float or array
        ε(k) for a charge in the slab (c = 2), in-plane (z = 0).
    """
    tanh_kd = np.tanh(k * d_total)
    sech_kd = 1.0 / np.cosh(k * d_total)

    A = eps_1 + eps_3
    B = 1.0 + eps_1 * eps_3 / eps_2**2
    C = 1.0 - eps_1 * eps_3 / eps_2**2

    numerator = A + B * eps_2 * tanh_kd
    denominator = B + C * sech_kd + (A / eps_2) * tanh_kd

    return numerator / denominator


def rytova_keldysh_potential(rhos, eps_1, eps_2, eps_3, d_total):
    """
    Rytova-Keldysh potential for a monolayer (thin slab, ε₂ ≫ ε₁,ε₃).

    Uses the R-K effective dielectric function  ε_RK(k) = (ε₁+ε₃)/2·(1 + ρ₀k)
    with  ρ₀ = ε₂ d_total / (ε₁+ε₃), integrated numerically via the
    Fourier–Bessel representation (Eqs. 3, 13 of the paper):

        V_RK(ρ) = 4/(ε₁+ε₃) ∫₀^∞ J₀(kρ) / (1 + ρ₀ k)  dk      [in Ry]

    The integral is computed with asymptotic subtraction
    (ε_RK(k → ∞) → ∞, so the subtracted term is zero for the tail).

    Parameters
    ----------
    rhos : array
        In-plane distances (in a₀).
    eps_1, eps_2, eps_3 : float
        Dielectric constants (bottom, slab, top).
    d_total : float
        Total slab thickness (in a₀).

    Returns
    -------
    V : ndarray
        R-K potential in Ry.
    """
    rho_0 = eps_2 * d_total / (eps_1 + eps_3)
    prefactor = 4.0 / (eps_1 + eps_3)          # 2 × e²/(2πε₀) / (ε₁+ε₃)  in Ry

    V = np.zeros_like(rhos, dtype=float)

    for i, rho in enumerate(rhos):
        if rho < 1e-15:
            V[i] = np.inf
            continue

        # asymptotic subtraction:  ε_RK(k) ~ (ε₁+ε₃)ρ₀k/2  as k → ∞
        # → 1/ε_RK(k) ~ 2/((ε₁+ε₃)ρ₀ k)
        # The "tail" integral ∫ J₀(kρ)/k dk is conditionally convergent;
        # we simply integrate to a large enough k_max where the Bessel
        # oscillations suppress further contribution.
        def integrand(k):
            k_safe = max(k, 1e-15)
            return j0(k_safe * rho) / (1.0 + rho_0 * k_safe)

        # k_max scales with 1/rho to capture enough Bessel oscillations
        kmax = max(30.0, 20.0 / rho)
        integral, _ = quad(integrand, 0.0, kmax, limit=300,
                           epsabs=1e-10, epsrel=1e-8)
        V[i] = prefactor * integral

    return V


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def V_from_epsilon_func(rhos, eps_func, eps_inf, k_max=25.0, n_quad=300):
    """
    Compute V(ρ) by numerically integrating a given ε(k) function.

        V(ρ) = 2 ∫₀^{k_max} J₀(kρ) / ε(k) dk   (+ asymptotic tail if eps_inf
                                                   is finite)

    If eps_inf is finite (ε(k) → ε_inf as k → ∞), asymptotic subtraction
    is used.  If eps_inf is inf, the integral is computed directly.

    Parameters
    ----------
    rhos : array
        In-plane distances.
    eps_func : callable
        Function ε(k) → float.
    eps_inf : float
        Asymptotic value ε(k→∞).  Use np.inf if ε → ∞.
    k_max : float
        Upper integration cutoff.
    n_quad : int
        Quadrature subintervals.

    Returns
    -------
    V : ndarray
        Potential in Ry.
    """
    use_subtraction = np.isfinite(eps_inf)
    V = np.zeros_like(rhos, dtype=float)

    for i, rho in enumerate(rhos):
        if rho < 1e-15:
            V[i] = np.inf
            continue

        if use_subtraction:
            def integrand(k):
                k_safe = max(k, 1e-15)
                return (1.0 / eps_func(k_safe) - 1.0 / eps_inf) * j0(k_safe * rho)
            integral, _ = quad(integrand, 0.0, k_max, limit=n_quad,
                               epsabs=1e-10, epsrel=1e-8)
            V[i] = 2.0 * (integral + 1.0 / (eps_inf * rho))
        else:
            def integrand(k):
                k_safe = max(k, 1e-15)
                return j0(k_safe * rho) / eps_func(k_safe)
            integral, _ = quad(integrand, 0.0, k_max, limit=n_quad,
                               epsabs=1e-10, epsrel=1e-8)
            V[i] = 2.0 * integral

    return V


def test_rytova_keldysh():
    """
    Test 1 — Rytova–Keldysh limit: 2×2 parameter sweep.

    Compares ETM (exact), Exact N=3, and R-K for four combinations of
    slab thickness d_total and dielectric constant ε₂, showing how the
    R-K error depends on BOTH parameters simultaneously.

    The 2×2 grid is:
      rows    : d_total = 0.5, 2.0  a₀   (thin → thick)
      columns : ε₂     = 4,   14          (weak → strong dielectric contrast)

    Each panel shows V(ρ) for the three methods plus the R-K error vs ρ.
    """
    from matplotlib import ticker

    d_totals = [0.5, 2.0]            # a₀
    eps_2_vals = [4.0, 14.0]
    eps_1, eps_3 = 1.0, 1.0

    rhos = np.logspace(-0.5, 1.5, 40)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    c = 2

    results = {}

    for row, d_total in enumerate(d_totals):
        for col, eps_2 in enumerate(eps_2_vals):
            ax = axes[row, col]
            ax2 = ax.twinx()

            d_half = d_total / 2.0
            epsilons = [eps_1, eps_2, eps_3]
            d_array = [-d_half, d_half]

            # (a) ETM
            V_etm = V_eh_direct(rhos, epsilons, d_array, c,
                                k_max=25.0, n_quad=300)

            # (b) Exact N=3
            eps_exact_func = lambda k, e1=eps_1, e2=eps_2, e3=eps_3, dt=d_total: \
                epsilon_N3_analytic(k, e1, e2, e3, dt)
            V_exact = V_from_epsilon_func(rhos, eps_exact_func, eps_inf=eps_2,
                                          k_max=25.0, n_quad=300)

            # (c) R-K
            V_rk = rytova_keldysh_potential(rhos, eps_1, eps_2, eps_3, d_total)

            # errors
            err_rk = np.where(V_rk > 1e-12,
                              np.abs(V_exact - V_rk) / V_rk * 100.0, 0.0)

            # -- potentials on left axis --
            ax.plot(rhos, V_etm, '-', color='crimson', linewidth=2,
                    label='ETM')
            ax.plot(rhos, V_exact, '--', color='forestgreen', linewidth=2,
                    label='Exact N=3')
            ax.plot(rhos, V_rk, ':', color='steelblue', linewidth=2.5,
                    label='R-K')
            ax.set_xscale('log')
            ax.set_xlabel(r'$\rho$ ($a_0$)')
            ax.set_ylabel(r'$V_{eh}$ (Ry)')
            ax.grid(True, linestyle='--', alpha=0.4)

            # -- error on right axis --
            ax2.semilogy(rhos, err_rk, 's-', color='darkorange',
                         markersize=3, linewidth=1, alpha=0.7,
                         label='|Exact − RK|/RK')
            ax2.set_ylabel('R-K relative error (%)', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            ax2.set_ylim([5e-3, 5e2])

            # combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
                      loc='upper right')

            rho0_val = eps_2 * d_total / (eps_1 + eps_3)
            ax.set_title(
                rf'$d_{{\rm total}}$ = {d_total} $a_0$,  '
                rf'$\varepsilon_2$ = {eps_2:.0f},  '
                rf'$\rho_0$ = {rho0_val:.2f} $a_0$' '\n'
                rf'max RK err = {np.max(err_rk):.2f}%',
                fontsize=10)

            results[(d_total, eps_2)] = {
                'rhos': rhos, 'V_etm': V_etm, 'V_exact': V_exact,
                'V_rk': V_rk, 'err_rk': err_rk}

    plt.tight_layout()
    plt.show()

    # --- print summary ---
    print("=" * 65)
    print("Test 1 -- Rytova-Keldysh: 2x2 parameter sweep")
    print(f"  eps1 = eps3 = 1.0")
    print(f"  {'d_total':>10s}  {'eps2':>6s}  {'rho0':>8s}  "
          f"{'max RK err':>12s}  {'mean RK err':>14s}")
    print("  " + "-" * 56)
    for (d_total, eps_2), res in results.items():
        rho0_val = eps_2 * d_total / (eps_1 + eps_3)
        print(f"  {d_total:10.2f}  {eps_2:6.1f}  {rho0_val:8.3f}  "
              f"{np.max(res['err_rk']):12.4f}  {np.mean(res['err_rk']):14.4f}")
    print("=" * 65)

    return results


def test_exact_N3():
    """
    Test 2 — Exact N = 3 analytic formula.

    Compares the ETM-computed ε(k) with the exact closed-form expression
    Eq. (14) of the PRB paper for the N = 3 case.  This validates the
    transfer-matrix algebra and the numerical propagation of the
    A, B coefficients.
    """
    eps_1, eps_2, eps_3 = 1.0, 14.0, 1.0
    d_half = 0.25
    d_total = 2.0 * d_half

    epsilons = [eps_1, eps_2, eps_3]
    d_array = [-d_half, d_half]
    c = 2

    # --- k-range covering the transition from ε(0) → ε(∞) ---
    k_vals = np.logspace(-3, 2, 200)          # 0.001 → 100

    # ETM ε(k) via transfer matrices
    eps_etm = np.array([compute_epsilon_k(epsilons, d_array, c, k)
                        for k in k_vals])

    # Exact N=3 analytic ε(k)
    eps_exact = epsilon_N3_analytic(k_vals, eps_1, eps_2, eps_3, d_total)

    # --- error ---
    abs_err = np.abs(eps_etm - eps_exact)
    rel_err = np.where(eps_exact > 1e-12,
                       abs_err / np.abs(eps_exact) * 100.0, 0.0)

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogx(k_vals, eps_etm, '-', color='crimson', linewidth=2,
                 label='ETM (this work)')
    ax1.semilogx(k_vals, eps_exact, '--', color='steelblue', linewidth=2,
                 label='Exact N=3, Eq. (14)')
    ax1.set_xlabel(r'$k$ ($a_0^{-1}$)')
    ax1.set_ylabel(r'$\varepsilon(k)$')
    ax1.set_title(rf'Dielectric function — ETM vs exact N=3 formula'
                  '\n'
                  rf'$\varepsilon$ = [{eps_1},{eps_2},{eps_3}],  '
                  rf'$d_{{\rm total}}$ = {d_total} $a_0$')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.semilogy(k_vals, rel_err, '-', color='darkred', linewidth=1.5)
    ax2.set_xlabel(r'$k$ ($a_0^{-1}$)')
    ax2.set_ylabel('Relative error (%)')
    ax2.set_title(r'|$\varepsilon_{\rm ETM} - \varepsilon_{\rm exact}$|'
                  r' / $\varepsilon_{\rm exact}$')
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

    # --- print summary ---
    print("\n" + "=" * 65)
    print("Test 2 -- Exact N=3 eps(k) comparison")
    print(f"  eps = [{eps_1}, {eps_2}, {eps_3}],  d_total = {d_total} a0")
    print(f"  eps(k->0) = {eps_exact[0]:.6f}")
    print(f"  eps(k->inf) = {eps_exact[-1]:.6f}")
    print(f"  max relative error: {np.max(rel_err):.6f} %")
    print(f"  mean relative error: {np.mean(rel_err):.6f} %")
    print("=" * 65)

    return k_vals, eps_etm, eps_exact, rel_err


def test_bilayer_eff_RK():
    """
    Test 3 — Bilayer effective R-K (from the moire-exciton paper).

    Compares three methods for an interlayer (indirect) exciton in a
    TMD bilayer where the two slabs touch at z = 0:

      (a) ETM — full transfer-matrix solution, N = 4
               ε = [ε_e, ε_t, ε_t, ε_e],  c = 2,  t = 3,  z_t = +d
      (b) Exact bilayer formula — Eq. (14) integrated numerically
      (c) Effective R-K — Eq. (15) with α = 1.8 (vacuum)

    Uses TMD parameters:  d = 6.15 Å ≈ 11.62 a₀,  ε_t = 14,  ε_e = 1.
    """
    # --- physical parameters (from the moire paper) ---
    eps_e = 1.0                     # vacuum
    eps_t = 14.0                    # TMD dielectric constant
    d_ang = 6.15                    # single-slab thickness in Angstrom
    d_a0 = d_ang / 0.529177         # convert to Bohr radii
    alpha_vac = 1.8                 # α for vacuum environment

    # --- ETM setup (N = 4) ---
    # Origin at centre of bottom slab → interfaces at
    #   z = −d/2  (env/tmd1),  z = +d/2  (tmd1/tmd2),  z = +3d/2  (tmd2/env)
    epsilons = [eps_e, eps_t, eps_t, eps_e]
    d_array = [-d_a0 / 2.0, d_a0 / 2.0, 3.0 * d_a0 / 2.0]
    c_src = 2                       # source in bottom slab
    t_obs = 3                       # observation in top slab
    z_t_obs = d_a0                  # vertical separation between centres

    # --- rho range ---
    rhos = np.logspace(-0.5, 2.0, 40)      # ~0.3 to ~100 a₀

    # Integration: for the indirect exciton ε(k) → ∞ as k → ∞, so
    # 1/ε(k) → 0 naturally.  No asymptotic subtraction needed.
    # ε(k) ∼ ε_t·e^{kd} for k ≳ 1/d, so at k_max ≈ 1.5 the integrand
    # ∼ J₀/ε ≈ 1 / (14·e^{1.5·11.6}) ≈ 10^{-9} — already negligible.
    k_max_bl = 1.5
    n_quad_bl = 400

    V_etm = V_eh(rhos, epsilons, d_array, c=c_src, t=t_obs, z_t=z_t_obs,
                 k_max=k_max_bl, n_quad=n_quad_bl)

    # (b) Exact bilayer — Eq. (14) integrated
    eps_exact_func = lambda k: epsilon_bilayer_exact(k, eps_e, eps_t, d_a0)
    V_exact = V_from_epsilon_func(rhos, eps_exact_func, eps_inf=np.inf,
                                  k_max=k_max_bl, n_quad=n_quad_bl)

    # (c) Effective R-K — Eq. (15)
    V_eff_rk = V_bilayer_eff_RK(rhos, eps_e, eps_t, d_a0, alpha=alpha_vac)

    # --- errors ---
    err_etm_vs_exact = np.where(V_exact > 1e-12,
                                np.abs(V_etm - V_exact) / V_exact * 100.0, 0.0)
    err_rk_vs_exact = np.where(V_exact > 1e-12,
                               np.abs(V_eff_rk - V_exact) / V_exact * 100.0, 0.0)

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(rhos, V_etm, 'o-', color='crimson', markersize=4,
             label='ETM (this work)')
    ax1.plot(rhos, V_exact, '-', color='forestgreen', linewidth=2,
             label='Exact bilayer, Eq. (14)')
    ax1.plot(rhos, V_eff_rk, 's--', color='steelblue', markersize=4,
             label=rf'Eff. R-K  ($\alpha$ = {alpha_vac})')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\rho$ ($a_0$)')
    ax1.set_ylabel(r'$|V_{eh}|$ (Ry)')
    ax1.set_title(
        r'Interlayer exciton potential — bilayer TMD in vacuum' '\n'
        rf'$\varepsilon_e$ = {eps_e},  $\varepsilon_t$ = {eps_t},  '
        rf'$d$ = {d_ang:.2f} $\AA$ = {d_a0:.2f} $a_0$')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.semilogy(rhos, err_etm_vs_exact, 'o-', color='darkgreen',
                 markersize=4, label='ETM vs Exact bilayer')
    ax2.semilogy(rhos, err_rk_vs_exact, 's--', color='darkorange',
                 markersize=4, label='Eff. R-K vs Exact bilayer')
    ax2.set_xlabel(r'$\rho$ ($a_0$)')
    ax2.set_ylabel('Relative error (%)')
    ax2.set_title('Relative errors')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim([1e-4, 1e2])

    plt.tight_layout()
    plt.show()

    # --- print summary ---
    rho_0 = alpha_vac * eps_t * d_a0 / (2.0 * eps_e)
    print("\n" + "=" * 65)
    print("Test 3 -- Bilayer effective R-K (moire paper)")
    print(f"  eps_e = {eps_e},  eps_t = {eps_t}")
    print(f"  d = {d_ang:.2f} Ang = {d_a0:.2f} a0")
    print(f"  alpha = {alpha_vac}  (vacuum)")
    print(f"  rho_0 = alpha * eps_t * d / (2 * eps_e) = {rho_0:.2f} a0")
    print(f"  --- ETM vs Exact bilayer ---")
    print(f"  max error:  {np.max(err_etm_vs_exact):.6f} %")
    print(f"  mean error: {np.mean(err_etm_vs_exact):.6f} %")
    print(f"  --- Effective R-K vs Exact bilayer ---")
    print(f"  max error:  {np.max(err_rk_vs_exact):.4f} %")
    print(f"  mean error: {np.mean(err_rk_vs_exact):.4f} %")
    print("=" * 65)

    return rhos, V_etm, V_exact, V_eff_rk, err_etm_vs_exact, err_rk_vs_exact


def test_homogeneous_limit():
    """
    Test 4 — Homogeneous-medium limit.

    When all three dielectric constants are equal (ε₁ = ε₂ = ε₃ = ε),
    the monolayer (N = 3) becomes a homogeneous infinite medium.  All
    interface reflections vanish and the potential must reduce to the
    bare 3D Coulomb form:

        V_Coulomb(ρ) = 2 / (ε ρ)                                    [in Ry]

    This is a strong test of the transfer-matrix cancellation: any
    residual interface artefact will show up as a deviation from 1/ρ.

    The test uses several ε values, each with a range of d_total, to
    confirm robustness.
    """
    eps_values = [2.0, 5.0, 10.0, 14.0]
    d_totals = [0.2, 1.0, 5.0]
    rhos = np.logspace(-1, 2, 30)

    fig, axes = plt.subplots(len(eps_values), len(d_totals),
                             figsize=(14, 12))
    c = 2

    print("=" * 70)
    print("Test 4 -- Homogeneous-medium limit")
    print("  All eps1 = eps2 = eps3 = eps  (N = 3)")
    print(f"  {'eps':>6s}  {'d_total':>8s}  {'max err %':>10s}  "
          f"{'mean err %':>10s}")
    print("  " + "-" * 44)

    for row, eps in enumerate(eps_values):
        for col, d_total in enumerate(d_totals):
            ax = axes[row, col]

            d_half = d_total / 2.0
            epsilons = [eps, eps, eps]
            d_array = [-d_half, d_half]

            # ETM
            V_etm = V_eh_direct(rhos, epsilons, d_array, c,
                                k_max=25.0, n_quad=300)

            # Exact Coulomb: V(ρ) = 2/(ε ρ)
            V_coulomb = 2.0 / (eps * rhos)

            # error
            rel_err = np.abs(V_etm - V_coulomb) / V_coulomb * 100.0

            # plot
            ax.loglog(rhos, V_etm, '-', color='crimson', linewidth=2,
                      label='ETM')
            ax.loglog(rhos, V_coulomb, '--', color='steelblue', linewidth=2,
                      label=r'$2/(\varepsilon\rho)$')
            ax.set_xlabel(r'$\rho$ ($a_0$)')
            ax.set_ylabel(r'$V_{eh}$ (Ry)')
            ax.set_title(
                rf'$\varepsilon$ = {eps},  '
                rf'$d_{{\rm total}}$ = {d_total} $a_0$' '\n'
                rf'max err = {np.max(rel_err):.4f}%',
                fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.4)

            print(f"  {eps:6.1f}  {d_total:8.2f}  {np.max(rel_err):10.4f}  "
                  f"{np.mean(rel_err):10.4f}")

    print("=" * 70)

    plt.tight_layout()
    plt.show()

    return


def convergence_analysis(
    epsilons=None,
    d_array=None,
    c_source=None,
    t_obs=None,
    z_t=None,
    rhos_test=None,
    k_max_values=None,
    reference_k_max=None,
    k_sweep_limit=None,
    limit_values=None,
    limit_sweep_k_max=None,
    epsabs=1e-10,
    epsrel=1e-8,
    show_plots=True,
    verbose=True,
):
    """
    Diagnose the convergence of the Fourier–Bessel integral with respect
    to the two tunable parameters:  *k_max* (upper integration cutoff)
    and *limit* (maximum `quad` subintervals, the surrogate for "dk").

    Produces a 2×2 figure and a printed table showing the error at each
    parameter choice, so the user can identify the threshold beyond which
    further refinement yields diminishing returns.

    Defaults target the bilayer system from `main()`:
       epsilons = [1, 14, 14, 1],  d = [-3, 3, 9],  c = 2  (direct exciton).
    """
    # ------------------------------------------------------------------
    # defaults — main() bilayer
    # ------------------------------------------------------------------
    if epsilons is None:
        epsilons = [1.0, 14.0, 14.0, 1.0]
    if d_array is None:
        d_array = [-3.0, 3.0, 9.0]
    if c_source is None:
        c_source = 2
    if t_obs is None:
        t_obs = c_source
        z_t = 0.0
    if z_t is None:
        z_t = 0.0
    if rhos_test is None:
        rhos_test = [0.1, 2.0, 10.0, 30.0]
    if k_max_values is None:
        k_max_values = np.logspace(-0.5, 1.3, 25)      # 0.316 → 20.0
    if reference_k_max is None:
        reference_k_max = 20.0
    if k_sweep_limit is None:
        k_sweep_limit = 400
    if limit_values is None:
        limit_values = [10, 20, 50, 100, 200, 400, 800]
    if limit_sweep_k_max is None:
        limit_sweep_k_max = 15.0

    rhos_test = np.atleast_1d(np.asarray(rhos_test, dtype=float))
    k_max_values = np.atleast_1d(np.asarray(k_max_values, dtype=float))
    limit_values = np.atleast_1d(np.asarray(limit_values, dtype=int))

    eps_c = epsilons[c_source - 1]
    is_direct = (t_obs == c_source and abs(z_t) < 1e-15)

    colors = ['crimson', 'steelblue', 'forestgreen', 'darkorange']

    # ------------------------------------------------------------------
    # reference — most conservative parameters
    # ------------------------------------------------------------------
    ref_kw = dict(k_max=reference_k_max, n_quad=max(limit_values))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if is_direct:
            V_ref = V_eh_direct(rhos_test, epsilons, d_array, c_source,
                                **ref_kw)
        else:
            V_ref = V_eh(rhos_test, epsilons, d_array, c_source,
                         t=t_obs, z_t=z_t, **ref_kw)

    # self-consistency: double-check reference
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        V_ref2 = (V_eh_direct(rhos_test, epsilons, d_array, c_source,
                              k_max=reference_k_max * 0.75,
                              n_quad=max(limit_values))
                  if is_direct else
                  V_eh(rhos_test, epsilons, d_array, c_source,
                       t=t_obs, z_t=z_t,
                       k_max=reference_k_max * 0.75,
                       n_quad=max(limit_values)))
    ref_drift = np.max(np.abs(V_ref - V_ref2) / np.where(V_ref > 1e-12,
                                                         V_ref, 1e-12)) * 100
    if verbose and ref_drift > 0.01:
        print(f"  [warning] reference drift at 2× k_max: {ref_drift:.4f} %")

    # ------------------------------------------------------------------
    # k_max sweep
    # ------------------------------------------------------------------
    err_k = np.zeros((len(rhos_test), len(k_max_values)))
    for j, km in enumerate(k_max_values):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if is_direct:
                V_k = V_eh_direct(rhos_test, epsilons, d_array, c_source,
                                  k_max=km, n_quad=k_sweep_limit,
                                  )
            else:
                V_k = V_eh(rhos_test, epsilons, d_array, c_source,
                           t=t_obs, z_t=z_t, k_max=km,
                           n_quad=k_sweep_limit)

        denom = np.where(np.abs(V_ref) > 1e-12, np.abs(V_ref), 1e-12)
        err_k[:, j] = np.abs(V_k - V_ref) / denom * 100.0

    # ------------------------------------------------------------------
    # limit sweep
    # ------------------------------------------------------------------
    err_lim = np.zeros((len(rhos_test), len(limit_values)))
    for j, lim in enumerate(limit_values):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if is_direct:
                V_lim = V_eh_direct(rhos_test, epsilons, d_array, c_source,
                                    k_max=limit_sweep_k_max, n_quad=int(lim),
                                    )
            else:
                V_lim = V_eh(rhos_test, epsilons, d_array, c_source,
                             t=t_obs, z_t=z_t, k_max=limit_sweep_k_max,
                             n_quad=int(lim))

        denom = np.where(np.abs(V_ref) > 1e-12, np.abs(V_ref), 1e-12)
        err_lim[:, j] = np.abs(V_lim - V_ref) / denom * 100.0

    # ------------------------------------------------------------------
    # integrand diagnostic — dense k-grid for smallest & largest rho
    # ------------------------------------------------------------------
    k_diag = np.linspace(0.05, reference_k_max, 600)
    integrand_profiles = {}
    for i_rho in [0, -1]:               # smallest and largest rho
        rho = rhos_test[i_rho]
        vals = np.array([
            _integrand_subtracted(k, epsilons, d_array, c_source, rho,
                                  t_obs, z_t)
            for k in k_diag
        ])
        integrand_profiles[rho] = vals

    # cumulative subtracted integral I(k) = ∫₀^k (1/ε−1/ε_c)·J₀ dk'
    cum_I = {}
    for i_rho, rho in enumerate(rhos_test):
        cum = np.zeros(len(k_diag))
        running = 0.0
        for m in range(1, len(k_diag)):
            kmid = 0.5 * (k_diag[m - 1] + k_diag[m])
            val_mid = _integrand_subtracted(kmid, epsilons, d_array,
                                            c_source, rho, t_obs, z_t)
            running += val_mid * (k_diag[m] - k_diag[m - 1])
            cum[m] = running
        cum_I[rho] = cum

    # ------------------------------------------------------------------
    # plot — 2×2 figure
    # ------------------------------------------------------------------
    if show_plots:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

        # --- panel 1: error vs k_max ---
        for i_rho, rho in enumerate(rhos_test):
            ax1.loglog(k_max_values, np.maximum(err_k[i_rho, :], 1e-12),
                       'o-', color=colors[i_rho], markersize=3,
                       label=rf'$\rho$ = {rho:.2f} $a_0$')
        ax1.axhline(0.1, color='gray', linestyle=':', linewidth=1)
        ax1.axhline(0.01, color='gray', linestyle=':', linewidth=0.7)
        ax1.axvline(15.0, color='black', linestyle='--', linewidth=0.8,
                    label='default k_max=15')
        ax1.set_xlabel(r'$k_{\rm max}$ ($a_0^{-1}$)')
        ax1.set_ylabel('Relative error (%)')
        ax1.set_title('Convergence with $k_{\\rm max}$')
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.3)

        # --- panel 2: error vs limit ---
        for i_rho, rho in enumerate(rhos_test):
            ax2.semilogy(limit_values, np.maximum(err_lim[i_rho, :], 1e-12),
                         's-', color=colors[i_rho], markersize=5,
                         label=rf'$\rho$ = {rho:.2f} $a_0$')
        ax2.axhline(0.1, color='gray', linestyle=':', linewidth=1)
        ax2.axhline(0.01, color='gray', linestyle=':', linewidth=0.7)
        ax2.axvline(200, color='black', linestyle='--', linewidth=0.8,
                    label='default limit=200')
        ax2.set_xlabel('limit (max subintervals)')
        ax2.set_ylabel('Relative error (%)')
        ax2.set_title('Convergence with quadrature limit')
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.3)

        # --- panel 3: |integrand(k)| for smallest & largest rho ---
        for i_rho in [0, -1]:
            rho = rhos_test[i_rho]
            ax3.semilogy(k_diag, np.abs(integrand_profiles[rho]),
                         '-', color=colors[i_rho if i_rho == 0 else -1],
                         linewidth=1.5,
                         label=rf'$\rho$ = {rho:.2f} $a_0$')
        ax3.axvline(15.0, color='black', linestyle='--', linewidth=0.8)
        ax3.axhline(np.finfo(float).eps, color='gray', linestyle=':',
                    linewidth=0.7, label='float64 eps')
        ax3.set_xlabel(r'$k$ ($a_0^{-1}$)')
        ax3.set_ylabel(r'$|(1/\varepsilon - 1/\varepsilon_c)\,J_0(k\rho)|$')
        ax3.set_title('Subtracted integrand decay')
        ax3.legend(fontsize=8)
        ax3.grid(True, linestyle='--', alpha=0.3)

        # --- panel 4: cumulative subtracted integral I(k) ---
        for i_rho, rho in enumerate(rhos_test):
            ax4.plot(k_diag, cum_I[rho], '-', color=colors[i_rho],
                     linewidth=1.5 if i_rho in (0, -1) else 1.0,
                     alpha=1.0 if i_rho in (0, -1) else 0.5,
                     label=rf'$\rho$ = {rho:.2f} $a_0$')
        ax4.axvline(15.0, color='black', linestyle='--', linewidth=0.8)
        ax4.set_xlabel(r'$k$ ($a_0^{-1}$)')
        ax4.set_ylabel('Cumulative integral  I(k)')
        ax4.set_title('Convergence of subtracted integral')
        ax4.legend(fontsize=8)
        ax4.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # printed summary
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 68)
        print("Convergence Analysis")
        print(f"  epsilons = {epsilons}")
        print(f"  d        = {d_array}  a0")
        print(f"  c = {c_source}  (source layer)")
        if t_obs != c_source:
            print(f"  t = {t_obs},  z_t = {z_t}  (indirect exciton)")
        else:
            print(f"  t = {c_source}  (direct exciton)")
        print(f"  reference: k_max = {reference_k_max},  limit = "
              f"{max(limit_values)}")
        if ref_drift > 0.01:
            print(f"  reference self-consistency drift: {ref_drift:.4f} %")

        def _threshold(k_or_lim, err_mat):
            """Find first index where all rho errors drop below target."""
            thresh = {}
            for target in [0.1, 0.01]:
                idx = np.where(np.max(err_mat, axis=0) < target)[0]
                thresh[target] = k_or_lim[idx[0]] if len(idx) > 0 else np.inf
            return thresh

        th_k = _threshold(k_max_values, err_k)
        th_lim = _threshold(limit_values.astype(float), err_lim)

        print(f"\n  -- k_max convergence --")
        hdr = f"  {'rho':>8s}  {'err at k=15':>12s}  "
        hdr += f"{'k @ 0.1%':>10s}  {'k @ 0.01%':>10s}"
        print(hdr)
        print("  " + "-" * 50)
        for i_rho, rho in enumerate(rhos_test):
            idx15 = np.argmin(np.abs(k_max_values - 15.0))
            print(f"  {rho:8.3f}  {err_k[i_rho, idx15]:12.4e}  "
                  f"{_first_below(k_max_values, err_k[i_rho, :], 0.1):10.3f}  "
                  f"{_first_below(k_max_values, err_k[i_rho, :], 0.01):10.3f}")

        print(f"\n  -- limit convergence --")
        hdr2 = f"  {'rho':>8s}  {'err at lim=200':>14s}  "
        hdr2 += f"{'lim @ 0.1%':>12s}  {'lim @ 0.01%':>14s}"
        print(hdr2)
        print("  " + "-" * 55)
        for i_rho, rho in enumerate(rhos_test):
            idx200 = np.argmin(np.abs(limit_values.astype(float) - 200.0))
            print(f"  {rho:8.3f}  {err_lim[i_rho, idx200]:14.4e}  "
                  f"{_first_below(limit_values.astype(float), err_lim[i_rho, :], 0.1):12.1f}  "
                  f"{_first_below(limit_values.astype(float), err_lim[i_rho, :], 0.01):14.1f}")

        print(f"\n  -- recommendation (worst-case rho) --")
        print(f"  k_max >= {th_k[0.1]:.2f}  for 0.1 % accuracy")
        print(f"  k_max >= {th_k[0.01]:.2f}  for 0.01 % accuracy")
        print(f"  limit >= {th_lim[0.1]:.0f}  for 0.1 % accuracy")
        print(f"  limit >= {th_lim[0.01]:.0f}  for 0.01 % accuracy")
        print("=" * 68)


def _first_below(x_vals, err_vals, target):
    """First x where error < target %, or NaN."""
    idx = np.where(err_vals < target)[0]
    return float(x_vals[idx[0]]) if len(idx) > 0 else np.nan


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'test':
            print("Running all validation tests...\n")
            test_rytova_keldysh()
            print("\n")
            test_exact_N3()
            print("\n")
            test_bilayer_eff_RK()
            print("\n")
            test_homogeneous_limit()
        elif arg == 'test1':
            test_rytova_keldysh()
        elif arg == 'test2':
            test_exact_N3()
        elif arg == 'test3':
            test_bilayer_eff_RK()
        elif arg == 'test4':
            test_homogeneous_limit()
        elif arg in ('convergence', 'conv'):
            convergence_analysis()
        elif arg in ('timing', 'time'):
            tolerance_timing()
        elif arg in ('fit', 'fits'):
            fit_benchmark()
        elif arg in ('bulk', 'bulk_fit'):
            bulk_fit_benchmark()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python main.py "
                  "[test | test1 | test2 | test3 | test4 | convergence]")
    else:
        main()