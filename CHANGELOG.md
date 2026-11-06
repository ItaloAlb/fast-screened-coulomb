<!--
SPDX-FileCopyrightText: 2026 The Fast Screened Coulomb Authors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to Fast Screened Coulomb are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
[docs/releasing.md](docs/releasing.md) defines what counts as its public
interface.

## [0.1.0] - 2026-11-06

First versioned release, under the GNU General Public License v3.0 or later.

### Added

- Effective dielectric function ε(k) of an arbitrary stack of dielectric
  slabs from the electrostatic transfer-matrix method (`fsc_epsilon_k`).
- Exact screened potential V(ρ) by adaptive Gauss–Kronrod integration of the
  Fourier–Bessel integral (`fsc_potential`, `fsc_potential_array`), for
  carriers in the same layer (direct) or in different layers (indirect).
- Automatic choice of the integration cutoff, quadrature size and radial
  range from accuracy tolerances (`fsc_optimal_params`).
- Fast fitted representations of V(ρ): Chebyshev, Legendre, cubic B-spline,
  Padé and image charges, and `fsc_fit_auto`, which picks the smallest
  Chebyshev degree meeting a target error.
- Exact first derivatives of every fitted representation (`fsc_fit_d1`,
  `fsc_fit_d1_array`).
- Controlled value and force fitting with separate accuracy targets and
  reported discrepancies (`fsc_fit_controlled`, `fsc_reference`).
- Benchmark of all fit methods (`fsc_benchmark`, `make bench`) and the test
  suites run by `make test`.
- Version macros `FSC_VERSION_MAJOR`, `FSC_VERSION_MINOR`,
  `FSC_VERSION_PATCH` and `FSC_VERSION` in the public header.

[0.1.0]: https://github.com/ItaloAlb/fast-screened-coulomb/releases/tag/v0.1.0
