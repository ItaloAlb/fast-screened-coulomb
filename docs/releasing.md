<!--
SPDX-FileCopyrightText: 2026 The Fast Screened Coulomb Authors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Releasing Fast Screened Coulomb

How Fast Screened Coulomb is versioned and how a version is released. This
page is the reference for maintainers and for anyone preparing a release.

## Versioning

Fast Screened Coulomb follows [Semantic Versioning 2.0.0](https://semver.org).
Its public interface is:

- every function, type, enumeration value, structure field and macro declared
  in `src/fast_screened_coulomb.h`;
- the units of every input and output (energies in Rydberg, distances in Bohr
  radii);
- the documented numerical contract of each function: what it computes, its
  accuracy targets and the inputs it rejects.

The file that defines it, and that a release compares against the previous
tag, is `src/fast_screened_coulomb.h`, together with the API reference in
`README.md`.

| Change | From 1.0.0 | While 0.x |
|---|---|---|
| A declaration is removed or changes signature, units or meaning; a documented result changes for valid input | MAJOR | MINOR |
| A compatible addition (new function, method, field at the end of an opaque type) | MINOR | MINOR |
| A fix with no interface change | PATCH | PATCH |

A fix that changes the numbers returned for valid input is a PATCH when the old
numbers were wrong; the changelog says which results change. A changed default
(an automatic parameter choice, a fit selection rule) that alters results is at
least a MINOR change.

A commit that breaks the interface says so in its body:
`BREAKING: <what breaks and how to migrate>`.

## Where the version lives

The single source is the set of macros `FSC_VERSION_MAJOR`,
`FSC_VERSION_MINOR`, `FSC_VERSION_PATCH` and `FSC_VERSION` in
`src/fast_screened_coulomb.h`. A release also updates `version` and
`date-released` in `CITATION.cff` and adds the entry to `CHANGELOG.md`.

## Release procedure

1. `main` is clean, in sync with `origin`, and `make test` passes.
2. Compare the previous tag with `HEAD`: `BREAKING:` lines in the commit
   bodies and the diff of `src/fast_screened_coulomb.h`. Choose the new
   version from the table.
3. Write the changelog entry for users (Added, Changed, Deprecated, Removed,
   Fixed, Security).
4. Update the version macros, `CITATION.cff` and `CHANGELOG.md` in one commit,
   `Release vX.Y.Z`.
5. Create the annotated tag `vX.Y.Z` with the changelog entry as its message,
   and push the commit and the tag.
6. Publish a GitHub release from the tag.

## Readiness checks

Run before every release:

- `make clean && make test`: `test_fsc` reports every check passed and
  `test_fsc_controlled` exits with status 0.
- `make bench` builds and runs to completion.
- `reuse lint` reports compliance.
- The examples in `README.md` still match the declarations in the header.
