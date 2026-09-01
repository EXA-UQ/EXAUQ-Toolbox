# Changelog

All notable changes to the EXAUQ-Toolbox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries for releases prior to this file's introduction are condensed from the
[GitHub release notes](https://github.com/EXA-UQ/EXAUQ-Toolbox/releases).

## [Unreleased]

### Security

- Hardened supply-chain defenses: 7-day cooldown on Dependabot version
  updates, all GitHub Actions pinned by commit SHA, read-only workflow
  tokens, pip-audit scans on pull requests and weekly, dependency review on
  pull requests, a gitleaks secret-scanning pre-commit hook, and a
  `SECURITY.md` policy with private vulnerability reporting.

- Updated `black` to 26.x, fixing a high-severity advisory (arbitrary file
  writes via unsanitized cache file names) that the previous `^24.10.0` pin
  blocked; codebase reformatted to the black 26 stable style.

## [0.3.3] - 2026-09-01

### Security

- Updated all dependencies to their latest compatible versions, including
  security fixes for `cryptography`, `urllib3`, `requests`, `tornado`,
  `setuptools`, `h11` and `pynacl`.

### Changed

- Pinned `numpy` (<2.3) and `scipy` (<1.16) to the last releases supporting
  Python 3.10.
- CI test and linting workflows now run automatically on pull requests instead
  of being gated on a PR approval.
- Dependency updates are consolidated under Dependabot (grouped weekly updates
  targeting `dev`), replacing the scheduled poetry-lock workflow.
- Updated deprecated GitHub Actions (`checkout` v7, `setup-python` v7,
  `cache` v4).

### Fixed

- Broken link to Harrison White in the README.

## [0.3.2] - 2025-04-01

### Added

- Citation support via `CITATION.cff`, enabling GitHub's citation UI, with a
  DOI ([10.5281/zenodo.15005642](https://doi.org/10.5281/zenodo.15005642)).
- Project badges in the README (tests, Python versions, documentation,
  license, DOI, code style).

### Changed

- Refined platform markers and extras in `poetry.lock` for cleaner environment
  resolution.

## [0.3.1] - 2025-03-28

### Added

- `NOTICE` file attributing MIT-licensed code from the Alan Turing Institute.

### Changed

- Replaced `LICENSE.txt` with a standard `LICENSE` file (BSD 3-Clause).
- Improved the warning issued when Gaussian Process fitting is attempted with
  insufficient training data.
- Minor dependency updates.

## [0.3.0] - 2025-03-27

### Added

- Multi-level adaptive sampling utilities:
  `create_data_for_multi_level_loo_sampling` (training data preparation with
  input deduplication and inter-level delta calculation) and
  `compute_delta_coefficients` (delta coefficients from Markov-style
  correlations).
- Simulation log enhancements: `get_simulations` now includes the simulation
  level, and `prepare_training_data` converts logs to
  `MultiLevel[TrainingDatum]` for direct use in model fitting.

### Changed

- Improved type checking and input validation; warnings when logs are empty or
  incomplete.

## [0.2.1] - 2025-03-14

### Added

- Level-wise predictions for multi-level Gaussian Process emulators.

### Changed

- Updated documentation launcher and contributing guide for GitHub Pages
  deployment.

## [0.2.0] - 2025-03-11

### Added

- Multi-interface initialisation and management in the CLI.
- `-v` / `--version` argument for the `exauq` command.
- `Input.sequence_from_array` class method.

### Fixed

- Empty input handling in `TrainingDatum`.

### Changed

- General code and documentation tidy-up.

## [0.1.1] - 2024-12-10

### Added

- GitHub Pages documentation deployment via MkDocs workflow.

### Changed

- Updated repository information post-release.

## [0.1.0] - 2024-12-06

First public release of the EXAUQ-Toolbox: Gaussian Process emulation for
complex computer simulations (single- and multi-level), experimental design
tools, simulation job management across distributed hardware, and an
interactive CLI.

[Unreleased]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.3.3...dev
[0.3.3]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/EXA-UQ/EXAUQ-Toolbox/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EXA-UQ/EXAUQ-Toolbox/releases/tag/v0.1.0
