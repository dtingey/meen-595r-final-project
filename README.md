# meen-595r-final-project
Final Project for Decision Making Under Uncertainty Class

This repository is an implementation of potential game and binary log-linear learning for multi-UAV cooperative search and surveillance. It is based on the paper:

> **Li, P. & Duan, H. (2017)**  
> *A potential game approach to multiple UAV cooperative search and surveillance*  
> Aerospace Science and Technology 68, 403–415

The main entrypoint is a single self-contained script:

- `sim.py`

It uses **uv** script dependency management and generates figures, GIFs, and HTML reports.

---

## 1) Prerequisites

### Required
- [uv](https://docs.astral.sh/uv/) installed

### Notes
- `sim.py` declares its dependencies inline (NumPy, Matplotlib, Pillow), so you do **not** need to manually create a venv or `pip install` packages.
- Script metadata currently requires Python `>=3.13`.

---

## 2) Quick start

From this directory:

```bash
uv run sim.py
```

This will:
1. Run all simulation scenarios
2. Save images/GIFs in `outputs/`
3. Save HTML report(s) in `reports/`

---

## 3) Common run modes

### Faster smoke test
```bash
uv run sim.py --quick
```

### Enable presentation report (in addition to normal report)
```bash
uv run sim.py --presentation-mode
```

### Custom seed/output locations
```bash
uv run sim.py --seed 123 --output-dir outputs_custom --report-dir reports_custom
```

---

## 4) CLI options

```bash
uv run sim.py --help
```

Current options:
- `--quick`
- `--presentation-mode` / `--no-presentation-mode`
- `--seed`
- `--output-dir`
- `--report-dir`

---

## 5) Relevant outputs

### Main reproduced figures (saved in `outputs/`)
- `fig4_homogeneous_with_failures.png`  
  Homogeneous coverage + failure reconfiguration
- `fig5_heterogeneous_coverage.png`  
  Heterogeneous sensing coverage
- `fig6_obstacle_coverage.png`  
  Coverage with obstacles
- `fig7_8_learning_comparison.png`  
  BLLL vs Best Response comparison
- `fig9_probability_snapshots.png`  
  Search probability-map snapshots (early-to-mid transient)
- `fig10_uncertainty_parameter_study.png`  
  Parameter sensitivity on uncertainty convergence

### Movement visualizations
- `homogeneous_trajectories.png`, `homogeneous_movement.gif`
- `obstacle_trajectories.png`, `obstacle_movement.gif`
- `search_trajectories.png`, `search_movement.gif`

### Additional diagnostics (generated in some workflows)
- `long_run_convergence_2500.png`
- `long_run_convergence_2500_summary.json`
- `long_run_convergence_5000_summary.json`
- `one_run_10000_homogeneous.png`
- `one_run_10000_homogeneous_summary.json`
- `one_run_10000_homogeneous_step_diagnostics.json`
- `paper_step_semantics_2500_epochs_compare.png`
- `paper_step_semantics_2500_epochs_summary.json`

---

## 6) Reports

Reports are saved in `reports/` with timestamped names.

### Normal report
Pattern:
- `simulation_report_<mode>_<timestamp>.html`

### Presentation report
Pattern:
- `simulation_presentation_<mode>_<timestamp>.html`

The presentation report uses large slide-style visual sections for easier presenting.

---
