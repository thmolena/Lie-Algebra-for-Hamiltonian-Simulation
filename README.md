# Lie GPT for Hamiltonian Simulation

## Research Summary

This repository presents Lie GPT as a Hamiltonian-simulation and constrained quantum-dynamics framework with benchmarked gains in four areas documented by the paper and notebook studies:

- lower spectral error in the transverse-field Ising model benchmark,
- exact unitarity at machine precision,
- stable long-horizon rollout in learned dynamics,
- improved data efficiency in trajectory-based training.

The repository materials are organized as a technical report rather than a product page. The emphasis is therefore on documented findings, the meaning of those findings for the tested applications, and the files needed to reproduce the reported results.

## Verified Findings

### Hamiltonian simulation accuracy

For the four-qubit transverse-field Ising model, the paper reports a verified 10× improvement regime over same-order product-formula baselines. In the reported evaluation setting at total time $T = 0.5$:

- Lie GPT-1 achieves 12.3× lower spectral error than Trotter-1 at $N = 10$ steps.
- Lie GPT-2 achieves 10.7× lower spectral error than Suzuki-2 at $N = 24$ steps.

These results indicate that the Lie GPT propagators track the target unitary more closely in the benchmarked digital simulation setting.

### Long-horizon stability

In the learned dynamics benchmark, training is performed on sequences of length $T = 25$ and evaluation is extended to rollout length $T = 200$. The repository summary and rendered outputs report bounded Lie GPT error over this 8× extrapolation range, consistent with the stated linear-in-time stability behavior.

### Exact unitarity by construction

The architecture notebook constrains the learned generator to the Lie algebra basis before exponentiation. The reported result is unitarity violation at approximately machine precision, about $10^{-16}$, in the architecture and unitarity study.

### Data efficiency

The data-efficiency notebook reports that the constrained model reaches a target accuracy with about 3× fewer trajectories. In this repository, that result is used as evidence that physically structured generator prediction reduces training demand in the learned quantum-dynamics application.

## Practical Meaning for the Project Applications

### Static spin-system simulation

In the transverse-field Ising model application, lower spectral error means that the simulated unitary remains closer to the target physical evolution over the same total time. In practical terms, this supports more reliable studies of spin-chain dynamics, phase-sensitive observables, and quantum-algorithm test cases under the tested circuit budgets.

### Learned quantum-dynamics rollout

In the rollout application, exact unitarity and stable long-horizon prediction mean that the model propagates states without leaving the physically valid state space. For repeated-step simulation, forecasting, and control-oriented studies, this gives trajectories that remain usable farther beyond the training horizon.

### Data-limited training settings

In the data-efficiency study, fewer trajectories for the same accuracy target translate directly to lower data-generation cost. That matters when training data are produced by simulation pipelines, calibration routines, or hardware measurement campaigns.

### Noise-aware evaluation

In the robustness study, maintaining machine-precision unitarity under noisy inputs means the model preserves the core physical constraint even when the input signal is perturbed. That is important when the predicted evolution operator must remain physically interpretable.

## Evidence Table

| Benchmark | Verified gain | Practical interpretation |
|---|---|---|
| TFIM spectral-error benchmark | Lie GPT-1: 12.3× lower error than Trotter-1 at $N = 10$ | More accurate digital simulation of the same spin model at the tested step budget |
| Higher-order TFIM benchmark | Lie GPT-2: 10.7× lower error than Suzuki-2 at $N = 24$ | Better accuracy at fixed total evolution time in the higher-order propagation setting |
| Architecture and unitarity study | Unitarity violation remains at machine precision | Predicted quantum evolution stays physically valid for simulation and control tasks |
| Data-efficiency study | About 3× fewer trajectories for the same accuracy target | Lower experimental or synthetic-data burden during training |

## Selected Figures

- `outputs/unitarity_benchmark.png` documents the machine-precision unitarity result.
- `outputs/stability_rollout.png` summarizes the long-horizon rollout behavior.
- `outputs/data_efficiency.png` documents the reported training-efficiency gain.

## Repository Structure

```text
research_paper/
  main.tex                    LaTeX source for the paper
  research_paper.pdf          Compiled paper
notebooks/
  liegpt_architecture_unitarity.ipynb
  liegpt_architecture_unitarity.html
  liegpt_stability.ipynb
  liegpt_stability.html
  liegpt_efficiency_robustness.ipynb
  liegpt_efficiency_robustness.html
outputs/
  unitarity_benchmark.png
  stability_rollout.png
  data_efficiency.png
results/
  tfim_baseline_results.csv
  tfim_sweep_results.csv
  static_selected_results.csv
  driven_selected_results.csv
scripts/
  tfim_experiment.py
  tfim_sweep.py
  generate_liegpt_figures.py
src/
  lc_qaoa/
index.html                   Root project homepage
website/index.html           Secondary homepage using the same report style
requirements.txt
```

## Reproducing the Reported Results

All experiments in the repository are configured to run on CPU.

### Paper benchmarks

```bash
conda activate qaoa
export PYTHONPATH=src

python scripts/tfim_experiment.py
python scripts/tfim_sweep.py
python scripts/generate_liegpt_figures.py
```

### Notebook studies

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_architecture_unitarity.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_stability.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_efficiency_robustness.ipynb

jupyter nbconvert --to html notebooks/liegpt_architecture_unitarity.ipynb
jupyter nbconvert --to html notebooks/liegpt_stability.ipynb
jupyter nbconvert --to html notebooks/liegpt_efficiency_robustness.ipynb
```

### Local page preview

```bash
python -m http.server 8000
```

Then open the root homepage or the report-style page under `website/` in a browser.

## Primary Materials

- [Research paper](research_paper/research_paper.pdf)
- [Architecture and unitarity notebook](notebooks/liegpt_architecture_unitarity.html)
- [Stability notebook](notebooks/liegpt_stability.html)
- [Efficiency and robustness notebook](notebooks/liegpt_efficiency_robustness.html)
- [Root landing page](index.html)
- [Website landing page](website/index.html)