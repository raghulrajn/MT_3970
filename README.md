# [Thesis Title]

**Author:** [Student Name]
**Student ID:** [ID]
**Thesis Type:** [Master Thesis / Bachelor Thesis / Research Project]
**Thesis Number:** [e.g., MT_2024_042]
**Supervisor:** [Supervisor Name]
**Date:** [Month Year]

---

## Abstract

[Brief summary of the thesis (150-300 words). Describe the problem, approach, and key findings.]

---

## Introduction

[Describe the problem domain, motivation, and objectives of this work.]

### Research Questions

1. [Primary research question]
2. [Secondary research question(s)]

### Contributions

- [Main contribution 1]
- [Main contribution 2]

---

## Methods

### Model Architecture

[Describe your model architecture and design decisions.]

### Training Setup

[Describe training configuration, hyperparameters, and optimization strategy.]

---

## Results

### Quantitative Results

[Present metrics and performance comparisons.]

| Dataset | MSE | RMSE | MAE | R² |
|---------|-----|------|-----|-----|
| DDACS   |     |      |     |     |
| HSH     |     |      |     |     |

### Qualitative Analysis

[Discuss observations, visualizations, and insights.]

---

## Conclusion

[Summarize findings and answer research questions.]

### Future Work

[Potential improvements and extensions.]

---

## Repository Structure

```
├── src/models/          # Model implementations
├── configs/model/       # Model configurations
├── notebooks/           # Analysis notebooks
├── experiments/         # Training outputs
└── docs/                # Documentation
```

---

## Getting Started

```bash
./sandbox.sh start && ./sandbox.sh shell   # your personal GPU container
```

See [SANDBOX.md](SANDBOX.md) for the sandbox. The full documentation lives
under `docs/` - read it in your browser with:

```bash
uv run make -C docs livehtml    # serves on port 7000
```

then open http://localhost:7000 (VS Code forwards the port automatically;
otherwise `ssh -L 7000:localhost:7000 <you>@<server>`). One-shot build:
`uv run make -C docs html` → `docs/_build/html/index.html`.

When you are done with the [Getting Started](docs/source/guides/getting_started.md)
guide, send your supervisor the run ID of your `fast_debug` training.

---

## References

[List key references]
