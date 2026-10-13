# Classical–Quantum Interface Design for Multi-Class Quantum Convolutional Neural Networks

**English** | [**简体中文**](README.zh-CN.md)

Experimental code and configuration files for the paper **“Classical–Quantum Interface Design for Multi-Class Quantum Convolutional Neural Networks”**.

This repository implements a progressive experimental framework that studies the **classical–quantum interface** of a multi-class **quantum convolutional neural network (QCNN)** under a setting with **only minimal classical assistance**. The pipeline consists of data dimensionality reduction and normalization, an angle-encoding circuit, a parameterized entangling Ansatz, single-qubit Pauli-*Z* measurements, and a classical fully-connected readout. All quantum circuits are simulated with **MindQuantum** (`mqvector` state-vector simulator) and trained with **MindSpore**.

---

## Contents

```
.
├── Experiment I Influence of different Ansatz structures and layer numbers/
│   ├── Influence of different Ansatz structures and layer numbers.ipynb
│   └── results_per_exp/                     # 9 × (_history.pkl + _model.ckpt), L = 1, 2, 3
├── Experiment II Influence of data dimensionality reduction methods/
│   ├── UMAP.ipynb
│   ├── autoencoder.ipynb
│   └── *.npz                                # UMAP / autoencoder results (PCA is reused from Experiment I)
├── Experiment III Effect of the pre-encoding scaling interface/
│   ├── PCA.ipynb   UMAP.ipynb   autoencoder.ipynb
│   └── *_scaling_results*.npz
├── Experiment IV Comparison of post-measurement weighting layers/
│   ├── PCA.ipynb   UMAP.ipynb   autoencoder.ipynb
│   └── *_weight_relu_results*.npz
├── Ablation study necessity of the quantum circuit/
│   ├── {PCA, UMAP, autoencoder} × {frozen ansatz, encoder only, MLP, SVM}  → 12 notebooks
│   └── 12 × *.npz
└── Dimensionality reduction on CIFAR-10/
    ├── {PCA, UMAP, autoencoder}_cifar-10_{2,10}.ipynb     → 6 notebooks
    └── cifar10_{binary,10class}_*_results.npz             → 6 result files
```

**Totals:** 27 notebooks, 26 `.npz` result archives, 9 training histories (`.pkl`), 9 model checkpoints (`.ckpt`).

---

## Notebook ↔ paper mapping

| Folder | Paper section | Main artefacts |
|---|---|---|
| Experiment I | §4.1 Experiment I | Figure 4; gate-count/depth analysis |
| Experiment II | §4.2 Experiment II | Table 3 |
| Experiment III | §4.3 Experiment III | Table 4, Figure 5 |
| Experiment IV | §4.4 Experiment IV | Table 5, Figure 6 |
| Ablation study | §4.5 Ablation study | Table 6, Figure 7 |
| Dimensionality reduction on CIFAR-10 | §4.6 Dimensionality reduction on CIFAR-10 | Table 7, Figure 8 |

---

## Environment

- Python 3.9+
- **MindSpore** and **MindQuantum** (CPU build is sufficient)
- `numpy`, `scipy`, `scikit-learn`, `matplotlib`
- `umap-learn` (for the UMAP experiments)
- `torchvision` (used only to download MNIST / CIFAR-10; the ablation notebooks download MNIST directly via `urllib`)

```bash
pip install mindspore mindquantum numpy scikit-learn matplotlib umap-learn torchvision
```

All experiments were executed on **CPU only** (`MS_DEVICE = "CPU"`); no GPU is required. The simulator used throughout is MindQuantum’s `mqvector`.

---

## Data

- **MNIST** — downloaded automatically by the notebooks; partitioned into training / test sets at a **8:2** ratio (56 000 / 14 000).
- **CIFAR-10** — downloaded automatically through `torchvision.datasets.CIFAR10`; two tasks are evaluated:
  - *binary*: 640 training / 100 test images **per class** for airplane vs. automobile → 1 280 / 200;
  - *ten-class*: 640 / 100 per class over all ten classes → 6 400 / 1 000.

Both datasets are flattened (`784` for MNIST, `3072` for CIFAR-10), scaled to `[0, 1]`, reduced to **8 dimensions**, and finally rescaled to `[-1, 1]` before angle encoding. All transformation parameters are fitted on the training set only.

---

## Usage

Open the notebook of interest and run all cells. Each notebook is self-contained: it loads the data, builds the circuit, trains the model, prints the metrics, and writes its results to an `.npz` archive in the same folder.

```bash
jupyter notebook
```

Typical runtime is dominated by the quantum simulation; the CIFAR-10 and ablation notebooks are the lightest, while the Experiment III / IV notebooks (three preprocessing methods × three interface variants) take the longest.

---

## Key configuration

| Parameter | Value |
|---|---|
| Qubits | 8 |
| Encoding | Hadamard layer → `RY` angle encoding → brick-wall `CNOT` |
| Ansatz (default) | ChainEntangled (`sequential_su4`), **L = 2** |
| Measurements | single-qubit Pauli-*Z* expectation on all 8 qubits |
| Readout | fully connected 8 → 10 (`Normal(0.02)` init, zero bias) |
| Optimizer | Adam, learning rate `0.01` |
| Batch size | 32 |
| Loss | sparse categorical cross-entropy on the logits |
| Epochs | 150 (no early stopping) |
| Random seed | 42 |
| Parallel workers | 2 |
| Device / simulator | CPU / MindQuantum `mqvector` |

**Dimensionality reduction**

| Method | Configuration |
|---|---|
| PCA | 8 components, applied to the `[0, 1]`-scaled features |
| Autoencoder | `input → 256 → 64 → 8` latent, HeNormal init + ReLU on hidden layers, MSE loss, Adam `lr = 0.001`, batch 128, 50 epochs; parameters **frozen** afterwards (used as a fixed feature extractor) |
| UMAP | `n_components = 8`, `n_neighbors = 10`, `min_dist = 0.5`, `random_state = 42` |

---

## Naming conventions

The code names of the three Ansatz structures differ from the names used in the paper:

| Code | Paper |
|---|---|
| `sequential_su4` | **ChainEntangled** |
| `center_sequential_su4` | **CenterEntangled** |
| `so4` | **LightEntangled** (RY + RYY only) |

Result files with the suffix `nn10_md0.5` correspond to UMAP configured with `n_neighbors = 10`, `min_dist = 0.5`.

In Experiment II the **PCA** baseline is not recomputed: Experiment I already fixes the preprocessing method to PCA, so its result is reused.

---

## Outputs

Each notebook writes an `.npz` archive (and, for Experiment I, a `_history.pkl` plus a `_model.ckpt` per configuration) containing the per-epoch training loss, the per-epoch test accuracy, and the best test accuracy obtained over the whole run. The exact keys are visible in the final cell of each notebook.

---

## Key results

| Experiment | Result |
|---|---|
| I | ChainEntangled L = 2: **84.52%** (CenterEntangled 76.73%, LightEntangled 63.12%) |
| II | PCA **84.52%** / autoencoder **91.17%** / UMAP **94.26%** |
| III | Scaling interface gives no consistent gain; fully-connected scaling at the pre-encoding side drops to 76.36% under PCA |
| IV | The same interface at the measurement side recovers to 84.54% under PCA and 91.85% under the autoencoder |
| Ablation | Removing or freezing the Ansatz degrades accuracy; a modest classical MLP reaches 88.94–94.78% on the same 8-dimensional features |
| CIFAR-10 | binary: PCA 83.50% / autoencoder 82.00% / UMAP 81.50%; ten-class: 29.90% / 28.80% / 31.00% |

These numbers are reproduced here for convenience; the definitive values, protocols and discussion are in the paper.

---

## Reproducibility notes

- The random seed is fixed at `42` throughout.
- Each experimental configuration was run **once**, so the reported accuracies do not carry an estimate of run-to-run variance. Differences of a few tenths of a percentage point (e.g. among the three preprocessing methods on CIFAR-10) should be read as indicative rather than as reliable differences.

---

## Citation

```bibtex
@article{liu_qcnn_interface,
  title   = {Classical--Quantum Interface Design for Multi-Class Quantum Convolutional Neural Networks},
  author  = {Liu, Zihao and Zheng, Chengyong and Wang, Jinge},
  journal = {},
  year    = {}
}
```

*(Please update the journal and year once the paper is published.)*

---

## License

*To be added (e.g. MIT / Apache-2.0).*

## Contact

- Zihao Liu — School of Mathematics and Computational Science, Wuyi University, Jiangmen, China
- Issues and questions are welcome via the GitHub issue tracker.
