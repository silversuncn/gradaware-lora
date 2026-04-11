# GradAware-LoRA: Gradient-Informed Rank Redistribution for Low-Rank Adaptation in Low-Resource Settings

This repository contains the experimental data and source code for the paper:

> **Gradient-Informed Rank Redistribution for Low-Rank Adaptation: An Empirical Assessment on GLUE Benchmarks in Low-Resource Settings**
> Yaowen Sun, Hai Fu, Gang Li — Navy Submarine Academy, PLA, Qingdao, China

## Overview

GradAware-LoRA collects per-layer gradient norms in a single forward-backward pass and redistributes the total LoRA rank budget using square-root scaling motivated by the Fisher Information diagonal. The method requires no custom modules and plugs directly into the standard HuggingFace PEFT `rank_pattern` API.

## Repository Structure

```
├── src/                          # Source code
│   ├── gradaware_lora.py         # Core GradAware-LoRA implementation
│   ├── training.py               # Training loop for encoder-only models
│   ├── constants.py              # Hyperparameter constants
│   ├── aggregate_results.py      # Result aggregation utilities
│   └── statistical_analysis.py   # Statistical testing and visualization
├── data/                         # Experimental results
│   ├── results.csv               # 423 rows (375 unique configs; 48 duplicate runs included)
│   ├── results.json              # Same data in JSON format with full metrics
│   └── statistical_analysis.json # Pre-computed statistical tests
└── figures/                      # All paper figures (matplotlib output)
```

## Experimental Setup

- **Tasks**: CoLA, MRPC, QNLI, RTE, SST-2 (GLUE benchmark)
- **Models**: DistilBERT, BERT-base, RoBERTa-base
- **Methods**: Full fine-tuning, BitFit, LoRA (r=8), TopHeavy-LoRA, GradAware-LoRA
- **Seeds**: 42, 123, 456, 789, 1024
- **Training samples**: 500 per task
- **Total runs**: 375

## Hardware & Environment

All experiments were conducted on a single workstation:

| Component | Specification |
|---|---|
| CPU | Intel Core i9-12900K (16C/24T) |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| OS | Ubuntu 22.04 (WSL2) |

### Software Versions

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.4.0 |
| PEFT | 0.18.1 |
| Datasets | 4.8.4 |
| scikit-learn | 1.8.0 |

## Key Results

- Pooled over all cells, GradAware-LoRA and uniform LoRA are statistically indistinguishable (mean Δ = +0.00069, p = 0.638).
- GradAware-LoRA wins consistently on CoLA across all three encoders.
- GradAware-LoRA significantly outperforms both BitFit and full fine-tuning in pooled terms.

## Requirements

```
torch>=2.11.0
transformers>=5.4.0
peft>=0.18.1
datasets>=4.8.0
scikit-learn>=1.8.0
scipy
matplotlib
```

## License

This repository is provided for academic reproducibility purposes. Please cite the paper if you use this code or data.

## Citation

```bibtex
@article{sun2026gradawarelora,
  title={Gradient-Informed Rank Redistribution for Parameter-Efficient Fine-Tuning in Low-Resource Settings},
  author={Sun, Yaowen and Fu, Hai and Li, Gang},
  year={2026}
}
```
