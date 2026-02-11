# LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs

This repository is the official implementation of [LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](https://arxiv.org/abs/2511.07379).

The workspace contains multiple components used in the paper workflow:

- `T-spear-shield/`: primary LoReTTA attack and robustness code
- `DyGLib/`: TGNN training/evaluation and temporal preprocessing utilities
- `tgl-main/`: temporal graph learning training framework used by attack pipelines
- `AnoGraph-main/`, `MIDAS.Python-master/`, `F-FADE-master/`: anomaly detection baselines/utilities

## LoReTTA at a Glance

LoReTTA is a low-resource, surrogate-free poisoning framework for continuous-time dynamic graphs (CTDGs).  
It applies a two-stage attack:

1. **Sparsification:** remove high-impact temporal edges (heuristic and timestamp-based variants)
2. **Replacement:** add constraint-aware adversarial negatives that preserve stealth constraints

The paper reports strong degradation across benchmark TGNNs while maintaining unnoticeability.

## Environment Setup (uv)

This workspace has been validated with a `uv`-managed virtual environment in `DyGLib/.venv`.

From workspace root:

```bash
cd DyGLib
uv venv --python 3.11 .venv
```

Install key dependencies:

```bash
uv pip install --python .venv/bin/python \
  torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchdata==0.7.1 torchtext==0.16.2 \
  --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install --python .venv/bin/python \
  torch-geometric==2.5.2 "numpy<2" opencv-python==4.8.1.78 \
  packaging setuptools dgl pydantic pybind11
```

Install the PyG auxiliary wheels:

```bash
uv pip install --python .venv/bin/python \
  "pyg-lib==0.4.0+pt21cu121" \
  "torch-scatter==2.1.2+pt21cu121" \
  "torch-sparse==0.6.18+pt21cu121" \
  "torch-cluster==1.6.3+pt21cu121" \
  "torch-spline-conv==1.2.2+pt21cu121" \
  --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## One-Time Build Step

`T-spear-shield` uses a C++ sampler extension.

```bash
cd T-spear-shield
../DyGLib/.venv/bin/python setup.py build_ext --inplace
```

## Smoke Tests

These commands verify import/startup integrity:

```bash
# DyGLib
cd DyGLib
.venv/bin/python train_link_prediction.py --help

# tgl-main (using DyGLib venv)
cd ../tgl-main
../DyGLib/.venv/bin/python train.py --help

# T-spear-shield
cd ../T-spear-shield
DGLBACKEND=pytorch ../DyGLib/.venv/bin/python Ours_train.py --help
```

## Data Path Assumptions

Current code paths in multiple modules are hardcoded to:

- `/raid/t2/TGN_adv/...`

This means full training/attack runs will fail unless:

- the `/raid/t2/TGN_adv` data tree exists and is writable, or
- the code is patched to use local configurable paths.

Startup/import tests can still pass without the full data tree.

## Important Local Compatibility Fixes Applied

To stabilize imports in this workspace:

- `DyGLib/preprocess_data/` imports were made package-relative to avoid import-order/circular path issues.
- `T-spear-shield/tatk/sampler.py` includes a fallback import for `sampler_core` when built as top-level extension.

## Suggested Reproduction Flow

1. Prepare datasets in the expected CTDG format.
2. Preprocess/sparsify data (DyGLib utilities).
3. Build sampler extension in `T-spear-shield`.
4. Run attack generation/training through `Ours_train.py`.
5. Evaluate victim models and compare against baselines.
6. Run anomaly detector scripts from their respective baseline folders.

## Citation

Please cite:

```bibtex
@misc{pal2025lorettalowresourceframework,
      title={LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs}, 
      author={Himanshu Pal and Venkata Sai Pranav Bachina and Ankit Gangwal and Charu Sharma},
      year={2025},
      eprint={2511.07379},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.07379}, 
}
```

