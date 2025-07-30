# geglu-vs-relu-mnist

This repository compares two feed-forward network variants `FFN_ReLU` and `FFN_GeGLU` on the MNIST dataset using PyTorch Lightning. The experiment tests whether GeGLU consistently performs better than ReLU at small hidden dimensions under limited capacity.

## What It Does
- Implements FFN_ReLU and FFN_GeGLU modules using `torch.nn` and `einsum`
- Trains and evaluates both on MNIST
- Runs experiments at hidden dimensions `[2, 4, 8, 16]`
- Performs random hyperparameter search with `k = 2,4,8` trials
- One epoch
- Computes test accuracy and bootstrapped error bars
- Plots accuracy vs. hidden dimension

## Requirements
This project uses [**uv**](https://github.com/astral-sh/uv) for managing dependencies. You can install it using:

```bash
pip install uv
```

Then install dependencies with:
```bash
uv venv
uv pip install -e .
```

## Results
Two plots are generated comparing ReLU vs GeGLU:
- Test accuracy vs hidden dim
- Error bars from bootstrap CI

## Citation
Based on the paper:
> "GLU Variants Improve Transformer" – Noam Shazeer, 2020
> https://arxiv.org/abs/2002.05202
