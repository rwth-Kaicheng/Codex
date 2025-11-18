# Codex PPO Demo

This repository contains a minimal PyTorch implementation of Proximal Policy Optimization (PPO) on a simple 1-D navigation task. The demo is self contained and only depends on PyTorch.

Install PyTorch for your platform before running the script, for example:

```bash
pip install torch
```

## Running the demo

```bash
python ppo_demo.py --episodes 200 --batch-size 8
```

After training, the script runs a greedy evaluation and prints the achieved return. See `python ppo_demo.py --help` for tunable hyperparameters.
