# AGENTS.md

## Repository Overview

nanoGPT is a minimal, flat Python project for training/finetuning GPT-2 models. No build system, no tests, no linting, no packaging — just plain scripts run directly.

## Key Commands

```bash
# Install dependencies (no requirements.txt — install manually)
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Prepare data (run before training)
python data/shakespeare_char/prepare.py   # char-level Shakespeare (~seconds)
python data/shakespeare/prepare.py        # BPE Shakespeare (~seconds)
python data/openwebtext/prepare.py        # OpenWebText (large download)

# Train
python train.py config/train_shakespeare_char.py          # small model, ~3 min on GPU
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py  # GPT-2 124M, ~4 days

# Sample / inference
python sample.py --out_dir=out-shakespeare-char

# Benchmark
python bench.py
```

## Configuration System

`configurator.py` is exec'd (not imported) by train.py, sample.py, and bench.py. It overrides global variables via:
- `--key=value` command-line args (type must match the existing global's type)
- Positional args treated as Python config file paths (exec'd in order)

Config files (in `config/`) are plain Python that directly assign globals.

## Architecture

- `train.py` — training loop (~300 lines)
- `model.py` — GPT model definition (~330 lines): LayerNorm, CausalSelfAttention (flash attention if PyTorch >= 2.0), MLP, Block, GPT
- `sample.py` — inference/sampling script
- `bench.py` — simplified training loop for benchmarking
- `configurator.py` — config override mechanism via exec + globals

## Gotchas

- **No flash attention warning**: If PyTorch < 2.0, slow attention is used with a runtime warning.
- **CPU training**: Must set `--device=cpu --compile=False`. Also reduce block_size, n_layer, n_head, n_embd for reasonable speed.
- **DDP without Infiniband**: Prepend `NCCL_IB_DISABLE=1` to torchrun commands.
- **Type enforcement**: `configurator.py` asserts `type(attempt) == type(globals()[key])` — mismatched types raise AssertionError.
- **Config order**: Config files are exec'd in argument order, then `--key=value` overrides apply. Later values win.
- **Apple Silicon**: Use `--device=mps` for Metal GPU acceleration.
- **Checkpoints**: Saved as `.pt` files in `--out_dir` (default `out/`). `.bin` and `.pkl` files are in `.gitignore`.
