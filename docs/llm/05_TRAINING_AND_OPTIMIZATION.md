# 05. Training & Optimization

This document outlines the optimization strategy, parameters, and debugging metrics used in the PG-MambaGAN trainer (`training/trainer.py`).

## Optimizer Setup
- **Generator & Discriminator Optimizers:** `torch.optim.AdamW`
- **Learning Rates:** `1.0e-4` for both. (WGAN-GP allows balanced learning rates, unlike standard GANs where D usually requires a lower LR).
- **Betas:** `(0.0, 0.99)`. Beta1 is 0.0, as standard momentum in adversarial settings causes instability in the gradient penalty.
- **Discriminator Critic Steps:** `n_critic = 5`. The discriminator updates 5 times for every 1 generator update to enforce the Wasserstein distance strictly.

## Mixed Precision & VRAM Protection
- **AMP (Automatic Mixed Precision):** Uses `torch.amp.autocast(dtype=torch.bfloat16)`. `bfloat16` prevents gradient overflow (NaN) issues inherent to standard `float16`, especially with adversarial losses.
- **Gradient Checkpointing:** Active by default in `vss_unet.py` to prevent OOM errors at 512x512 resolution (saves ~60% VRAM).
- **Physical Batch vs. Effective Batch:** With ~16 GB GPUs (e.g., RTX 5080 Mobile), the current configuration uses `batch_size=8` with `gradient_accumulation=1` directly — the gradient-checkpointing + bfloat16 combination keeps VRAM at ~12.6 GB peak. On more constrained GPUs (e.g., T4 12 GB), fall back to `batch_size=1` with `gradient_accumulation=8` for the same effective batch of 8. Override via CLI: `--batch-size N --accumulation M`.

## EMA (Exponential Moving Average)
- `G_ema` maintains a shadow copy of the generator's weights, updated slowly (`decay=0.99`).
- **Use Case:** Validation metrics, test evaluation, and checkpoint saving rely strictly on `G_ema`. This prevents capturing the generator during a temporary unstable spike.
- **Decay choice:** A higher decay (e.g., `0.999`) creates a heavy lag between the live generator and the EMA copy. With per-epoch image logging, this lag visibly delays the appearance of new structural detail in samples — readers/observers see "stale" outputs. `decay=0.99` is a balanced compromise that smooths instability but tracks the generator closely enough for diagnostic visualization.

## W&B Logging Expectations
When observing a training run on Weights & Biases (W&B), an agent should look for:
- **`d/d_total` and `g/g_total`:** Should fluctuate but generally remain stable. A sudden spike in `g_total` accompanied by `d_total` dropping to zero indicates Discriminator Mode Collapse.
- **`w/l1_weight` & `w/nps_weight`:** Should show the exact Cosine Cross-Fade schedule crossing paths (L1 going down, NPS going up).
- **Images:** Logged using `G_raw` (the actively training model, not EMA) with a **Fixed Sample Batch**. By fixing the samples per epoch (finding slices with the highest anatomical variance/STD), visual assessment is perfectly consistent over time.

## Diagnosing Mode Collapse
If the generator starts outputting **solid black squares** or **checkerboards of pure white dots**, check:
1. **D.eval() state:** Is the discriminator explicitly in `.eval()` during the `G` step? If not, SN buffers are poisoned.
2. **Gradient Clipping:** Ensure `torch.nn.utils.clip_grad_norm_(G, max_norm=1.0)` is active. NPS FFT losses can cause sudden infinite gradients.
3. **Tanh Output Saturation/Collapse:** Verify that the final layer uses `nn.init.xavier_normal_(weight, gain=1.0)`. An older variant (`xavier_uniform_(gain=0.1)`) caused the opposite failure — uniform near-zero "dead" outputs that took many epochs to escape.
4. **Data Range Sanity:** `data/dataset.py` expects `.npy` files already normalized to `[-1, 1]`. If the dataset is fed raw HU-valued NPYs (or a stale preprocessing output that double-normalizes), the input distribution is silently broken and the generator learns nothing useful.
