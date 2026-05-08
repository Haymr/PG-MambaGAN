# PG-MambaGAN LLM Documentation

Welcome to the comprehensive, directory-by-directory, file-by-file documentation hierarchy for **PG-MambaGAN** (Physics-Guided Mamba Generative Adversarial Network for Low-Dose CT Denoising).

This documentation is specifically tailored for future AI Agents / LLMs. It explains *what* we are doing, *how* we are doing it, and crucially, *why* we are doing it, covering the architectural decisions and domain-specific risks.

## Documentation Index

The documentation is split into specialized modules for precise context retrieval. Please read the specific module related to your current task before making changes to the codebase.

1. [Project Overview & Fatal Risks](md files/01_PROJECT_OVERVIEW.md)
   * High-level goals, and critical domain-specific pitfalls to avoid (checkerboarding, SN buffer corruption, gradient leakage).
2. [Data Handling & Preprocessing](md files/02_DATA_HANDLING.md)
   * How raw DICOM files are converted to HU, normalized, and preprocessed. Morphological masking techniques.
3. [Model Architecture](md files/03_MODEL_ARCHITECTURE.md)
   * Detailed breakdown of the VSS-U-Net Generator and PatchGAN Discriminator. Exact tensor shapes and dimensional transformations.
4. [Loss Functions & Methodology](md files/04_LOSS_FUNCTIONS.md)
   * Complete theoretical and practical breakdown of L1, Adversarial, Perceptual, Frequency, and Anatomy-Aware NPS Loss. Cosine scheduling and tissue-specific lambdas.
5. [Training & Optimization](md files/05_TRAINING_AND_OPTIMIZATION.md)
   * Optimizer setups, AMP/BFloat16, EMA logic, WandB logging expectations, and diagnosing mode collapse.
6. [Evaluation Metrics](md files/06_EVALUATION.md)
   * Quantitative metrics (PSNR, SSIM, RMSE, MAE) and clinical 3D evaluations (EPI, CNR, Flickering Index).
7. [File-by-File Reference](md files/07_FILE_BY_FILE_REFERENCE.md)
   * A dictionary of every script in the repository detailing its purpose, inputs, and outputs.

---
**Core Design Philosophy:**
PG-MambaGAN combines the global receptive field efficiency of State Space Models (Mamba) with the realism of GANs, constrained heavily by the clinical physics of CT imaging (Noise Power Spectrum) to prevent hallucination. Any changes to the code must respect these clinical constraints.
