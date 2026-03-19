---
description: How to evaluate a trained PG-MambaGAN model
---

# Model Değerlendirme Workflow'u

Eğitilmiş modelin PSNR, SSIM ve NPS metriklerini hesaplar.

// turbo-all

### 1. Değerlendirme Scriptini Çalıştır

```bash
python -c "
import sys, glob, numpy as np
sys.path.insert(0, '.')
from models.generators import build_mamba_u_generator
from evaluation.metrics import evaluate_batch, print_results

# Model yükle
generator = build_mamba_u_generator()
generator.load_weights('experiments/mamba_u/checkpoints/G_final.h5')

# Veri yükle
val_A = sorted(glob.glob('data/mayo/trainA/*.npy'))[-100:]  # Son 100 slice
val_B = sorted(glob.glob('data/mayo/trainB/*.npy'))[-100:]

X = np.array([np.expand_dims(np.load(f), -1) if np.load(f).ndim==2 else np.load(f) for f in val_A])
Y = np.array([np.expand_dims(np.load(f), -1) if np.load(f).ndim==2 else np.load(f) for f in val_B])

results = evaluate_batch(generator, X, Y)
print_results(results, 'Mayo Validation')
"
```

### 2. Baseline ile Karşılaştır

```bash
python -c "
import sys, glob, numpy as np
sys.path.insert(0, '.')
from models.generators import build_unet_baseline
from evaluation.metrics import evaluate_batch, print_results

generator = build_unet_baseline()
generator.load_weights('experiments/unet_baseline/checkpoints/G_final.h5')

val_A = sorted(glob.glob('data/mayo/trainA/*.npy'))[-100:]
val_B = sorted(glob.glob('data/mayo/trainB/*.npy'))[-100:]

X = np.array([np.expand_dims(np.load(f), -1) if np.load(f).ndim==2 else np.load(f) for f in val_A])
Y = np.array([np.expand_dims(np.load(f), -1) if np.load(f).ndim==2 else np.load(f) for f in val_B])

results = evaluate_batch(generator, X, Y)
print_results(results, 'Baseline U-Net')
"
```
