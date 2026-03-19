---
description: How to train the PG-MambaGAN model end-to-end
---

# PG-MambaGAN Eğitim Workflow'u

Bu workflow, LDCT Denoising için PG-MambaGAN modelini sıfırdan eğitir.

## Ön Koşullar
- NVIDIA GPU (RTX 5080 veya benzeri, min. 12GB VRAM)
- Mayo Clinic LDCT DICOM veri seti (diskte)
- Python 3.9+, CUDA toolkit kurulu

---

## Adımlar

// turbo-all

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 2. DICOM Verisini NPY'ye Dönüştür

DICOM verisi eğitimde doğrudan kullanılmaz, önce NPY formatına çevrilmelidir.

```bash
python preprocess.py --input /path/to/DICOM/dataset --output ./data/mayo --size 256
```

**Parametreler:**
- `--input`: Ham DICOM klasörü (hasta alt klasörleri içeren, örn: `C002/`, `L004/` gibi)
- `--output`: NPY çıktı klasörü (trainA/trainB otomatik oluşturulur)
- `--size`: Görüntü boyutu (256 veya 512)

**Beklenen çıktı:**
```
data/mayo/
├── trainA/    # Low-dose CT (LDCT) slices — .npy
└── trainB/    # Normal-dose CT (NDCT) slices — .npy
```

### 3. PG-MambaGAN Eğitimi (Önerilen Model)

```bash
python train.py --data-path ./data/mayo --config configs/default.yaml
```

Bu, Mamba-U Generator + WGAN-GP + Physics-Guided Loss ile eğitim başlatır.

**Eğitim süresi:** ~6-10 saat (RTX 5080, 256×256, 100 epoch)

**Checkpoint'lar:** `experiments/mamba_u/checkpoints/` altında her 5 epoch'ta kaydedilir.

### 4. Baseline U-Net Eğitimi (Karşılaştırma İçin)

Ablation study için baseline modeli de eğitin:

```bash
python train.py --data-path ./data/mayo --generator unet_baseline
```

### 5. Checkpoint'tan Devam Etme (opsiyonel)

Eğitim kesilirse:

```bash
python train.py --data-path ./data/mayo --resume experiments/mamba_u/checkpoints/G_epoch_50.h5
```

### 6. Sonuçları Kontrol Et

Eğitim sırasında üretilen görseller:
```
experiments/mamba_u/results/       # Epoch bazlı LDCT → Denoised → NDCT karşılaştırma
experiments/mamba_u/checkpoints/   # Model ağırlıkları (.h5)
```

---

## Proje Yapısı (Referans)

```
PG-MambaGAN/
├── preprocess.py          # DICOM → NPY dönüşümü
├── train.py               # Eğitim entry point
├── configs/default.yaml   # Hiperparametreler
├── models/
│   ├── generators/
│   │   ├── mamba_gen.py   # ⭐ Mamba-U Generator (yeni mimari)
│   │   └── unet_baseline.py  # Baseline (karşılaştırma)
│   ├── discriminators/
│   │   └── patch_disc.py
│   └── losses/
│       ├── standard.py        # L1 + Wasserstein + GP
│       ├── perceptual.py      # VGG perceptual loss
│       └── physics_guided.py  # ⭐ NPS + FFT loss (yeni)
├── training/trainer.py    # PGMambaGAN eğitim sınıfı
└── evaluation/metrics.py  # PSNR, SSIM, NPS
```

## Kayıp Fonksiyonu

```
L_total = λ₁·L_adv + λ₂·L_l1 + λ₃·L_perceptual + λ₄·L_NPS + λ₅·L_freq
```

Ağırlıklar `configs/default.yaml` dosyasından kontrol edilir.

## Notlar
- Eğer GPU bellek hatası alırsan, `configs/default.yaml` içindeki `batch_size`'ı 2'ye düşür.
- VGG perceptual loss ilk çalıştırmada ImageNet ağırlıklarını indirir (~500MB).
- NPS loss frekans alanında çalışır, eğitimi biraz yavaşlatabilir.
