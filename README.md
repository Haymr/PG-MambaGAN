# PG-MambaGAN — `local-fixes-512` Branch

Bu branch, `main` üzerinde 512×512 batch=8 eğitiminin stabilize edilmesi için yapılan düzeltmeleri içerir.

> **Not:** Projenin bilimsel/teorik anlatımı için `main` branch'indeki README'ye bakınız. Bu README sadece bu branch'te yapılan değişiklikleri belgeler.

---

## 📊 Doğrulama

| Metrik | Önceki Çöküş | Bu Branch (Epoch 8) |
|---|---|---|
| HU PSNR | 15.16 dB | **31.44 dB** (+16 dB) |
| Val L1_HU | 279.1 | 47.8 |
| GP | 4202 (patladı) | 0.019 (Lipschitz-1) |
| D real / fake | dengesiz | 0.0066 / 0.0064 (denge) |

`λ_gp=10.0` korundu — Spectral Norm + GP=10 kombinasyonu 8 epoch'ta `‖∇D‖ → 1`'e yakınsadı.

---

## 🐛 Bug Fix'ler

### `data/dataset.py`
Çift HU normalizasyonu kaldırıldı. NPY'ler önişlemeden zaten `[-1,1]` aralığında geliyor; class onları HU varsayıp tekrar normalize ediyordu, veri dağılımı bozuluyordu.

```diff
- ldct = np.clip(ldct, -1000.0, 1000.0)
- ldct = (ldct + 1000.0) / 2000.0 * 2.0 - 1.0
+ ldct = np.clip(ldct, -1.0, 1.0)
```

### `models/generators/vss_unet.py`
Tanh öncesi son Conv'un init'i değiştirildi. `gain=0.1` başlangıç çıktısını sıfıra yapıştırıyor, generator gradyan alamıyordu.

```diff
- nn.init.xavier_uniform_(self.final_upsample[-2].weight, gain=0.1)
+ nn.init.xavier_normal_(self.final_upsample[-2].weight, gain=1.0)
```

### `setup/environment.py`
CUDA API tipo: `total_mem` → `total_memory`. Environment self-test artık çökmüyor.

```diff
- total_mem = torch.cuda.get_device_properties(0).total_mem
+ total_mem = torch.cuda.get_device_properties(0).total_memory
```

---

## 🎯 Algoritmik İyileştirme

### `models/losses/anatomy_nps.py`
NPS loss yeniden tasarlandı:

- `torch.log1p(power)` kaldırıldı (magnitude'u eziyordu)
- Unit-integral shape normalization eklendi (`nps / nps.sum()`)
- MSE → L1 (outlier'a robust, AAPM TG-233 ruhuna uygun)

```diff
- power = torch.log1p(power)
...
- nps_diff = F.mse_loss(nps_pred, nps_ndct)
+ nps_pred = nps_pred / (nps_pred.sum() + 1e-8)
+ nps_ndct = nps_ndct / (nps_ndct.sum() + 1e-8)
+ nps_diff = F.l1_loss(nps_pred, nps_ndct)
```

---

## ⚙️ Tuning

### `configs/default.yaml`
Eğitim parametreleri güncellendi:

| Parametre | Eski | Yeni |
|---|---|---|
| `epochs` | 200 | 120 |
| `batch_size` | 1 | 8 |
| `gradient_accumulation` | 8 | 1 |
| `ema_decay` | 0.999 | 0.99 |
| `log_images_every` | 10 | 1 |

> Effective batch aynı (8), ama accumulation overhead'i kalktı → hız artışı.
> EMA decay düşürmesi sample log gecikme sorununu çözdü.

---

## 🗑️ Cleanup

### `setup/colab_setup.py` (163 satır, **silindi**)
Aynı `total_mem` tipo'su burada da vardı. Proje WSL'e taşındığı için Colab setup'ı artık aktif kullanımda değil. Çift bakım yükü temizlendi.

---

## 📝 Yeni Dokümantasyon

| Dosya | İçerik |
|---|---|
| `CLAUDE.md` | Eğitim çöküş notları ve teşhis durumu |
| `COLLAPSE_DIAGNOSIS.md` | 8 adımlık kök-neden analizi |
| `tez_raporu_taslak.md` | ~12k kelime, 9 bölüm tez taslağı |

---

## 📋 Değişen Dosyaların Özeti

```
configs/default.yaml          | 10 +-       (tuning)
data/dataset.py               |  8 +-       (bug fix)
models/generators/vss_unet.py |  2 +-       (bug fix)
models/losses/anatomy_nps.py  |  9 +-       (algoritma)
setup/environment.py          |  2 +-       (bug fix)
setup/colab_setup.py          | 163 ---     (silindi)
README.md                     | yeni içerik (bu dosya)
CLAUDE.md                     | yeni        (notlar)
COLLAPSE_DIAGNOSIS.md         | yeni        (teşhis)
tez_raporu_taslak.md          | yeni        (tez)
```

**Toplam:** 6 dosya değişti, 1 silindi, 4 yeni dosya eklendi.
