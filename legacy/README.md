# Legacy Code Archive

> **⚠️ Bu klasördeki kodlar artık aktif olarak kullanılmamaktadır.**

Bu dizin, projenin orijinal TensorFlow/Keras tabanlı ilk versiyonunu referans
amaçlı içermektedir. Aktif geliştirme tamamen PyTorch/MONAI ekosistemine
taşınmıştır.

## Neden Arşivlendi?

| Sorun | Orijinal Kod | Yeni Çözüm |
|---|---|---|
| Veri sızıntısı | Slice-bazlı rastgele bölme | Hasta-bazlı `PatientManifest` |
| Framework | TensorFlow 2.x (sınırlı Mamba desteği) | PyTorch 2.x + `mamba-ssm` |
| Mimari | Basit U-Net + Pix2Pix | VSS-U-Net (4-yönlü Mamba SSM) |
| Loss | Global L1 + Perceptual | Anatomy-Aware NPS + 5 bileşen |
| Çözünürlük | 256×256 | 512×512 |
| Değerlendirme | Sadece 2D PSNR/SSIM | 3D NIfTI + Halüsinasyon + Klinik |

## Dosya Yapısı

```
legacy/
├── configs/default.yaml       # Eski hiperparametreler
├── models/
│   ├── generators/
│   │   ├── mamba_gen.py        # İlk Mamba denemesi (TF)
│   │   └── unet_baseline.py   # TF U-Net
│   ├── discriminators/
│   │   └── patch_disc.py      # TF PatchGAN
│   └── losses/
│       ├── standard.py        # TF L1 + Wasserstein
│       ├── perceptual.py      # TF VGG loss
│       └── physics_guided.py  # İlk NPS denemesi
├── training/trainer.py        # TF eğitim döngüsü
└── train_legacy.py            # Eski entry point
```

## Kullanım

Bu kodlar **çalıştırılmak için değil**, akademik süreklilik ve referans
için saklanmaktadır. Aktif kod için ana dizini kullanın.
