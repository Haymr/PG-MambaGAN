"""
PG-MambaGAN: Train Script
===========================
Komut satırından eğitim başlatma.

Kullanım:
    python train.py                           # Default config
    python train.py --config configs/default.yaml  # Custom config
    python train.py --generator unet_baseline  # Baseline karşılaştırma
"""

import os
import sys
import glob
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import (
    PGMambaGAN, NPYDataset, GANMonitor,
    load_config, build_model_from_config
)


def parse_args():
    parser = argparse.ArgumentParser(description='PG-MambaGAN Eğitimi')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config dosyası yolu')
    parser.add_argument('--generator', type=str, default=None,
                        help='Generator tipi override: unet_baseline / mamba_u')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Epoch sayısı override')
    parser.add_argument('--data-path', type=str, required=True,
                        help='NPY veri seti yolu (trainA/trainB alt klasörlü)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Çıktı klasörü')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint dosyası ile devam et')
    return parser.parse_args()


def main():
    args = parse_args()

    # Config yükle
    config = load_config(args.config)

    # Override'ları uygula
    if args.generator:
        config['model']['generator'] = args.generator
    if args.epochs:
        config['training']['epochs'] = args.epochs

    tc = config['training']

    # =========================================================================
    # Veri Yükleme
    # =========================================================================
    dataset_path = args.data_path

    all_files_A = sorted(glob.glob(os.path.join(dataset_path, 'trainA', '*.npy')))
    all_files_B = sorted(glob.glob(os.path.join(dataset_path, 'trainB', '*.npy')))

    # Eşleşen dosyaları bul
    filenames_A = {os.path.basename(f) for f in all_files_A}
    filenames_B = {os.path.basename(f) for f in all_files_B}
    common = sorted(list(filenames_A.intersection(filenames_B)))

    files_A = [os.path.join(dataset_path, 'trainA', f) for f in common]
    files_B = [os.path.join(dataset_path, 'trainB', f) for f in common]

    print(f"📁 Toplam eşleşen veri: {len(files_A)}")

    # Train/Val split
    train_A, val_A, train_B, val_B = train_test_split(
        files_A, files_B,
        test_size=tc.get('val_split', 0.10),
        random_state=tc.get('random_seed', 42)
    )

    print(f"🔧 Eğitim: {len(train_A)} | Doğrulama: {len(val_A)}")

    train_dataset = NPYDataset(train_A, train_B, batch_size=tc['batch_size'])
    val_dataset = NPYDataset(val_A, val_B, batch_size=tc['batch_size'], shuffle=False)

    # =========================================================================
    # Model Oluşturma
    # =========================================================================
    model = build_model_from_config(config)

    from tensorflow import keras
    model.compile(
        d_optimizer=keras.optimizers.Adam(
            learning_rate=tc['learning_rate_d'],
            beta_1=tc['beta1'], beta_2=tc['beta2']
        ),
        g_optimizer=keras.optimizers.Adam(
            learning_rate=tc['learning_rate_g'],
            beta_1=tc['beta1'], beta_2=tc['beta2']
        )
    )

    gen_name = config['model']['generator']
    print(f"\n🚀 Model: PG-MambaGAN (Generator: {gen_name})")
    print(f"   Generator params: {model.generator.count_params():,}")
    print(f"   Discriminator params: {model.discriminator.count_params():,}")

    # Resume from checkpoint
    if args.resume:
        model.generator.load_weights(args.resume)
        print(f"♻️  Checkpoint yüklendi: {args.resume}")

    # =========================================================================
    # Eğitim
    # =========================================================================
    exp_dir = os.path.join(args.output_dir, gen_name)
    results_dir = os.path.join(exp_dir, 'results')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')

    monitor = GANMonitor(
        val_dataset, results_dir, ckpt_dir,
        save_freq=tc.get('checkpoint_freq', 5)
    )

    print(f"\n{'=' * 60}")
    print(f"  PG-MambaGAN EĞİTİMİ BAŞLIYOR")
    print(f"  Generator: {gen_name}")
    print(f"  Epochs: {tc['epochs']}")
    print(f"  Batch Size: {tc['batch_size']}")
    print(f"  Loss: L_adv + L_l1 + L_perceptual + L_nps + L_freq")
    print(f"{'=' * 60}\n")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=tc['epochs'],
        callbacks=[monitor]
    )

    # Final model kaydet
    final_path = os.path.join(ckpt_dir, 'G_final.h5')
    model.generator.save_weights(final_path)
    print(f"\n✅ Eğitim tamamlandı! Final model: {final_path}")


if __name__ == '__main__':
    main()
