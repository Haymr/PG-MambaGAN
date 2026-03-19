"""
PG-MambaGAN Training Module
============================
WGAN-GP adversarial eğitim döngüsü + PG-MambaGAN kayıp fonksiyonları.

Mevcut Pix2Pix WGAN-GP eğitim framework'ünün genişletilmiş hali:
- Orijinal: L_adv + L_l1
- PG-MambaGAN: L_adv + L_l1 + L_perceptual + L_nps + L_freq
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

import tensorflow as tf
from tensorflow import keras

from models.generators import build_unet_baseline, build_mamba_u_generator
from models.discriminators import build_discriminator
from models.losses.standard import l1_loss, gradient_penalty
from models.losses.perceptual import PerceptualLoss
from models.losses.physics_guided import NPSLoss, FrequencyLoss


# =============================================================================
# Data Loader
# =============================================================================

class NPYDataset(keras.utils.Sequence):
    """
    Memory-efficient NPY veri yükleyici.

    Her batch'te dosyaları diskten okur (büyük veri setleri için).
    Her epoch sonunda shuffle yapar.
    """

    def __init__(self, file_list_A, file_list_B, batch_size=4, shuffle=True):
        self.files_A = np.array(file_list_A)
        self.files_B = np.array(file_list_B)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files_A))

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files_A) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_A, batch_B = [], []
        for k in indexes:
            imgA = np.load(self.files_A[k])
            imgB = np.load(self.files_B[k])

            if imgA.ndim == 2:
                imgA = np.expand_dims(imgA, axis=-1)
            if imgB.ndim == 2:
                imgB = np.expand_dims(imgB, axis=-1)

            batch_A.append(imgA)
            batch_B.append(imgB)

        return np.array(batch_A, dtype=np.float32), np.array(batch_B, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# =============================================================================
# PG-MambaGAN Model
# =============================================================================

class PGMambaGAN(keras.Model):
    """
    Physics-Guided Mamba-WGAN (PG-MambaGAN)

    Genişletilmiş WGAN-GP framework:
    - Generator: Mamba-U (veya baseline U-Net)
    - Discriminator: PatchGAN
    - Loss: L_adv + L_l1 + L_perceptual + L_nps + L_freq
    """

    def __init__(self, generator, discriminator, config=None, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

        # Loss weights (config'den veya default)
        if config and 'loss' in config:
            lc = config['loss']
            self.lambda_adv = lc.get('lambda_adv', 1.0)
            self.lambda_l1 = lc.get('lambda_l1', 100.0)
            self.lambda_perceptual = lc.get('lambda_perceptual', 10.0)
            self.lambda_nps = lc.get('lambda_nps', 5.0)
            self.lambda_freq = lc.get('lambda_freq', 1.0)
            self.lambda_gp = lc.get('gradient_penalty', 10.0)
        else:
            self.lambda_adv = 1.0
            self.lambda_l1 = 100.0
            self.lambda_perceptual = 10.0
            self.lambda_nps = 5.0
            self.lambda_freq = 1.0
            self.lambda_gp = 10.0

        # Loss modülleri
        self.perceptual_loss_fn = PerceptualLoss()
        self.nps_loss_fn = NPSLoss()
        self.freq_loss_fn = FrequencyLoss()

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        return self.generator(inputs, training=training)

    @tf.function
    def train_step(self, data):
        input_image, target_image = data
        batch_size = tf.shape(input_image)[0]

        # ----- DISCRIMINATOR EĞİTİMİ -----
        with tf.GradientTape() as d_tape:
            fake_image = self.generator(input_image, training=True)

            fake_pred = self.discriminator([input_image, fake_image], training=True)
            real_pred = self.discriminator([input_image, target_image], training=True)

            # Wasserstein distance
            d_cost = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

            # Gradient Penalty
            gp = gradient_penalty(
                self.discriminator, batch_size,
                target_image, fake_image, input_image
            )

            d_loss = d_cost + (gp * self.lambda_gp)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # ----- GENERATOR EĞİTİMİ -----
        with tf.GradientTape() as g_tape:
            fake_image = self.generator(input_image, training=True)
            fake_pred = self.discriminator([input_image, fake_image], training=True)

            # 1. Adversarial loss (WGAN)
            g_adv_loss = -tf.reduce_mean(fake_pred)

            # 2. L1 reconstruction loss
            g_l1 = l1_loss(target_image, fake_image) * self.lambda_l1

            # 3. Perceptual loss (VGG)
            g_perc = self.perceptual_loss_fn(
                target_image, fake_image
            ) * self.lambda_perceptual

            # 4. NPS loss (fizik tabanlı)
            g_nps = self.nps_loss_fn(
                target_image, fake_image
            ) * self.lambda_nps

            # 5. Frequency loss (FFT)
            g_freq = self.freq_loss_fn(
                target_image, fake_image
            ) * self.lambda_freq

            # Toplam Generator Loss
            g_loss = (self.lambda_adv * g_adv_loss +
                      g_l1 + g_perc + g_nps + g_freq)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_adv": g_adv_loss,
            "g_l1": g_l1,
            "g_perc": g_perc,
            "g_nps": g_nps,
            "g_freq": g_freq,
        }


# =============================================================================
# Training Callbacks
# =============================================================================

class GANMonitor(keras.callbacks.Callback):
    """Eğitim sırasında görsel izleme ve model checkpoint kaydetme"""

    def __init__(self, val_dataset, results_dir, checkpoint_dir,
                 num_img=3, save_freq=5):
        super().__init__()
        self.val_dataset = val_dataset
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_img = num_img
        self.save_freq = save_freq

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        try:
            idx = np.random.randint(0, len(self.val_dataset))
            inp, tar = self.val_dataset[idx]
            prediction = self.model.generator(inp, training=False)

            titles = ['Input (LDCT)', 'Generated (PG-MambaGAN)', 'Target (NDCT)']
            display_count = min(self.num_img, inp.shape[0])

            for i in range(display_count):
                img_list = [inp[i], prediction[i], tar[i]]

                plt.figure(figsize=(15, 5))
                for j in range(3):
                    plt.subplot(1, 3, j + 1)
                    plt.title(titles[j])
                    img_data = img_list[j][:, :, 0]
                    _min, _max = np.min(img_data), np.max(img_data)
                    if _max - _min > 0:
                        show_img = (img_data - _min) / (_max - _min)
                    else:
                        show_img = img_data
                    plt.imshow(show_img, cmap='gray')
                    plt.axis('off')

                filename = f"epoch_{epoch + 1:03d}_{i}.png"
                save_path = os.path.join(self.results_dir, filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

            if (epoch + 1) % self.save_freq == 0:
                model_name = f"G_epoch_{epoch + 1}.h5"
                ckpt_path = os.path.join(self.checkpoint_dir, model_name)
                self.model.generator.save_weights(ckpt_path)
                print(f"✅ Model kaydedildi: {ckpt_path}")

        except Exception as e:
            print(f"Kayıt hatası: {e}")


# =============================================================================
# Helper: Config yükleme
# =============================================================================

def load_config(config_path):
    """YAML config dosyasını yükle"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_config(config):
    """Config'den model oluştur"""
    mc = config['model']
    gen_type = mc.get('generator', 'mamba_u')

    if gen_type == 'unet_baseline':
        generator = build_unet_baseline(
            mc['img_width'], mc['img_height'], mc['channels']
        )
    elif gen_type == 'mamba_u':
        gc = config.get('generator', {})
        generator = build_mamba_u_generator(
            img_width=mc['img_width'],
            img_height=mc['img_height'],
            channels=mc['channels'],
            mamba_d_state=gc.get('mamba_d_state', 16),
            mamba_d_conv=gc.get('mamba_d_conv', 4),
            mamba_expand=gc.get('mamba_expand', 2),
        )
    else:
        raise ValueError(f"Bilinmeyen generator tipi: {gen_type}")

    discriminator = build_discriminator(
        mc['img_width'], mc['img_height'], mc['channels']
    )

    return PGMambaGAN(generator, discriminator, config=config)
