"""
Physics-Guided Loss Functions (PG-MambaGAN Orijinal Katkı)
===========================================================
CT fiziğine dayalı kayıp fonksiyonları:

1. NPS Loss: Noise Power Spectrum koruması
   - Gürültü giderme sonrası görüntünün gürültü dokusunun
     NDCT ile aynı frekans dağılımını korumasını sağlar.

2. Frequency Loss: FFT tabanlı frekans alanı kaybı
   - Yüksek frekans bileşenlerinin (kenar, doku detayları)
     korunmasını sağlar.

Bu fonksiyonlar, PG-MambaGAN'ın ikinci temel katkısıdır (Katkı #2).
Adversarial eğitim + fizik tabanlı loss kombinasyonu littratürde
daha önce LDCT denoising için birleştirilmemiştir.
"""

import tensorflow as tf
from tensorflow import keras


class NPSLoss(keras.layers.Layer):
    """
    Noise Power Spectrum (NPS) Loss

    NPS, görüntüdeki gürültünün frekans alanındaki dağılımını ölçer.
    İdeal durumda, denoised görüntünün NPS'si NDCT'nin NPS'sine yakın olmalıdır.

    NPS hesabı:
    1. Gürültü haritası çıkar: noise = image - mean_filtered
    2. 2D FFT uygula
    3. Power spectrum: |FFT|²
    4. Radial ortalama ile 1D NPS

    Loss: L_NPS = ||NPS(denoised) - NPS(target)||₂²

    Referans: CT-Mamba (2025) NPS loss'undan ilham alınmıştır,
    ancak adversarial framework'e entegrasyon orijinaldir.
    """

    def __init__(self, patch_size=64, num_patches=4, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches

    def _extract_noise_patches(self, image):
        """
        Görüntüden rastgele gürültü patch'leri çıkar.
        Gürültü = orijinal - düzleştirilmiş (mean filter)
        """
        # Mean filter (3x3) ile düzleştir
        kernel = tf.ones([3, 3, 1, 1]) / 9.0
        smoothed = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1],
                                padding='SAME')

        # Gürültü haritası
        noise = image - smoothed

        # Rastgele patch'ler çıkar
        batch_size = tf.shape(image)[0]
        h = tf.shape(image)[1]
        w = tf.shape(image)[2]

        patches = []
        for _ in range(self.num_patches):
            y = tf.random.uniform([], 0, h - self.patch_size, dtype=tf.int32)
            x = tf.random.uniform([], 0, w - self.patch_size, dtype=tf.int32)
            patch = noise[:, y:y + self.patch_size, x:x + self.patch_size, :]
            patches.append(patch)

        return patches

    def _compute_nps(self, patches):
        """
        Patch'lerden 2D NPS hesapla.

        NPS = mean(|FFT(noise_patch)|²)
        """
        nps_sum = tf.zeros([self.patch_size, self.patch_size])

        for patch in patches:
            # (batch, H, W, 1) → (batch, H, W)
            p = tf.squeeze(patch, axis=-1)

            # 2D FFT
            fft = tf.signal.fft2d(tf.cast(p, tf.complex64))

            # Power spectrum
            power = tf.abs(fft) ** 2

            # Batch ortalaması
            nps_sum += tf.reduce_mean(power, axis=0)

        # Patch sayısına böl
        nps = nps_sum / float(self.num_patches)

        # FFT shift (DC bileşeni merkeze)
        nps = tf.signal.fftshift(nps)

        return nps

    def call(self, y_true, y_pred):
        """
        NPS Loss hesapla.

        Args:
            y_true: NDCT görüntü
            y_pred: Denoised görüntü

        Returns:
            NPS loss (L2 distance between NPS curves)
        """
        # Gürültü patch'leri çıkar
        true_patches = self._extract_noise_patches(y_true)
        pred_patches = self._extract_noise_patches(y_pred)

        # NPS hesapla
        nps_true = self._compute_nps(true_patches)
        nps_pred = self._compute_nps(pred_patches)

        # L2 mesafe (normalize)
        nps_true_norm = nps_true / (tf.reduce_max(nps_true) + 1e-8)
        nps_pred_norm = nps_pred / (tf.reduce_max(nps_pred) + 1e-8)

        loss = tf.reduce_mean(tf.square(nps_true_norm - nps_pred_norm))

        return loss


class FrequencyLoss(keras.layers.Layer):
    """
    Frequency Domain (FFT) Loss

    Frekans alanında doğrudan karşılaştırma yaparak
    yüksek frekans bileşenlerinin (kenar, doku) korunmasını sağlar.

    Loss komponentleri:
    1. Genlik (magnitude) kaybı: |FFT(pred)| vs |FFT(true)|
    2. Faz (phase) kaybı: angle(FFT(pred)) vs angle(FFT(true))

    Bu, spatial domain'deki L1 loss'un tamamlayıcısıdır:
    L1 piksel düzeyinde çalışırken, FFT loss frekans düzeyinde çalışır.
    """

    def __init__(self, alpha_magnitude=1.0, alpha_phase=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha_magnitude = alpha_magnitude
        self.alpha_phase = alpha_phase

    def call(self, y_true, y_pred):
        """
        Frekans alanı kaybı hesapla.

        Args:
            y_true: NDCT görüntü — [-1, 1]
            y_pred: Denoised görüntü — [-1, 1]

        Returns:
            Toplam frekans kaybı
        """
        # (batch, H, W, 1) → (batch, H, W)
        true_2d = tf.squeeze(y_true, axis=-1)
        pred_2d = tf.squeeze(y_pred, axis=-1)

        # 2D FFT
        fft_true = tf.signal.fft2d(tf.cast(true_2d, tf.complex64))
        fft_pred = tf.signal.fft2d(tf.cast(pred_2d, tf.complex64))

        # Genlik (magnitude) kaybı
        mag_true = tf.abs(fft_true)
        mag_pred = tf.abs(fft_pred)

        # Log-scale magnitude (büyük değer aralığını daraltır)
        mag_true_log = tf.math.log1p(mag_true)
        mag_pred_log = tf.math.log1p(mag_pred)

        magnitude_loss = tf.reduce_mean(tf.abs(mag_true_log - mag_pred_log))

        # Faz (phase) kaybı
        phase_true = tf.math.angle(fft_true)
        phase_pred = tf.math.angle(fft_pred)

        phase_loss = tf.reduce_mean(tf.abs(phase_true - phase_pred))

        # Toplam
        total_loss = (self.alpha_magnitude * magnitude_loss +
                      self.alpha_phase * phase_loss)

        return total_loss
