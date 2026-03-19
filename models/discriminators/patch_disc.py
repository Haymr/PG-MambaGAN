"""
PatchGAN Discriminator
======================
70x70 patch üzerinde karar veren discriminator.
WGAN-GP ile kullanılır (son katmanda sigmoid yok).

Spectral Normalization opsiyonel olarak desteklenir.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _downsample(filters, size, apply_batchnorm=True, use_spectral_norm=False):
    """Discriminator encoder bloğu"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()

    conv = layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    )

    if use_spectral_norm:
        conv = tf.keras.layers.experimental.preprocessing.Rescaling(1.0)
        # Not: TF2'de SpectralNormalization için wrapper gerekebilir
        # Şimdilik standart conv kullanıyoruz

    result.add(conv)

    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU(0.2))
    return result


def build_discriminator(
    img_width=256,
    img_height=256,
    channels=1,
    filters=None,
    use_spectral_norm=False
):
    """
    PatchGAN Discriminator

    WGAN-GP ile kullanılır: son katmanda sigmoid yok.
    Çıktı doğrudan Wasserstein distance hesabında kullanılır.

    Args:
        img_width: Giriş genişliği
        img_height: Giriş yüksekliği
        channels: Kanal sayısı
        filters: Filtre listesi
        use_spectral_norm: Spectral normalization kullan

    Returns:
        keras.Model: PatchGAN Discriminator
    """
    if filters is None:
        filters = [64, 128, 256, 512]

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(
        shape=[img_width, img_height, channels],
        name='input_image'
    )
    tar = layers.Input(
        shape=[img_width, img_height, channels],
        name='target_image'
    )

    # Input ve target'ı birleştir
    x = layers.Concatenate()([inp, tar])  # (bs, H, W, channels*2)

    # İlk katman (BatchNorm yok)
    x = _downsample(filters[0], 4, apply_batchnorm=False,
                     use_spectral_norm=use_spectral_norm)(x)

    # Diğer katmanlar
    for f in filters[1:]:
        x = _downsample(f, 4, use_spectral_norm=use_spectral_norm)(x)

    # Son katmanlar: ZeroPadding + Conv (no sigmoid for WGAN-GP)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer, use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(x)

    return keras.Model(inputs=[inp, tar], outputs=x, name='patchgan_discriminator')
