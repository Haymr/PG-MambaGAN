"""
U-Net Baseline Generator
========================
Orijinal Pix2Pix U-Net Generator mimarisi.
Ablation study ve kıyaslama (baseline) için korunuyor.

Mimari: 8-layer Encoder, 7-layer Decoder, Skip Connections
Çıktı aktivasyonu: tanh ([-1, 1] aralığı)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def downsample(filters, size, apply_batchnorm=True):
    """Encoder bloğu: Conv2D → BatchNorm → LeakyReLU"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """Decoder bloğu: Conv2DTranspose → BatchNorm → Dropout → ReLU"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result


def build_generator(img_width=256, img_height=256, channels=1):
    """
    U-Net Generator (Baseline)

    Args:
        img_width: Giriş genişliği
        img_height: Giriş yüksekliği
        channels: Kanal sayısı (CT için 1)

    Returns:
        keras.Model: U-Net Generator
    """
    inputs = layers.Input(shape=[img_width, img_height, channels])

    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),   # (bs, 128, 128, 64)
        downsample(128, 4),                           # (bs, 64, 64, 128)
        downsample(256, 4),                           # (bs, 32, 32, 256)
        downsample(512, 4),                           # (bs, 16, 16, 512)
        downsample(512, 4),                           # (bs, 8, 8, 512)
        downsample(512, 4),                           # (bs, 4, 4, 512)
        downsample(512, 4),                           # (bs, 2, 2, 512)
        downsample(512, 4),                           # (bs, 1, 1, 512)
    ]

    # Decoder
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        channels, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'
    )

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    return keras.Model(inputs=inputs, outputs=x, name='unet_baseline_generator')
