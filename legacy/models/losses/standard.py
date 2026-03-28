"""
Standard Loss Functions
========================
Temel kayıp fonksiyonları: L1, Wasserstein, Gradient Penalty

Mevcut Pix2Pix + WGAN-GP eğitim framework'ünün çekirdeği.
"""

import tensorflow as tf


def l1_loss(y_true, y_pred):
    """
    L1 (Mean Absolute Error) Loss

    Piksel düzeyinde doğrudan benzerlik sağlar.
    Yapısal detayları korumada etkilidir.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein Loss (Earth Mover's Distance)

    WGAN formülasyonu:
    - Discriminator: D(real) - D(fake) minimize (real → yüksek, fake → düşük)
    - Generator: -D(fake) minimize (fake → yüksek)
    """
    return tf.reduce_mean(y_true * y_pred)


def gradient_penalty(discriminator, batch_size, real_images, fake_images,
                     input_images):
    """
    Gradient Penalty (GP) hesaplama — WGAN-GP

    Real ve fake görüntüler arasında interpolasyon yaparak
    discriminator gradyanlarının Lipschitz constraint'ini sağlar.

    Args:
        discriminator: Discriminator model
        batch_size: Batch boyutu
        real_images: Gerçek (NDCT) görüntüler
        fake_images: Üretilen görüntüler
        input_images: Giriş (LDCT) görüntüleri

    Returns:
        Gradient penalty değeri
    """
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([input_images, interpolated], training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
