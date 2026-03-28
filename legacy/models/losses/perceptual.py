"""
Perceptual Loss (VGG-based)
============================
VGG19'un ara katman özelliklerini kullanarak perceptual benzerlik ölçer.

Piksel düzeyinde loss'un yanında, insan algısına yakın bir metrik sağlar.
GAN eğitiminde doku kalitesini önemli ölçüde artırır.
"""

import tensorflow as tf
from tensorflow import keras


class PerceptualLoss(keras.layers.Layer):
    """
    VGG19 Perceptual Loss

    VGG19'un seçili katmanlarından çıkarılan feature map'ler
    arasındaki L1/L2 mesafeyi hesaplar.

    Kullanılan katmanlar:
    - block1_conv2: Düşük seviye doku özellikleri
    - block2_conv2: Orta kenar özellikleri
    - block3_conv4: Yüksek seviye yapı
    - block5_conv4: Semantik özellikler
    """

    def __init__(self, feature_layers=None, **kwargs):
        super().__init__(**kwargs)

        if feature_layers is None:
            feature_layers = [
                'block1_conv2',
                'block2_conv2',
                'block3_conv4',
                'block5_conv4',
            ]

        # VGG19 backbone (ImageNet ağırlıkları)
        vgg = keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        # Seçili katmanlardan çıktı al
        outputs = [vgg.get_layer(name).output for name in feature_layers]
        self.feature_extractor = keras.Model(
            inputs=vgg.input, outputs=outputs, name='vgg_features'
        )
        self.feature_extractor.trainable = False

    def _preprocess(self, x):
        """
        CT görüntüsünü VGG için uygun formata dönüştür.
        [-1, 1] → [0, 255], 1ch → 3ch
        """
        # [-1, 1] → [0, 1]
        x = (x + 1.0) / 2.0
        # [0, 1] → [0, 255]
        x = x * 255.0
        # 1 kanal → 3 kanal (VGG RGB bekler)
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        # VGG preprocessing (BGR dönüşümü + ortalama çıkarma)
        x = keras.applications.vgg19.preprocess_input(x)
        return x

    def call(self, y_true, y_pred):
        """
        Perceptual loss hesapla.

        Args:
            y_true: Hedef görüntü (NDCT) — [-1, 1]
            y_pred: Üretilen görüntü — [-1, 1]

        Returns:
            Toplam perceptual loss
        """
        y_true_vgg = self._preprocess(y_true)
        y_pred_vgg = self._preprocess(y_pred)

        true_features = self.feature_extractor(y_true_vgg)
        pred_features = self.feature_extractor(y_pred_vgg)

        loss = 0.0
        for tf_feat, pf_feat in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.abs(tf_feat - pf_feat))

        return loss / len(true_features)
