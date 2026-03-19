from .standard import l1_loss, wasserstein_loss
from .perceptual import PerceptualLoss
from .physics_guided import NPSLoss, FrequencyLoss

__all__ = [
    'l1_loss', 'wasserstein_loss',
    'PerceptualLoss',
    'NPSLoss', 'FrequencyLoss'
]
