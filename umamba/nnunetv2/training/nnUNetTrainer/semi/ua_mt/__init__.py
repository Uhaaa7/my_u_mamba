"""
UA-MT (Uncertainty-Aware Mean Teacher) 半监督学习方法

核心组件:
1. nnUNetTrainerUMambaSDGUAMT: UA-MT 训练器
2. ua_mt_modules: 工具函数模块
"""

from .nnUNetTrainerUMambaSDGUAMT import nnUNetTrainerUMambaSDGUAMT
from .ua_mt_modules import (
    update_ema_variables,
    sigmoid_rampup,
    get_current_consistency_weight,
    compute_entropy_uncertainty,
    normalize_entropy,
    generate_uncertainty_mask,
    softmax_mse_loss,
    softmax_kl_loss,
    EMAModel,
    UncertaintyEstimator
)

__all__ = [
    'nnUNetTrainerUMambaSDGUAMT',
    'update_ema_variables',
    'sigmoid_rampup',
    'get_current_consistency_weight',
    'compute_entropy_uncertainty',
    'normalize_entropy',
    'generate_uncertainty_mask',
    'softmax_mse_loss',
    'softmax_kl_loss',
    'EMAModel',
    'UncertaintyEstimator'
]
