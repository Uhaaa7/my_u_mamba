"""
Mean Teacher 半监督学习模块

包含:
1. semi_supervised_modules: 核心组件 (BoundaryHead, PseudoLabelRefiner, SkipDistillationLoss, etc.)
2. mean_teacher_wrapper: 网络包装器
3. nnUNetTrainerUMambaSDG_MeanTeacher: Mean Teacher 训练器
"""

from .semi_supervised_modules import (
    BoundaryHead,
    PseudoLabelRefiner,
    SkipDistillationLoss,
    BoundaryConsistencyLoss,
    SemiSupervisedLoss,
    EMAModel,
    get_current_consistency_weight
)

from .mean_teacher_wrapper import (
    MeanTeacherWrapper,
    InferenceOnlyWrapper,
    wrap_model_for_mean_teacher
)

from .nnUNetTrainerUMambaSDG_MeanTeacher import nnUNetTrainerUMambaSDG_MeanTeacher

__all__ = [
    'BoundaryHead',
    'PseudoLabelRefiner',
    'SkipDistillationLoss',
    'BoundaryConsistencyLoss',
    'SemiSupervisedLoss',
    'EMAModel',
    'get_current_consistency_weight',
    'MeanTeacherWrapper',
    'InferenceOnlyWrapper',
    'wrap_model_for_mean_teacher',
    'nnUNetTrainerUMambaSDG_MeanTeacher'
]
