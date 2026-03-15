"""
半监督学习模块

包含:
1. urpc: URPC (Uncertainty-aware Rectified Pyramid Consistency) 方法
2. mean_teacher: Mean Teacher 方法 (本项目的第二个创新点)
"""

from . import urpc
from . import mean_teacher

__all__ = ['urpc', 'mean_teacher']
