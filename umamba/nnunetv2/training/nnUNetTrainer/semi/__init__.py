"""
半监督学习模块

包含:
1. urpc: URPC (Uncertainty-aware Rectified Pyramid Consistency) 方法
2. mean_teacher: Mean Teacher 方法 (本项目的第二个创新点)
3. ua_mt: UA-MT (Uncertainty-Aware Mean Teacher) 方法
4. cps: CPS (Cross Pseudo Supervision) 方法
5. abd: ABD (Attention-Based Dual-branch) 方法
"""

from . import urpc
from . import mean_teacher
from . import ua_mt
from . import cps
from . import abd

__all__ = ['urpc', 'mean_teacher', 'ua_mt', 'cps', 'abd']
