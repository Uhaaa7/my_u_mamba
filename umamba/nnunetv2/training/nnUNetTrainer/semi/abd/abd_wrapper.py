"""
ABD 双分支网络包装器

核心机制:
- 包含两个独立的 UMambaSDG 分支 (branch1 和 branch2)
- 两个分支参数独立，不共享权重
- 训练阶段可以分别获取两个分支的输出
- 验证/推理阶段默认返回 branch1 的主输出

与 CPS 的区别:
- ABD 使用 Cross Teaching 机制
- ABD 需要 weak/strong 两种增强视图
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List


class ABDDualBranchWrapper(nn.Module):
    """
    ABD 双分支网络包装器
    
    内部包含两个独立的 UMambaSDG 网络:
    - branch1: 第一个分支 (对应 weak augmentation)
    - branch2: 第二个分支 (对应 strong augmentation)
    
    两个分支完全独立，参数不共享
    """
    
    def __init__(
        self,
        branch1: nn.Module,
        branch2: nn.Module,
        enable_aux_head: bool = True,
        enable_deep_supervision: bool = True
    ):
        """
        Args:
            branch1: 第一个 UMambaSDG 网络
            branch2: 第二个 UMambaSDG 网络
            enable_aux_head: 是否启用了辅助头
            enable_deep_supervision: 是否启用了深度监督
        """
        super().__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.enable_aux_head = enable_aux_head
        self.enable_deep_supervision = enable_deep_supervision
        
        print("🔥🔥🔥 ABD Dual Branch Wrapper 初始化完成! 🔥🔥🔥")
        print(f"📋 enable_aux_head: {enable_aux_head}")
        print(f"📋 enable_deep_supervision: {enable_deep_supervision}")
    
    def forward(
        self,
        x: torch.Tensor,
        branch: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple]:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            branch: 指定分支 ('branch1', 'branch2', None)
                   - None: 训练时返回两个分支的输出，推理时返回 branch1
                   - 'branch1': 只返回 branch1 的输出
                   - 'branch2': 只返回 branch2 的输出
        
        Returns:
            训练模式 + branch=None: (output1, output2)
            其他情况: 单个分支的输出
        """
        if not self.training:
            output1 = self.branch1(x)
            if self.enable_aux_head and isinstance(output1, tuple):
                main_output1, _ = output1
            else:
                main_output1 = output1
            return main_output1
        
        if branch == 'branch1':
            return self.branch1(x)
        elif branch == 'branch2':
            return self.branch2(x)
        else:
            output1 = self.branch1(x)
            output2 = self.branch2(x)
            return output1, output2
    
    def get_branch(self, branch_name: str) -> nn.Module:
        """
        获取指定分支
        
        Args:
            branch_name: 'branch1' 或 'branch2'
            
        Returns:
            对应分支的网络模块
        """
        if branch_name == 'branch1':
            return self.branch1
        elif branch_name == 'branch2':
            return self.branch2
        else:
            raise ValueError(f"Unknown branch: {branch_name}")
