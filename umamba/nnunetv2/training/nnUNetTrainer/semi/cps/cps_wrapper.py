"""
CPS (Cross Pseudo Supervision) 双分支网络包装器

核心机制:
- 包含两个独立的 UMambaSDG 分支 (branch1 和 branch2)
- 两个分支参数独立，不共享权重
- 训练阶段可以分别获取两个分支的输出
- 验证/推理阶段默认返回 branch1 的主输出

设计原则:
- 完全复用现有 UMambaSDG 构建方式
- 兼容 deep supervision 和 aux head 输出格式
- 最小化对原有训练流程的改动
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List


class CPSDualBranchWrapper(nn.Module):
    """
    CPS 双分支网络包装器
    
    内部包含两个独立的 UMambaSDG 网络:
    - branch1: 第一个分支
    - branch2: 第二个分支
    
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
        
        print("🔥🔥🔥 CPS Dual Branch Wrapper 初始化完成! 🔥🔥🔥")
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
    
    def get_branch_output(
        self,
        x: torch.Tensor,
        branch: str
    ) -> torch.Tensor:
        """
        获取指定分支的主输出 (不含 aux)
        
        Args:
            x: 输入张量
            branch: 'branch1' 或 'branch2'
        
        Returns:
            主输出 logits [B, C, H, W] 或 deep supervision 列表
        """
        if branch == 'branch1':
            output = self.branch1(x)
        elif branch == 'branch2':
            output = self.branch2(x)
        else:
            raise ValueError(f"Invalid branch: {branch}, must be 'branch1' or 'branch2'")
        
        if self.enable_aux_head and isinstance(output, tuple):
            main_output, _ = output
        else:
            main_output = output
        
        return main_output
    
    def get_main_logits(self, output: Union[torch.Tensor, Tuple, List]) -> torch.Tensor:
        """
        从网络输出中提取主 logits
        
        处理以下情况:
        1. 直接返回 tensor [B, C, H, W]
        2. 返回 (main_output, aux_output) 元组
        3. main_output 是 deep supervision 的列表
        
        Args:
            output: 网络输出
        
        Returns:
            主 logits [B, C, H, W]
        """
        if isinstance(output, tuple):
            main_output = output[0]
        else:
            main_output = output
        
        if isinstance(main_output, (list, tuple)):
            return main_output[0]
        else:
            return main_output
    
    def train_mode_for_branch(self, branch: str, mode: bool = True):
        """
        设置指定分支的训练模式
        
        Args:
            branch: 'branch1' 或 'branch2'
            mode: True for training, False for evaluation
        """
        if branch == 'branch1':
            self.branch1.train(mode)
        elif branch == 'branch2':
            self.branch2.train(mode)
        else:
            raise ValueError(f"Invalid branch: {branch}")
    
    def get_num_parameters(self, branch: Optional[str] = None) -> int:
        """
        获取参数数量
        
        Args:
            branch: 'branch1', 'branch2', 或 None (总参数)
        
        Returns:
            参数数量
        """
        if branch == 'branch1':
            return sum(p.numel() for p in self.branch1.parameters())
        elif branch == 'branch2':
            return sum(p.numel() for p in self.branch2.parameters())
        else:
            return sum(p.numel() for p in self.parameters())
