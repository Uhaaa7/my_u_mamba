"""
CPS (Cross Pseudo Supervision) 损失函数

核心机制:
- 每个分支用对方分支的伪标签进行监督
- 伪标签为 argmax 得到的 hard pseudo label
- loss = CE(pred_l, pseudo_r) + CE(pred_r, pseudo_l)

设计原则:
- 兼容 UMambaSDG 的多种输出格式
- 支持 ignore label 处理
- 支持类别数配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Optional


def extract_main_logits(output: Union[torch.Tensor, Tuple, List]) -> torch.Tensor:
    """
    从网络输出中提取主 logits
    
    处理以下情况:
    1. 直接返回 tensor [B, C, H, W]
    2. 返回 (main_output, aux_output) 元组
    3. main_output 是 deep supervision 的列表 [scale0, scale1, ...]
    
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


def compute_cps_loss(
    pred_l: torch.Tensor,
    pred_r: torch.Tensor,
    ce_loss_fn: nn.Module,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算 CPS (Cross Pseudo Supervision) 损失
    
    逻辑:
    1. 用 branch2 的预测生成伪标签，监督 branch1
    2. 用 branch1 的预测生成伪标签，监督 branch2
    3. 总损失 = CE(pred_l, pseudo_r) + CE(pred_r, pseudo_l)
    
    Args:
        pred_l: branch1 的主 logits [B, C, H, W]
        pred_r: branch2 的主 logits [B, C, H, W]
        ce_loss_fn: 交叉熵损失函数
        ignore_index: 忽略的标签索引
    
    Returns:
        CPS 损失值
    """
    with torch.no_grad():
        pseudo_r = pred_r.detach().argmax(dim=1)
        pseudo_l = pred_l.detach().argmax(dim=1)
    
    loss_l_to_r = ce_loss_fn(pred_l, pseudo_r)
    loss_r_to_l = ce_loss_fn(pred_r, pseudo_l)
    
    cps_loss = loss_l_to_r + loss_r_to_l
    
    return cps_loss


class CPSLoss(nn.Module):
    """
    CPS 损失模块
    
    封装了 CPS 损失计算逻辑，支持:
    - 自动提取主 logits
    - 配置 ignore label
    - 配置损失权重
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Args:
            ignore_index: 忽略的标签索引
            reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    def forward(
        self,
        output_l: Union[torch.Tensor, Tuple, List],
        output_r: Union[torch.Tensor, Tuple, List]
    ) -> torch.Tensor:
        """
        计算 CPS 损失
        
        Args:
            output_l: branch1 的网络输出 (可能包含 aux 和 deep supervision)
            output_r: branch2 的网络输出 (可能包含 aux 和 deep supervision)
        
        Returns:
            CPS 损失值
        """
        pred_l = extract_main_logits(output_l)
        pred_r = extract_main_logits(output_r)
        
        return compute_cps_loss(pred_l, pred_r, self.ce_loss, self.ignore_index)


class CPSLossWithConfidence(nn.Module):
    """
    带置信度过滤的 CPS 损失
    
    只在高置信度区域计算 CPS 损失
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        confidence_threshold: float = 0.9,
        reduction: str = 'mean'
    ):
        """
        Args:
            ignore_index: 忽略的标签索引
            confidence_threshold: 置信度阈值
            reduction: 损失归约方式
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction
    
    def forward(
        self,
        output_l: Union[torch.Tensor, Tuple, List],
        output_r: Union[torch.Tensor, Tuple, List]
    ) -> torch.Tensor:
        """
        计算带置信度过滤的 CPS 损失
        
        Args:
            output_l: branch1 的网络输出
            output_r: branch2 的网络输出
        
        Returns:
            CPS 损失值
        """
        pred_l = extract_main_logits(output_l)
        pred_r = extract_main_logits(output_r)
        
        with torch.no_grad():
            prob_r = F.softmax(pred_r.detach(), dim=1)
            prob_l = F.softmax(pred_l.detach(), dim=1)
            
            max_prob_r, pseudo_r = prob_r.max(dim=1)
            max_prob_l, pseudo_l = prob_l.max(dim=1)
            
            confidence_mask_r = (max_prob_r >= self.confidence_threshold).float()
            confidence_mask_l = (max_prob_l >= self.confidence_threshold).float()
        
        loss_l_to_r = F.cross_entropy(
            pred_l, pseudo_r,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        loss_r_to_l = F.cross_entropy(
            pred_r, pseudo_l,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        if loss_l_to_r.dim() == 3:
            loss_l_to_r = loss_l_to_r.mean(dim=[1, 2])
            loss_r_to_l = loss_r_to_l.mean(dim=[1, 2])
            confidence_mask_r = confidence_mask_r.mean(dim=[1, 2])
            confidence_mask_l = confidence_mask_l.mean(dim=[1, 2])
        
        loss_l_to_r = (loss_l_to_r * confidence_mask_r).sum() / (confidence_mask_r.sum() + 1e-8)
        loss_r_to_l = (loss_r_to_l * confidence_mask_l).sum() / (confidence_mask_l.sum() + 1e-8)
        
        return loss_l_to_r + loss_r_to_l
