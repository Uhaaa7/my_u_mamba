"""
UA-MT (Uncertainty-Aware Mean Teacher) 工具函数模块

核心功能:
1. EMA 模型参数更新
2. Consistency weight ramp-up
3. Uncertainty estimation (基于熵)
4. Uncertainty mask 生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Optional


def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float, global_step: int):
    """
    更新 EMA teacher 模型参数
    
    公式: ema_param = alpha * ema_param + (1 - alpha) * model_param
    
    UA-MT 论文中建议在训练初期使用较小的 alpha，逐渐增大到目标值
    
    Args:
        model: student 模型
        ema_model: EMA teacher 模型
        alpha: EMA decay 系数 (通常 0.99 或 0.999)
        global_step: 当前全局步数，用于 warmup
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
                ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1 - alpha)
            else:
                ema_buffer.data.copy_(buffer.data)


def sigmoid_rampup(current: int, rampup_length: int) -> float:
    """
    Sigmoid ramp-up 函数
    
    用于逐渐增加 consistency weight
    
    Args:
        current: 当前 epoch 或 step
        rampup_length: ramp-up 的总长度
    
    Returns:
        当前 ramp-up 系数 [0, 1]
    """
    if rampup_length == 0:
        return 1.0
    
    current = np.clip(current, 0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(
    epoch: int, 
    consistency: float, 
    rampup_length: int,
    warmup_epochs: int = 0
) -> float:
    """
    获取当前 consistency weight
    
    Args:
        epoch: 当前 epoch
        consistency: 最大 consistency weight
        rampup_length: ramp-up 长度
        warmup_epochs: 预热 epochs (在此期间 consistency weight 为 0)
    
    Returns:
        当前 consistency weight
    """
    if epoch < warmup_epochs:
        return 0.0
    
    return consistency * sigmoid_rampup(epoch - warmup_epochs, rampup_length)


def compute_entropy_uncertainty(prob: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    基于预测概率计算熵不确定性
    
    熵越高 -> 不确定性越高
    
    Args:
        prob: softmax 概率分布 [B, C, H, W]
        dim: 类别维度
    
    Returns:
        entropy: 熵图 [B, 1, H, W]
    """
    prob = torch.clamp(prob, min=1e-8, max=1.0)
    entropy = -torch.sum(prob * torch.log(prob), dim=dim, keepdim=True)
    return entropy


def normalize_entropy(entropy: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将熵归一化到 [0, 1] 范围
    
    最大熵 = log(num_classes)
    
    Args:
        entropy: 原始熵值
        num_classes: 类别数
    
    Returns:
        归一化后的熵 [0, 1]
    """
    max_entropy = np.log(num_classes)
    return entropy / max_entropy


def generate_uncertainty_mask(
    entropy: torch.Tensor, 
    threshold: float,
    normalize: bool = True,
    num_classes: int = None
) -> torch.Tensor:
    """
    基于熵生成不确定性 mask
    
    低熵区域 (高确定性) -> mask = 1
    高熵区域 (低确定性) -> mask = 0
    
    Args:
        entropy: 熵图 [B, 1, H, W]
        threshold: 不确定性阈值 (归一化后)
        normalize: 是否需要归一化熵
        num_classes: 类别数 (用于归一化)
    
    Returns:
        mask: 确定性 mask [B, 1, H, W]
    """
    if normalize and num_classes is not None:
        entropy = normalize_entropy(entropy, num_classes)
    
    mask = (entropy < threshold).float()
    return mask


def softmax_mse_loss(
    student_output: torch.Tensor, 
    teacher_output: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Softmax MSE Consistency Loss
    
    计算 student 和 teacher softmax 输出之间的 MSE
    
    Args:
        student_output: student logits [B, C, H, W]
        teacher_output: teacher logits [B, C, H, W]
        mask: 可选的像素级 mask [B, 1, H, W]
    
    Returns:
        loss: consistency loss
    """
    student_softmax = F.softmax(student_output, dim=1)
    teacher_softmax = F.softmax(teacher_output, dim=1).detach()
    
    loss = (student_softmax - teacher_softmax) ** 2
    
    if mask is not None:
        loss = loss * mask
        return loss.mean()
    
    return loss.mean()


def softmax_kl_loss(
    student_output: torch.Tensor, 
    teacher_output: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Softmax KL Divergence Consistency Loss
    
    Args:
        student_output: student logits [B, C, H, W]
        teacher_output: teacher logits [B, C, H, W]
        mask: 可选的像素级 mask [B, 1, H, W]
    
    Returns:
        loss: consistency loss
    """
    student_log_softmax = F.log_softmax(student_output, dim=1)
    teacher_softmax = F.softmax(teacher_output, dim=1).detach()
    
    loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='none')
    
    if mask is not None:
        loss = loss * mask
        return loss.mean()
    
    return loss.mean()


class EMAModel:
    """
    EMA 模型包装器
    
    用于创建和维护 teacher 模型的 EMA 副本
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: student 模型
            decay: EMA decay 系数
        """
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def update(self, model: nn.Module, global_step: int = 0):
        """
        更新 EMA 参数
        
        Args:
            model: student 模型
            global_step: 当前全局步数
        """
        update_ema_variables(model, self.model, self.decay, global_step)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def train(self, mode: bool = True):
        pass
    
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class UncertaintyEstimator:
    """
    不确定性估计器
    
    通过多次随机扰动前向传播来估计不确定性
    """
    def __init__(self, num_samples: int = 8, uncertainty_threshold: float = 0.6):
        """
        Args:
            num_samples: teacher 前向采样次数
            uncertainty_threshold: 不确定性阈值
        """
        self.num_samples = num_samples
        self.uncertainty_threshold = uncertainty_threshold
    
    def estimate_uncertainty(
        self, 
        model: nn.Module, 
        input_tensor: torch.Tensor,
        dropout_rate: float = 0.1
    ) -> tuple:
        """
        通过多次随机扰动估计不确定性
        
        实现方式: 在 eval 模式下强制启用 dropout
        
        Args:
            model: teacher 模型
            input_tensor: 输入张量 [B, C, H, W]
            dropout_rate: dropout 概率
        
        Returns:
            mean_pred: 平均预测概率 [B, C, H, W]
            uncertainty: 不确定性图 [B, 1, H, W]
            mask: 确定性 mask [B, 1, H, W]
        """
        model.train()
        
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_rate
                m.train()
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.SyncBatchNorm)):
                m.eval()
        
        preds = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = model(input_tensor)
                
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                pred = F.softmax(output, dim=1)
                preds.append(pred)
        
        model.eval()
        
        preds = torch.stack(preds, dim=0)
        mean_pred = preds.mean(dim=0)
        
        variance = preds.var(dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
        
        num_classes = mean_pred.shape[1]
        entropy = compute_entropy_uncertainty(mean_pred, dim=1)
        normalized_entropy = normalize_entropy(entropy, num_classes)
        
        mask = (normalized_entropy < self.uncertainty_threshold).float()
        
        return mean_pred, uncertainty, mask
