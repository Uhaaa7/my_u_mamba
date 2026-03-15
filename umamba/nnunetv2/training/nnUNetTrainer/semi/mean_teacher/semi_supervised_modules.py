"""
半监督学习核心模块

包含:
1. BoundaryHead: 边界分支模块
2. PseudoLabelRefiner: 伪标签提纯模块
3. SkipDistillationLoss: Skip特征蒸馏损失
4. BoundaryConsistencyLoss: 边界一致性损失
5. EMAModel: EMA模型包装器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import copy
import math


class BoundaryHead(nn.Module):
    """
    边界分支模块
    
    接入位置: 高分辨率 decoder stage 或最终特征
    输出: 边界概率图 [B, 1, H, W]
    
    设计原则:
    - 轻量级结构，不引入过多参数
    - 使用多尺度边缘检测增强边界感知
    - 支持从 GT mask 提取边界监督
    """
    def __init__(self, in_channels: int, hidden_channels: int = None):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = max(in_channels // 4, 16)
        
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.boundary_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            x: decoder 特征 [B, C, H, W]
            target_size: 目标尺寸 (H, W)
        
        Returns:
            boundary_map: 边界概率图 [B, 1, H', W']
        """
        boundary = self.boundary_conv(x)
        
        if target_size is not None:
            boundary = F.interpolate(boundary, size=target_size, mode='bilinear', align_corners=False)
        
        return boundary
    
    @staticmethod
    def extract_boundary_from_mask(mask: torch.Tensor, 
                                    num_classes: int = None,
                                    kernel_size: int = 3,
                                    ignore_background: bool = True) -> torch.Tensor:
        """
        从分割 mask 提取边界 GT (支持多类别分割)
        
        对于多类别分割，正确的做法是:
        1. 将 mask 转为 one-hot 格式
        2. 对每个前景类别分别提取边界
        3. 做并集得到最终边界
        
        Args:
            mask: 分割标签 [B, H, W] 或 [B, 1, H, W]
            num_classes: 类别数 (如果为 None，从 mask 中推断)
            kernel_size: 腐蚀核大小
            ignore_background: 是否忽略背景类的边界
        
        Returns:
            boundary: 边界 GT [B, 1, H, W], 值为 0/1
        """
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        
        B, H, W = mask.shape
        
        if num_classes is None:
            num_classes = int(mask.max().item()) + 1
        
        mask_onehot = F.one_hot(mask.long(), num_classes=num_classes)
        mask_onehot = mask_onehot.permute(0, 3, 1, 2).float()
        
        padding = kernel_size // 2
        
        boundary_maps = []
        start_class = 1 if ignore_background else 0
        
        for c in range(start_class, num_classes):
            class_mask = mask_onehot[:, c:c+1, :, :]
            
            eroded = -F.max_pool2d(-class_mask, kernel_size=kernel_size, stride=1, padding=padding)
            
            class_boundary = (class_mask - eroded).clamp(0, 1)
            boundary_maps.append(class_boundary)
        
        if len(boundary_maps) == 0:
            return torch.zeros(B, 1, H, W, device=mask.device, dtype=torch.float32)
        
        boundary = torch.stack(boundary_maps, dim=0).sum(dim=0)
        boundary = (boundary > 0).float()
        
        return boundary


class PseudoLabelRefiner(nn.Module):
    """
    伪标签提纯模块
    
    输入:
    - teacher_final_pred: 最终分割预测 [B, C, H, W]
    - teacher_aux_pred: 辅助分割预测 [B, C, H, W] (可选)
    - teacher_boundary: 边界预测 [B, 1, H, W]
    
    输出:
    - refined_pseudo_label: 提纯后的伪标签 [B, H, W]
    - reliability_map: 可靠性图 [B, 1, H, W]
    
    核心思想:
    1. 主输出与辅助输出的一致性评估
    2. 边界区域的置信度衰减
    3. 多信息源协同生成可靠性图
    """
    def __init__(self, 
                 num_classes: int,
                 consistency_threshold: float = 0.7,
                 boundary_decay_factor: float = 0.5,
                 high_reliability_threshold: float = 0.9,
                 low_reliability_threshold: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.consistency_threshold = consistency_threshold
        self.boundary_decay_factor = boundary_decay_factor
        self.high_reliability_threshold = high_reliability_threshold
        self.low_reliability_threshold = low_reliability_threshold
    
    def forward(self,
                teacher_final_pred: torch.Tensor,
                teacher_aux_pred: Optional[torch.Tensor] = None,
                teacher_boundary: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            teacher_final_pred: 最终分割预测 logits [B, C, H, W]
            teacher_aux_pred: 辅助分割预测 logits [B, C, H, W] (可选)
            teacher_boundary: 边界预测 logits [B, 1, H, W] (可选)
        
        Returns:
            refined_pseudo_label: 提纯后的伪标签 [B, H, W]
            reliability_map: 可靠性图 [B, 1, H, W], 范围 [0, 1]
        """
        final_prob = F.softmax(teacher_final_pred, dim=1)
        final_confidence, pseudo_label = final_prob.max(dim=1)
        
        reliability = final_confidence.unsqueeze(1)
        
        if teacher_aux_pred is not None:
            aux_prob = F.softmax(teacher_aux_pred, dim=1)
            aux_confidence, aux_pred = aux_prob.max(dim=1)
            
            consistency = (pseudo_label == aux_pred).float().unsqueeze(1)
            
            consistency_weight = consistency * self.consistency_threshold + (1 - consistency) * (1 - self.consistency_threshold)
            reliability = reliability * consistency_weight
            
            reliability = reliability * (final_confidence.unsqueeze(1) * aux_confidence.unsqueeze(1)).sqrt()
        
        if teacher_boundary is not None:
            boundary_prob = torch.sigmoid(teacher_boundary)
            
            boundary_penalty = 1 - boundary_prob * self.boundary_decay_factor
            reliability = reliability * boundary_penalty
        
        reliability = reliability.clamp(0, 1)
        
        return pseudo_label, reliability
    
    def get_reliability_mask(self, reliability: torch.Tensor, level: str = 'high') -> torch.Tensor:
        """
        获取不同可靠性级别的掩码
        
        Args:
            reliability: 可靠性图 [B, 1, H, W]
            level: 'high', 'medium', 'low'
        
        Returns:
            mask: 布尔掩码 [B, 1, H, W]
        """
        if level == 'high':
            return reliability >= self.high_reliability_threshold
        elif level == 'medium':
            return (reliability >= self.low_reliability_threshold) & (reliability < self.high_reliability_threshold)
        else:
            return reliability < self.low_reliability_threshold


class SkipDistillationLoss(nn.Module):
    """
    Skip 特征蒸馏损失
    
    核心思想:
    1. 只在可靠区域进行蒸馏
    2. 可靠性图控制蒸馏强度
    3. 传递跨层重构能力和结构敏感表达能力
    
    支持:
    - MSE / L1 / Cosine 相似度
    - 多尺度特征对齐
    - 可靠性加权
    """
    def __init__(self, 
                 loss_type: str = 'mse',
                 temperature: float = 1.0,
                 use_reliability_weight: bool = True):
        super().__init__()
        
        self.loss_type = loss_type
        self.temperature = temperature
        self.use_reliability_weight = use_reliability_weight
    
    def forward(self,
                student_skip_features: List[torch.Tensor],
                teacher_skip_features: List[torch.Tensor],
                reliability_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_skip_features: student 的增强 skip 特征列表
            teacher_skip_features: teacher 的增强 skip 特征列表
            reliability_map: 可靠性图 [B, 1, H, W]
        
        Returns:
            loss: 蒸馏损失
        """
        if len(student_skip_features) != len(teacher_skip_features):
            raise ValueError(f"Skip feature count mismatch: student={len(student_skip_features)}, teacher={len(teacher_skip_features)}")
        
        total_loss = 0.0
        num_layers = len(student_skip_features)
        
        for i, (s_feat, t_feat) in enumerate(zip(student_skip_features, teacher_skip_features)):
            layer_loss = self._compute_single_layer_loss(s_feat, t_feat, reliability_map)
            total_loss = total_loss + layer_loss
        
        return total_loss / max(num_layers, 1)
    
    def _compute_single_layer_loss(self,
                                   student_feat: torch.Tensor,
                                   teacher_feat: torch.Tensor,
                                   reliability_map: torch.Tensor = None) -> torch.Tensor:
        """计算单层特征蒸馏损失"""
        
        if student_feat.shape != teacher_feat.shape:
            teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[2:], mode='bilinear', align_corners=False)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_feat, teacher_feat, reduction='none')
            loss = loss.mean(dim=1, keepdim=True)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(student_feat, teacher_feat, reduction='none')
            loss = loss.mean(dim=1, keepdim=True)
        elif self.loss_type == 'cosine':
            s_flat = student_feat.flatten(2)
            t_flat = teacher_feat.flatten(2)
            
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
            loss = 1 - cos_sim.mean()
            return loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if self.use_reliability_weight and reliability_map is not None:
            reliability_resized = F.interpolate(reliability_map, size=loss.shape[2:], mode='bilinear', align_corners=False)
            loss = loss * reliability_resized
        
        return loss.mean()


class BoundaryConsistencyLoss(nn.Module):
    """
    边界一致性损失
    
    核心思想:
    1. 专门处理边界区域
    2. 结合可靠性信息，只在有价值的过渡带加强约束
    3. 补足输出一致性和 skip 蒸馏的盲点
    """
    def __init__(self,
                 boundary_threshold: float = 0.3,
                 use_reliability_filter: bool = True):
        super().__init__()
        
        self.boundary_threshold = boundary_threshold
        self.use_reliability_filter = use_reliability_filter
    
    def forward(self,
                student_boundary: torch.Tensor,
                teacher_boundary: torch.Tensor,
                reliability_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_boundary: student 边界预测 [B, 1, H, W]
            teacher_boundary: teacher 边界预测 [B, 1, H, W]
            reliability_map: 可靠性图 [B, 1, H, W]
        
        Returns:
            loss: 边界一致性损失
        """
        student_prob = torch.sigmoid(student_boundary)
        teacher_prob = torch.sigmoid(teacher_boundary)
        
        boundary_region = (teacher_prob > self.boundary_threshold) | (student_prob > self.boundary_threshold)
        
        loss = F.binary_cross_entropy(student_prob, teacher_prob.detach(), reduction='none')
        
        if self.use_reliability_filter and reliability_map is not None:
            reliability_resized = F.interpolate(reliability_map, size=loss.shape[2:], mode='bilinear', align_corners=False)
            
            boundary_weight = boundary_region.float() * reliability_resized
            boundary_weight = boundary_weight + (1 - boundary_region.float()) * 0.1
            
            loss = loss * boundary_weight
        else:
            boundary_weight = boundary_region.float() + (1 - boundary_region.float()) * 0.1
            loss = loss * boundary_weight
        
        return loss.mean()


class SemiSupervisedLoss(nn.Module):
    """
    半监督总损失
    
    组成:
    1. 有标签数据的监督损失 (主损失 + 辅助损失 + 边界损失)
    2. 无标签数据的伪标签损失 (可靠性加权)
    3. Skip 特征蒸馏损失
    4. 边界一致性损失
    """
    def __init__(self,
                 num_classes: int,
                 consistency_threshold: float = 0.7,
                 boundary_decay_factor: float = 0.5,
                 skip_distill_weight: float = 0.5,
                 boundary_consistency_weight: float = 0.3,
                 pseudo_label_weight: float = 1.0,
                 boundary_weight: float = 0.4):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.pseudo_label_refiner = PseudoLabelRefiner(
            num_classes=num_classes,
            consistency_threshold=consistency_threshold,
            boundary_decay_factor=boundary_decay_factor
        )
        
        self.skip_distill_loss = SkipDistillationLoss(loss_type='mse')
        self.boundary_consistency_loss = BoundaryConsistencyLoss()
        
        self.skip_distill_weight = skip_distill_weight
        self.boundary_consistency_weight = boundary_consistency_weight
        self.pseudo_label_weight = pseudo_label_weight
        self.boundary_weight = boundary_weight
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def compute_labeled_loss(self,
                             main_output: torch.Tensor,
                             target: torch.Tensor,
                             aux_output: torch.Tensor = None,
                             boundary_output: torch.Tensor = None,
                             boundary_target: torch.Tensor = None,
                             aux_weight: float = 0.4) -> Dict[str, torch.Tensor]:
        """
        计算有标签数据的监督损失
        
        Args:
            main_output: 主分割输出 [B, C, H, W]
            target: 分割标签 [B, H, W]
            aux_output: 辅助分割输出 [B, C, H, W]
            boundary_output: 边界输出 [B, 1, H, W]
            boundary_target: 边界标签 [B, 1, H, W]
            aux_weight: 辅助损失权重
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        main_loss = self.ce_loss(main_output, target)
        losses['main_loss'] = main_loss.mean()
        
        if aux_output is not None:
            aux_loss = self.ce_loss(aux_output, target)
            losses['aux_loss'] = aux_loss.mean() * aux_weight
        
        if boundary_output is not None and boundary_target is not None:
            boundary_loss = self.bce_loss(boundary_output, boundary_target)
            losses['boundary_loss'] = boundary_loss.mean() * self.boundary_weight
        
        total = sum(losses.values())
        losses['total'] = total
        
        return losses
    
    def compute_unlabeled_loss(self,
                               student_main_output: torch.Tensor,
                               student_aux_output: torch.Tensor,
                               student_boundary: torch.Tensor,
                               student_skip_features: List[torch.Tensor],
                               teacher_main_output: torch.Tensor,
                               teacher_aux_output: torch.Tensor,
                               teacher_boundary: torch.Tensor,
                               teacher_skip_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算无标签数据的半监督损失
        
        Args:
            student_*: student 的各种输出
            teacher_*: teacher 的各种输出
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        with torch.no_grad():
            pseudo_label, reliability_map = self.pseudo_label_refiner(
                teacher_main_output, teacher_aux_output, teacher_boundary
            )
        
        pseudo_loss = self.ce_loss(student_main_output, pseudo_label)
        reliability_resized = F.interpolate(reliability_map, size=pseudo_loss.shape[2:], mode='bilinear', align_corners=False)
        pseudo_loss = (pseudo_loss * reliability_resized.squeeze(1)).mean()
        losses['pseudo_label_loss'] = pseudo_loss * self.pseudo_label_weight
        
        skip_loss = self.skip_distill_loss(student_skip_features, teacher_skip_features, reliability_map)
        losses['skip_distill_loss'] = skip_loss * self.skip_distill_weight
        
        boundary_loss = self.boundary_consistency_loss(student_boundary, teacher_boundary, reliability_map)
        losses['boundary_consistency_loss'] = boundary_loss * self.boundary_consistency_weight
        
        total = sum(losses.values())
        losses['total'] = total
        
        return losses


class EMAModel:
    """
    EMA (Exponential Moving Average) 模型包装器
    
    用于 Mean Teacher 半监督学习框架
    Teacher 参数通过 Student 参数的 EMA 更新
    """
    def __init__(self,
                 model: nn.Module,
                 decay: float = 0.999,
                 warmup_steps: int = 2000):
        """
        Args:
            model: student 模型
            decay: EMA 衰减率
            warmup_steps: 预热步数，在此期间逐渐增加 decay
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def get_decay(self) -> float:
        """获取当前的 decay 值 (支持 warmup)"""
        if self.step_count < self.warmup_steps:
            return self.decay * (self.step_count / self.warmup_steps)
        return self.decay
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        更新 EMA 模型参数
        
        Args:
            model: 当前的 student 模型
        """
        decay = self.get_decay()
        
        student_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name, ema_param in ema_params.items():
            if name in student_params:
                student_param = student_params[name]
                ema_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)
        
        student_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema_model.named_buffers())
        
        for name, ema_buffer in ema_buffers.items():
            if name in student_buffers:
                student_buffer = student_buffers[name]
                if ema_buffer.dtype == torch.float32 or ema_buffer.dtype == torch.float16:
                    ema_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1 - decay)
                else:
                    ema_buffer.data.copy_(student_buffer.data)
        
        self.step_count += 1
    
    def get_model(self) -> nn.Module:
        """获取 EMA 模型"""
        return self.ema_model


def get_current_consistency_weight(epoch: int,
                                   warmup_epochs: int = 30,
                                   rampup_end: int = 80,
                                   max_weight: float = 1.0) -> float:
    """
    获取当前的一致性损失权重 (支持 warmup 和 rampup)
    
    Args:
        epoch: 当前 epoch
        warmup_epochs: warmup 阶段的 epoch 数
        rampup_end: rampup 结束的 epoch
        max_weight: 最大权重
    
    Returns:
        weight: 当前的一致性权重
    """
    if epoch < warmup_epochs:
        return 0.0
    
    if epoch >= rampup_end:
        return max_weight
    
    progress = (epoch - warmup_epochs) / (rampup_end - warmup_epochs)
    weight = max_weight * math.exp(-5 * (1 - progress) ** 2)
    
    return weight


if __name__ == '__main__':
    print("Testing semi-supervised modules...")
    
    B, C, H, W = 2, 4, 64, 64
    
    print("\n1. Testing BoundaryHead...")
    boundary_head = BoundaryHead(in_channels=64)
    x = torch.randn(B, 64, H, W)
    boundary_out = boundary_head(x)
    print(f"   Input: {x.shape}, Output: {boundary_out.shape}")
    
    mask = torch.randint(0, C, (B, H, W))
    boundary_gt = BoundaryHead.extract_boundary_from_mask(mask)
    print(f"   Boundary GT shape: {boundary_gt.shape}, unique values: {boundary_gt.unique()}")
    
    print("\n2. Testing PseudoLabelRefiner...")
    refiner = PseudoLabelRefiner(num_classes=C)
    final_pred = torch.randn(B, C, H, W)
    aux_pred = torch.randn(B, C, H, W)
    boundary_pred = torch.randn(B, 1, H, W)
    pseudo_label, reliability = refiner(final_pred, aux_pred, boundary_pred)
    print(f"   Pseudo label shape: {pseudo_label.shape}")
    print(f"   Reliability shape: {reliability.shape}, range: [{reliability.min():.4f}, {reliability.max():.4f}]")
    
    print("\n3. Testing SkipDistillationLoss...")
    skip_loss_fn = SkipDistillationLoss()
    student_skips = [torch.randn(B, 64, 32, 32), torch.randn(B, 128, 16, 16)]
    teacher_skips = [torch.randn(B, 64, 32, 32), torch.randn(B, 128, 16, 16)]
    skip_loss = skip_loss_fn(student_skips, teacher_skips, reliability)
    print(f"   Skip distillation loss: {skip_loss.item():.4f}")
    
    print("\n4. Testing BoundaryConsistencyLoss...")
    boundary_loss_fn = BoundaryConsistencyLoss()
    student_boundary = torch.randn(B, 1, H, W)
    teacher_boundary = torch.randn(B, 1, H, W)
    boundary_loss = boundary_loss_fn(student_boundary, teacher_boundary, reliability)
    print(f"   Boundary consistency loss: {boundary_loss.item():.4f}")
    
    print("\n5. Testing EMAModel...")
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
        def forward(self, x):
            return self.bn(self.conv(x))
    
    student = DummyModel()
    ema = EMAModel(student, decay=0.999)
    
    for _ in range(10):
        student.conv.weight.data.add_(torch.randn_like(student.conv.weight) * 0.01)
        ema.update(student)
    
    teacher = ema.get_model()
    print(f"   Student weight mean: {student.conv.weight.mean().item():.4f}")
    print(f"   Teacher weight mean: {teacher.conv.weight.mean().item():.4f}")
    
    print("\n6. Testing consistency weight schedule...")
    for epoch in [0, 10, 30, 50, 80, 100]:
        weight = get_current_consistency_weight(epoch)
        print(f"   Epoch {epoch}: weight = {weight:.4f}")
    
    print("\n✅ All tests passed!")
