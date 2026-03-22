"""
ABD (Attention-Based Dual-branch) 损失函数

核心损失:
1. 监督损失: CE + Dice
2. 伪标签损失: Cross Teaching
3. ABD-I 损失: 位移后的监督损失
4. ABD-R 损失: 位移后的伪标签损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice 损失
    
    用于 ABD 训练中的分割损失计算
    """
    
    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    
    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        return 1 - loss
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        target: torch.Tensor, 
        softmax: bool = True
    ) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            target: [B, H, W] 标签
            softmax: 是否对 inputs 应用 softmax
            
        Returns:
            dice_loss: 标量损失
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        
        assert inputs.size() == target.size(), f'predict & target shape do not match: {inputs.size()} vs {target.size()}'
        
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        
        return loss / self.n_classes


def compute_supervised_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    ce_loss: nn.Module,
    dice_loss: DiceLoss
) -> torch.Tensor:
    """
    计算监督损失 (CE + Dice)
    
    Args:
        output: [B, C, H, W] logits
        target: [B, H, W] 标签
        ce_loss: CE 损失函数
        dice_loss: Dice 损失函数
        
    Returns:
        loss: 标量损失
    """
    output_soft = torch.softmax(output, dim=1)
    loss_ce = ce_loss(output, target.long())
    loss_dice = dice_loss(output_soft, target, softmax=False)
    return 0.5 * (loss_ce + loss_dice)


def compute_pseudo_supervision_loss(
    output: torch.Tensor,
    pseudo_label: torch.Tensor,
    dice_loss: DiceLoss
) -> torch.Tensor:
    """
    计算伪标签监督损失
    
    Args:
        output: [B, C, H, W] logits
        pseudo_label: [B, H, W] 伪标签
        dice_loss: Dice 损失函数
        
    Returns:
        loss: 标量损失
    """
    output_soft = torch.softmax(output, dim=1)
    return dice_loss(output_soft, pseudo_label, softmax=False)


def compute_abd_loss(
    outputs1_labeled: torch.Tensor,
    outputs2_labeled: torch.Tensor,
    outputs1_unlabeled: torch.Tensor,
    outputs2_unlabeled: torch.Tensor,
    target_labeled: torch.Tensor,
    image_output_supervised_1: torch.Tensor,
    image_output_supervised_2: torch.Tensor,
    label_patch_supervised: torch.Tensor,
    image_output_1: torch.Tensor,
    image_output_2: torch.Tensor,
    pseudo_image_output_1: torch.Tensor,
    pseudo_image_output_2: torch.Tensor,
    labeled_bs: int,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
    consistency_weight: float,
    iteration: int = 0,
    disable_abd_i_after: int = 20000
) -> dict:
    """
    计算 ABD 总损失
    
    Args:
        outputs1_labeled: branch1 有标注样本输出 [B, C, H, W]
        outputs2_labeled: branch2 有标注样本输出 [B, C, H, W]
        outputs1_unlabeled: branch1 无标注样本输出 [B', C, H, W]
        outputs2_unlabeled: branch2 无标注样本输出 [B', C, H, W]
        target_labeled: 有标注样本标签 [B, H, W]
        image_output_supervised_1: ABD-I 后 branch1 输出
        image_output_supervised_2: ABD-I 后 branch2 输出
        label_patch_supervised: ABD-I 后的标签
        image_output_1: ABD-R 后 branch1 输出
        image_output_2: ABD-R 后 branch2 输出
        pseudo_image_output_1: ABD-R 后 branch1 伪标签
        pseudo_image_output_2: ABD-R 后 branch2 伪标签
        labeled_bs: 有标注样本数
        ce_loss: CE 损失函数
        dice_loss: Dice 损失函数
        consistency_weight: 一致性权重
        iteration: 当前迭代次数
        disable_abd_i_after: 在此迭代次数后禁用 ABD-I 损失
        
    Returns:
        dict: 包含各项损失的字典
    """
    outputs_soft1 = torch.softmax(outputs1_labeled, dim=1)
    outputs_soft2 = torch.softmax(outputs2_labeled, dim=1)
    
    pseudo_outputs1 = torch.argmax(outputs_soft1[labeled_bs:].detach(), dim=1)
    pseudo_outputs2 = torch.argmax(outputs_soft2[labeled_bs:].detach(), dim=1)
    
    loss1 = compute_supervised_loss(outputs1_labeled[:labeled_bs], target_labeled[:labeled_bs], ce_loss, dice_loss)
    loss2 = compute_supervised_loss(outputs2_labeled[:labeled_bs], target_labeled[:labeled_bs], ce_loss, dice_loss)
    
    if outputs1_unlabeled.shape[0] > 0:
        pseudo_supervision1 = compute_pseudo_supervision_loss(outputs1_labeled[labeled_bs:], pseudo_outputs2, dice_loss)
        pseudo_supervision2 = compute_pseudo_supervision_loss(outputs2_labeled[labeled_bs:], pseudo_outputs1, dice_loss)
    else:
        pseudo_supervision1 = torch.tensor(0.0, device=outputs1_labeled.device)
        pseudo_supervision2 = torch.tensor(0.0, device=outputs1_labeled.device)
    
    if iteration > disable_abd_i_after:
        loss3 = torch.tensor(0.0, device=outputs1_labeled.device)
        loss4 = torch.tensor(0.0, device=outputs1_labeled.device)
    else:
        loss3 = compute_supervised_loss(image_output_supervised_1, label_patch_supervised, ce_loss, dice_loss)
        loss4 = compute_supervised_loss(image_output_supervised_2, label_patch_supervised, ce_loss, dice_loss)
    
    if image_output_1.shape[0] > 0:
        pseudo_supervision3 = compute_pseudo_supervision_loss(image_output_1, pseudo_image_output_2, dice_loss)
        pseudo_supervision4 = compute_pseudo_supervision_loss(image_output_2, pseudo_image_output_1, dice_loss)
    else:
        pseudo_supervision3 = torch.tensor(0.0, device=outputs1_labeled.device)
        pseudo_supervision4 = torch.tensor(0.0, device=outputs1_labeled.device)
    
    model1_loss = loss1 + 2 * loss3 + consistency_weight * (pseudo_supervision1 + pseudo_supervision3)
    model2_loss = loss2 + 2 * loss4 + consistency_weight * (pseudo_supervision2 + pseudo_supervision4)
    
    total_loss = model1_loss + model2_loss
    
    return {
        'total_loss': total_loss,
        'loss1': loss1,
        'loss2': loss2,
        'loss3': loss3,
        'loss4': loss4,
        'pseudo_supervision1': pseudo_supervision1,
        'pseudo_supervision2': pseudo_supervision2,
        'pseudo_supervision3': pseudo_supervision3,
        'pseudo_supervision4': pseudo_supervision4,
        'model1_loss': model1_loss,
        'model2_loss': model2_loss,
        'consistency_weight': consistency_weight
    }
