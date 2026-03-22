"""
ABD (Attention-Based Dual-branch) 辅助函数

核心机制:
1. extract_main_logits: 统一提取主输出 logits
2. ABD_I: 标注样本上的 patch 位移增强 (用于监督训练)
3. ABD_R: 无标注样本上的置信度引导 patch 位移 (用于伪标签训练)

所有函数适配 2D segmentation
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import random
from typing import Tuple, Optional


def extract_main_logits(output) -> torch.Tensor:
    """
    统一提取主输出 logits
    
    兼容:
    - tensor: 直接返回
    - (main_output, aux_output): 返回 main_output
    - deep supervision list/tuple: 返回第一个元素
    
    Args:
        output: 网络输出
        
    Returns:
        main_logits: [B, C, H, W]
    """
    if isinstance(output, tuple):
        main_output = output[0]
    else:
        main_output = output
    
    if isinstance(main_output, (list, tuple)):
        return main_output[0]
    else:
        return main_output


def get_confidence_map(logits: torch.Tensor) -> torch.Tensor:
    """
    计算置信度图
    
    Args:
        logits: [B, C, H, W]
        
    Returns:
        confidence: [B, H, W] - 每个像素的最大 softmax 概率
    """
    softmax_output = F.softmax(logits, dim=1)
    confidence, _ = torch.max(softmax_output, dim=1)
    return confidence


def ABD_I(
    outputs1_max: torch.Tensor,
    outputs2_max: torch.Tensor,
    volume_batch: torch.Tensor,
    volume_batch_strong: torch.Tensor,
    label_batch: torch.Tensor,
    label_batch_strong: torch.Tensor,
    labeled_bs: int,
    patch_h: int = 64,
    patch_w: int = 56,
    h_size: int = 4,
    w_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ABD-I: 标注样本上的双向 Patch 位移
    
    原理:
    - 基于置信度图找到高/低置信度 patch
    - 将一个增强视图的高置信度 patch 移植到另一个视图的低置信度区域
    - 同时移植对应的标签
    
    Args:
        outputs1_max: branch1 的置信度图 [B, H, W]
        outputs2_max: branch2 的置信度图 [B, H, W]
        volume_batch: weak augmentation 图像 [B, C, H, W]
        volume_batch_strong: strong augmentation 图像 [B, C, H, W]
        label_batch: weak augmentation 标签 [B, H, W]
        label_batch_strong: strong augmentation 标签 [B, H, W]
        labeled_bs: 有标注样本数
        patch_h: patch 高度
        patch_w: patch 宽度
        h_size: patch 网格高度方向数量
        w_size: patch 网格宽度方向数量
        
    Returns:
        image_patch_supervised_last: 位移后的图像 [B, H, W]
        label_patch_supervised_last: 位移后的标签 [B, H, W]
    """
    B, C, H, W = volume_batch.shape
    
    patches_supervised_1 = rearrange(
        outputs1_max[:labeled_bs], 
        'b (h p1) (w p2)->b (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    patches_supervised_2 = rearrange(
        outputs2_max[:labeled_bs], 
        'b (h p1) (w p2)->b (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    
    image_patch_supervised_1 = rearrange(
        volume_batch.squeeze(1)[:labeled_bs], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    image_patch_supervised_2 = rearrange(
        volume_batch_strong.squeeze(1)[:labeled_bs], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    label_patch_supervised_1 = rearrange(
        label_batch[:labeled_bs], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    label_patch_supervised_2 = rearrange(
        label_batch_strong[:labeled_bs], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)
    
    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1)
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1)
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1)
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1)
    
    for i in range(labeled_bs):
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2[i][h[i]]
            image_patch_supervised_1[i][e[i]] = min_patch_supervised_1
            min_patch_supervised_2 = image_patch_supervised_1[i][f[i]]
            image_patch_supervised_2[i][g[i]] = min_patch_supervised_2

            min_label_supervised_1 = label_patch_supervised_2[i][h[i]]
            label_patch_supervised_1[i][e[i]] = min_label_supervised_1
            min_label_supervised_2 = label_patch_supervised_1[i][f[i]]
            label_patch_supervised_2[i][g[i]] = min_label_supervised_2
    
    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(
        image_patch_supervised, 
        'b (h w)(p1 p2) -> b (h p1) (w p2)', 
        h=h_size, w=w_size, p1=patch_h, p2=patch_w
    )
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(
        label_patch_supervised, 
        'b (h w)(p1 p2) -> b (h p1) (w p2)', 
        h=h_size, w=w_size, p1=patch_h, p2=patch_w
    )
    
    return image_patch_supervised_last, label_patch_supervised_last


def ABD_R(
    outputs1_max: torch.Tensor,
    outputs2_max: torch.Tensor,
    volume_batch: torch.Tensor,
    volume_batch_strong: torch.Tensor,
    outputs1_unlabel: torch.Tensor,
    outputs2_unlabel: torch.Tensor,
    labeled_bs: int,
    patch_h: int = 64,
    patch_w: int = 56,
    h_size: int = 4,
    w_size: int = 4,
    top_num: int = 4
) -> torch.Tensor:
    """
    ABD-R: 无标注样本上的置信度引导双向 Patch 位移
    
    原理:
    - 基于置信度图找到高置信度 patch (top-k) 和低置信度 patch
    - 使用 KL 散度找到语义最相似的高/低置信度 patch 对
    - 进行双向 patch 交换，生成新的训练样本
    
    Args:
        outputs1_max: branch1 的置信度图 [B, H, W]
        outputs2_max: branch2 的置信度图 [B, H, W]
        volume_batch: weak augmentation 图像 [B, C, H, W]
        volume_batch_strong: strong augmentation 图像 [B, C, H, W]
        outputs1_unlabel: branch1 的无标注样本 logits [B', C, H, W]
        outputs2_unlabel: branch2 的无标注样本 logits [B', C, H, W]
        labeled_bs: 有标注样本数
        patch_h: patch 高度
        patch_w: patch 宽度
        h_size: patch 网格高度方向数量
        w_size: patch 网格宽度方向数量
        top_num: 选择的高置信度 patch 数量
        
    Returns:
        image_patch_last: 位移后的图像 [B', H, W]
    """
    unlabeled_bs = outputs1_max.shape[0] - labeled_bs
    if unlabeled_bs <= 0:
        return volume_batch.squeeze(1)[labeled_bs:]
    
    patches_1 = rearrange(
        outputs1_max[labeled_bs:], 
        'b (h p1) (w p2)->b (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    patches_2 = rearrange(
        outputs2_max[labeled_bs:], 
        'b (h p1) (w p2)->b (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    
    image_patch_1 = rearrange(
        volume_batch.squeeze(1)[labeled_bs:], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    image_patch_2 = rearrange(
        volume_batch_strong.squeeze(1)[labeled_bs:], 
        'b (h p1) (w p2) -> b (h w)(p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(
        outputs1_unlabel, 
        'b c (h p1) (w p2)->b c (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    patches_outputs_2 = rearrange(
        outputs2_unlabel, 
        'b c (h p1) (w p2)->b c (h w) (p1 p2)', 
        p1=patch_h, p2=patch_w
    )
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)

    patches_mean_1_topk_values, patches_mean_1_topk_indices = patches_mean_1.topk(top_num, dim=1)
    patches_mean_2_topk_values, patches_mean_2_topk_indices = patches_mean_2.topk(top_num, dim=1)
    
    for i in range(unlabeled_bs):
        kl_similarities_1 = torch.empty(top_num)
        kl_similarities_2 = torch.empty(top_num)
        b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]
        patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]
        patches_mean_outputs_topk_1 = patches_mean_outputs_1[i, patches_mean_1_topk_indices[i, :], :]
        patches_mean_outputs_topk_2 = patches_mean_outputs_2[i, patches_mean_2_topk_indices[i, :], :]

        for j in range(top_num):
            kl_similarities_1[j] = F.kl_div(
                patches_mean_outputs_topk_1[j].softmax(dim=-1).log(), 
                patches_mean_outputs_min_2.softmax(dim=-1), 
                reduction='sum'
            )
            kl_similarities_2[j] = F.kl_div(
                patches_mean_outputs_topk_2[j].softmax(dim=-1).log(), 
                patches_mean_outputs_min_1.softmax(dim=-1), 
                reduction='sum'
            )

        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
        a_ori = patches_mean_1_topk_indices[i, a]
        c_ori = patches_mean_2_topk_indices[i, c]

        max_patch_1 = image_patch_2[i][c_ori]
        image_patch_1[i][b] = max_patch_1
        max_patch_2 = image_patch_1[i][a_ori]
        image_patch_2[i][d] = max_patch_2

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(
        image_patch, 
        'b (h w)(p1 p2) -> b (h p1) (w p2)', 
        h=h_size, w=w_size, p1=patch_h, p2=patch_w
    )
    
    return image_patch_last


def get_current_consistency_weight(epoch: int, consistency: float = 0.1, consistency_rampup: float = 200.0) -> float:
    """
    计算当前一致性损失权重 (sigmoid ramp-up)
    
    Args:
        epoch: 当前 epoch
        consistency: 最大一致性权重
        consistency_rampup: ramp-up 周期
        
    Returns:
        当前权重
    """
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """
    Sigmoid ramp-up 函数
    
    Args:
        current: 当前值
        rampup_length: ramp-up 长度
        
    Returns:
        ramp-up 系数 [0, 1]
    """
    if rampup_length == 0:
        return 1.0
    current = min(current, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


import numpy as np
