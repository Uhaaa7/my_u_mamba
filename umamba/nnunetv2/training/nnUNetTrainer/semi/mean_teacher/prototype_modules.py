"""
Prototype-based Semi-supervised Learning Modules

核心组件:
1. ProjectionHead: 单阶段投影头，将 skip 特征映射到 embedding space
2. MultiStageProjectionHeads: 多阶段投影头管理器
3. PrototypeBank: 分层、分类别、多原型库 (core + transition prototypes)
4. PrototypeLoss: 包含 PGPC、APC、PAC 三种损失

设计原则:
- 每层 enhanced skip 都有自己的一套 prototype bank
- 每个类别在每一层都有多个 prototypes (core + transition)
- 原型通过 momentum update 更新，不是 learnable parameters
- 更新权重受 reliability 和 boundary 双重控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import math


class ProjectionHead(nn.Module):
    """
    单阶段投影头
    
    将 enhanced skip 特征映射到 embedding space
    
    结构: Conv3x3 -> ReLU -> Conv1x1
    """
    def __init__(self, in_channels: int, projection_dim: int = 128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, projection_dim, kernel_size=1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: skip 特征 [B, C_in, H, W]
        
        Returns:
            projection: embedding [B, projection_dim, H, W]
        """
        return self.projection(x)


class MultiStageProjectionHeads(nn.Module):
    """
    多阶段投影头管理器
    
    为每个 enhanced skip stage 创建独立的 projection head
    """
    def __init__(self, 
                 input_dims: List[int], 
                 projection_dim: int = 128):
        """
        Args:
            input_dims: 每个 skip stage 的通道数列表
            projection_dim: 投影维度
        """
        super().__init__()
        
        self.num_stages = len(input_dims)
        self.projection_dim = projection_dim
        
        self.projection_heads = nn.ModuleList([
            ProjectionHead(dim, projection_dim) for dim in input_dims
        ])
        
        print(f"🔧 MultiStageProjectionHeads initialized:")
        print(f"   - Num stages: {self.num_stages}")
        print(f"   - Input dims: {input_dims}")
        print(f"   - Projection dim: {projection_dim}")
    
    def forward(self, skip_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            skip_features: 每个 stage 的 skip 特征列表
        
        Returns:
            projections: 每个 stage 的 embedding 列表
        """
        projections = []
        for i, feat in enumerate(skip_features):
            if i < len(self.projection_heads):
                proj = self.projection_heads[i](feat)
                projections.append(proj)
        return projections


class PrototypeBank:
    """
    分层、分类别、多原型库
    
    核心设计:
    1. 每层 enhanced skip 都有自己的一套 prototype bank
    2. 每个类别在每一层都有多个 prototypes
    3. 分为 core prototypes (内部区) 和 transition prototypes (过渡区)
    4. 通过 momentum update 更新，不是 learnable parameters
    
    原型更新逻辑:
    - 更新 core prototypes 时，偏向高可靠、低边界概率区域
    - 更新 transition prototypes 时，偏向高可靠、边界概率高的区域
    - 低可靠区域默认不更新
    """
    
    def __init__(self,
                 num_stages: int,
                 num_classes: int,
                 num_prototypes_per_class: int = 5,
                 projection_dim: int = 128,
                 momentum: float = 0.9,
                 device: torch.device = torch.device('cuda')):
        """
        Args:
            num_stages: skip stage 数量
            num_classes: 类别数 (不含背景)
            num_prototypes_per_class: 每个类别的原型数 (core + transition 各一半)
            projection_dim: 投影维度
            momentum: 动量更新系数
            device: 设备
        """
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.projection_dim = projection_dim
        self.momentum = momentum
        self.device = device
        
        num_core = max(1, num_prototypes_per_class // 2)
        num_transition = max(1, num_prototypes_per_class - num_core)
        self.num_core_prototypes = num_core
        self.num_transition_prototypes = num_transition
        
        self.core_prototypes = []
        self.transition_prototypes = []
        self.core_update_counts = []
        self.transition_update_counts = []
        
        for stage in range(num_stages):
            stage_core = []
            stage_transition = []
            stage_core_counts = []
            stage_transition_counts = []
            
            for c in range(num_classes):
                core_proto = torch.randn(num_core, projection_dim, device=device)
                core_proto = F.normalize(core_proto, dim=1)
                stage_core.append(core_proto)
                stage_core_counts.append(torch.zeros(num_core, device=device))
                
                trans_proto = torch.randn(num_transition, projection_dim, device=device)
                trans_proto = F.normalize(trans_proto, dim=1)
                stage_transition.append(trans_proto)
                stage_transition_counts.append(torch.zeros(num_transition, device=device))
            
            self.core_prototypes.append(stage_core)
            self.transition_prototypes.append(stage_transition)
            self.core_update_counts.append(stage_core_counts)
            self.transition_update_counts.append(stage_transition_counts)
        
        print(f"🔧 PrototypeBank initialized:")
        print(f"   - Num stages: {num_stages}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Prototypes per class: {num_prototypes_per_class} (core={num_core}, transition={num_transition})")
        print(f"   - Projection dim: {projection_dim}")
        print(f"   - Momentum: {momentum}")
    
    def get_prototypes(self, stage: int, class_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定 stage 和 class 的原型
        
        Args:
            stage: stage 索引
            class_idx: 类别索引
        
        Returns:
            core_prototypes: [num_core, projection_dim]
            transition_prototypes: [num_transition, projection_dim]
        """
        return self.core_prototypes[stage][class_idx], self.transition_prototypes[stage][class_idx]
    
    def get_all_prototypes(self, stage: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取指定 stage 的所有原型
        
        Args:
            stage: stage 索引
        
        Returns:
            core_prototypes: List of [num_core, projection_dim] for each class
            transition_prototypes: List of [num_transition, projection_dim] for each class
        """
        return self.core_prototypes[stage], self.transition_prototypes[stage]
    
    @torch.no_grad()
    def update(self,
               projections: List[torch.Tensor],
               labels: torch.Tensor,
               reliability_map: torch.Tensor,
               boundary_map: torch.Tensor,
               high_reliability_threshold: float = 0.75,
               boundary_threshold: float = 0.3):
        """
        更新原型库
        
        更新逻辑:
        - 高可靠、低边界概率 -> 更新 core prototypes
        - 高可靠、高边界概率 -> 更新 transition prototypes
        - 低可靠区域不更新
        
        Args:
            projections: 每个 stage 的 embedding 列表 [B, projection_dim, H, W]
            labels: 分割标签 [B, H, W] 或 [B, 1, H, W]
            reliability_map: 可靠性图 [B, 1, H, W]
            boundary_map: 边界图 [B, 1, H, W]
            high_reliability_threshold: 高可靠性阈值
            boundary_threshold: 边界阈值
        """
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        
        for stage, proj in enumerate(projections):
            if stage >= self.num_stages:
                continue
                
            B, C, H, W = proj.shape
            
            proj_flat = proj.permute(0, 2, 3, 1).reshape(-1, C)
            
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            labels_flat = labels_resized.reshape(-1)
            
            reliability_flat = F.interpolate(reliability_map, size=(H, W), mode='bilinear', align_corners=False)
            reliability_flat = reliability_flat.reshape(-1)
            boundary_flat = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)
            boundary_flat = boundary_flat.reshape(-1)
            
            high_reliable_mask = reliability_flat >= high_reliability_threshold
            core_region_mask = boundary_flat < boundary_threshold
            transition_region_mask = boundary_flat >= boundary_threshold
            
            core_mask = high_reliable_mask & core_region_mask
            transition_mask = high_reliable_mask & transition_region_mask
            
            for c in range(self.num_classes):
                class_mask = labels_flat == c
                
                core_class_mask = core_mask & class_mask
                if core_class_mask.sum() > 0:
                    class_features = proj_flat[core_class_mask]
                    self._update_prototypes_momentum(
                        stage, c, class_features, is_core=True
                    )
                
                trans_class_mask = transition_mask & class_mask
                if trans_class_mask.sum() > 0:
                    class_features = proj_flat[trans_class_mask]
                    self._update_prototypes_momentum(
                        stage, c, class_features, is_core=False
                    )
    
    def _update_prototypes_momentum(self,
                                    stage: int,
                                    class_idx: int,
                                    features: torch.Tensor,
                                    is_core: bool):
        """
        使用动量更新原型
        
        Args:
            stage: stage 索引
            class_idx: 类别索引
            features: 用于更新的特征 [N, projection_dim]
            is_core: 是否更新 core prototypes
        """
        if is_core:
            prototypes = self.core_prototypes[stage][class_idx]
            update_counts = self.core_update_counts[stage][class_idx]
            num_protos = self.num_core_prototypes
        else:
            prototypes = self.transition_prototypes[stage][class_idx]
            update_counts = self.transition_update_counts[stage][class_idx]
            num_protos = self.num_transition_prototypes
        
        features = F.normalize(features, dim=1)
        
        similarities = torch.mm(features, prototypes.t())
        assignments = similarities.argmax(dim=1)
        
        for p_idx in range(num_protos):
            mask = assignments == p_idx
            if mask.sum() > 0:
                cluster_features = features[mask]
                cluster_mean = cluster_features.mean(dim=0)
                cluster_mean = F.normalize(cluster_mean, dim=0)
                
                prototypes[p_idx] = self.momentum * prototypes[p_idx] + (1 - self.momentum) * cluster_mean
                prototypes[p_idx] = F.normalize(prototypes[p_idx], dim=0)
                
                update_counts[p_idx] += mask.sum().item()
    
    @torch.no_grad()
    def initialize_from_labeled_data(self,
                                     projections: List[torch.Tensor],
                                     labels: torch.Tensor,
                                     boundary_map: torch.Tensor):
        """
        使用有标签数据初始化原型库
        
        这一步非常重要，让原型空间从训练一开始就有明确语义骨架
        
        Args:
            projections: 每个 stage 的 embedding 列表
            labels: GT 分割标签 [B, H, W]
            boundary_map: GT 边界图 [B, 1, H, W]
        """
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        
        print("🔧 Initializing prototype bank from labeled data...")
        
        for stage, proj in enumerate(projections):
            if stage >= self.num_stages:
                continue
                
            B, C, H, W = proj.shape
            
            proj_flat = proj.permute(0, 2, 3, 1).reshape(-1, C)
            
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            labels_flat = labels_resized.reshape(-1)
            
            boundary_flat = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)
            boundary_flat = boundary_flat.reshape(-1)
            
            core_region_mask = boundary_flat < 0.3
            transition_region_mask = boundary_flat >= 0.3
            
            for c in range(self.num_classes):
                class_mask = labels_flat == c
                
                core_class_mask = core_region_mask & class_mask
                if core_class_mask.sum() > 0:
                    class_features = proj_flat[core_class_mask]
                    class_features = F.normalize(class_features, dim=1)
                    
                    num_samples = min(class_features.shape[0], self.num_core_prototypes)
                    if num_samples > 0:
                        indices = torch.randperm(class_features.shape[0], device=class_features.device)[:num_samples]
                        sampled_features = class_features[indices]
                        new_proto = self.core_prototypes[stage][c].clone()
                        new_proto[:num_samples] = sampled_features
                        self.core_prototypes[stage][c] = new_proto
                
                trans_class_mask = transition_region_mask & class_mask
                if trans_class_mask.sum() > 0:
                    class_features = proj_flat[trans_class_mask]
                    class_features = F.normalize(class_features, dim=1)
                    
                    num_samples = min(class_features.shape[0], self.num_transition_prototypes)
                    if num_samples > 0:
                        indices = torch.randperm(class_features.shape[0], device=class_features.device)[:num_samples]
                        sampled_features = class_features[indices]
                        new_proto = self.transition_prototypes[stage][c].clone()
                        new_proto[:num_samples] = sampled_features
                        self.transition_prototypes[stage][c] = new_proto
        
        print("✅ Prototype bank initialized!")


class PrototypeLoss(nn.Module):
    """
    Prototype-based 损失函数
    
    包含三种损失:
    1. PGPC (Prototype-Guided Pixel Classification): 通过 nearest-prototype 进行像素分类
    2. APC (Adaptive Prototype Contrast): 边界感知的自适应原型对比损失
    3. PAC (Prototype Assignment Consistency): 原型分配一致性
    
    所有损失都支持可靠性分级监督
    """
    
    def __init__(self,
                 prototype_bank: PrototypeBank,
                 num_classes: int,
                 temperature: float = 0.1):
        """
        Args:
            prototype_bank: 原型库
            num_classes: 类别数
            temperature: softmax 温度
        """
        super().__init__()
        
        self.prototype_bank = prototype_bank
        self.num_classes = num_classes
        self.temperature = temperature
    
    def pgpc_loss(self,
                  student_projections: List[torch.Tensor],
                  pseudo_labels: torch.Tensor,
                  reliability_map: torch.Tensor,
                  high_reliability_threshold: float = 0.75,
                  low_reliability_threshold: float = 0.45) -> torch.Tensor:
        """
        Prototype-Guided Pixel Classification Loss
        
        核心思想:
        - 通过 nearest-prototype 的方式，把像素分类从纯 logits 监督扩展到 feature space
        - 对每个像素 embedding，计算它与所有类别 prototypes 的相似度
        - 对每个类别，取与该类别多个 prototype 中最大的相似度
        - 形成 prototype-induced class score，再做 softmax
        
        可靠性分级:
        - 高可靠区域: hard PGPC (直接用 pseudo label 做 CE)
        - 中可靠区域: soft PGPC (匹配 teacher 的 prototype-induced soft distribution)
        - 低可靠区域: 不参与
        
        Args:
            student_projections: student 的 embedding 列表
            pseudo_labels: 伪标签 [B, H, W]
            reliability_map: 可靠性图 [B, 1, H, W]
            high_reliability_threshold: 高可靠性阈值
            low_reliability_threshold: 低可靠性阈值
        
        Returns:
            loss: PGPC 损失
        """
        total_loss = 0.0
        num_valid_stages = 0
        
        for stage, proj in enumerate(student_projections):
            if stage >= self.prototype_bank.num_stages:
                continue
                
            B, C, H, W = proj.shape
            
            proj_flat = proj.permute(0, 2, 3, 1).reshape(-1, C)
            proj_flat = F.normalize(proj_flat, dim=1)
            
            labels_resized = F.interpolate(
                pseudo_labels.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            labels_flat = labels_resized.reshape(-1)
            
            reliability_flat = F.interpolate(reliability_map, size=(H, W), mode='bilinear', align_corners=False)
            reliability_flat = reliability_flat.reshape(-1)
            
            high_mask = reliability_flat >= high_reliability_threshold
            medium_mask = (reliability_flat >= low_reliability_threshold) & (reliability_flat < high_reliability_threshold)
            
            if not high_mask.any() and not medium_mask.any():
                continue
            
            class_scores = []
            for c in range(self.num_classes):
                core_proto, trans_proto = self.prototype_bank.get_prototypes(stage, c)
                all_proto = torch.cat([core_proto, trans_proto], dim=0).detach()
                
                similarities = torch.mm(proj_flat, all_proto.t())
                max_sim, _ = similarities.max(dim=1)
                class_scores.append(max_sim)
            
            class_scores = torch.stack(class_scores, dim=1) / self.temperature
            
            if high_mask.any():
                ce_loss = F.cross_entropy(class_scores[high_mask], labels_flat[high_mask], reduction='mean')
                total_loss = total_loss + ce_loss
                num_valid_stages += 1
            
            if medium_mask.any():
                with torch.no_grad():
                    target_probs = F.softmax(class_scores[medium_mask], dim=1)
                log_probs = F.log_softmax(class_scores[medium_mask], dim=1)
                kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
                total_loss = total_loss + kl_loss * 0.5
                num_valid_stages += 1
        
        return total_loss / max(num_valid_stages, 1)
    
    def apc_loss(self,
                 student_projections: List[torch.Tensor],
                 pseudo_labels: torch.Tensor,
                 boundary_map: torch.Tensor,
                 reliability_map: torch.Tensor,
                 high_reliability_threshold: float = 0.75) -> torch.Tensor:
        """
        Boundary-aware Adaptive Prototype Contrast Loss
        
        核心思想:
        - 显式建模类内变化，尤其是边界过渡带
        - 正样本: 所属类别、所属区域子库中的最近 prototype
        - 负样本: 同类其他 prototypes、另一子区域中的同类 prototypes、所有异类 prototypes
        
        Args:
            student_projections: student 的 embedding 列表
            pseudo_labels: 伪标签 [B, H, W]
            boundary_map: 边界图 [B, 1, H, W]
            reliability_map: 可靠性图 [B, 1, H, W]
            high_reliability_threshold: 高可靠性阈值
        
        Returns:
            loss: APC 损失
        """
        total_loss = 0.0
        num_valid_stages = 0
        
        for stage, proj in enumerate(student_projections):
            if stage >= self.prototype_bank.num_stages:
                continue
                
            B, C, H, W = proj.shape
            
            proj_flat = proj.permute(0, 2, 3, 1).reshape(-1, C)
            proj_flat = F.normalize(proj_flat, dim=1)
            
            labels_resized = F.interpolate(
                pseudo_labels.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            labels_flat = labels_resized.reshape(-1)
            
            reliability_flat = F.interpolate(reliability_map, size=(H, W), mode='bilinear', align_corners=False)
            reliability_flat = reliability_flat.reshape(-1)
            boundary_flat = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)
            boundary_flat = boundary_flat.reshape(-1)
            
            high_mask = reliability_flat >= high_reliability_threshold
            if not high_mask.any():
                continue
            
            is_transition = boundary_flat >= 0.3
            
            all_logits = []
            all_targets = []
            
            for c in range(self.num_classes):
                class_mask = (labels_flat == c) & high_mask
                if not class_mask.any():
                    continue
                
                class_features = proj_flat[class_mask]
                class_is_transition = is_transition[class_mask]
                
                core_proto, trans_proto = self.prototype_bank.get_prototypes(stage, c)
                core_proto = core_proto.detach()
                trans_proto = trans_proto.detach()
                all_class_proto = torch.cat([core_proto, trans_proto], dim=0)
                
                core_sims = torch.mm(class_features, core_proto.t())
                trans_sims = torch.mm(class_features, trans_proto.t())
                
                core_max = core_sims.max(dim=1)[0]
                trans_max = trans_sims.max(dim=1)[0]
                pos_sim = torch.where(class_is_transition, trans_max, core_max).unsqueeze(1)
                
                neg_sims_list = []
                for other_c in range(self.num_classes):
                    other_core, other_trans = self.prototype_bank.get_prototypes(stage, other_c)
                    other_all = torch.cat([other_core.detach(), other_trans.detach()], dim=0)
                    neg_sims_list.append(torch.mm(class_features, other_all.t()))
                
                neg_sims = torch.cat(neg_sims_list, dim=1)
                
                logits = torch.cat([pos_sim, neg_sims], dim=1) / self.temperature
                all_logits.append(logits)
                all_targets.append(torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
            
            if all_logits:
                all_logits = torch.cat(all_logits, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                stage_loss = F.cross_entropy(all_logits, all_targets)
                total_loss = total_loss + stage_loss
                num_valid_stages += 1
        
        return total_loss / max(num_valid_stages, 1)
    
    def pac_loss(self,
                 student_projections: List[torch.Tensor],
                 teacher_projections: List[torch.Tensor],
                 reliability_map: torch.Tensor,
                 high_reliability_threshold: float = 0.75) -> torch.Tensor:
        """
        Prototype Assignment Consistency Loss
        
        核心思想:
        - teacher weak view 与 student strong view 在同一空间位置上的像素
        - 不仅"类别预测一致"，还应该"prototype assignment 也尽量一致"
        - 约束它们落到同一个 prototype 或至少同一个 prototype subset
        
        Args:
            student_projections: student 的 embedding 列表
            teacher_projections: teacher 的 embedding 列表
            reliability_map: 可靠性图 [B, 1, H, W]
            high_reliability_threshold: 高可靠性阈值
        
        Returns:
            loss: PAC 损失
        """
        total_loss = 0.0
        num_valid_stages = 0
        
        for stage, (s_proj, t_proj) in enumerate(zip(student_projections, teacher_projections)):
            if stage >= self.prototype_bank.num_stages:
                continue
                
            B, C, H, W = s_proj.shape
            
            s_flat = s_proj.permute(0, 2, 3, 1).reshape(-1, C)
            s_flat = F.normalize(s_flat, dim=1)
            
            t_flat = t_proj.permute(0, 2, 3, 1).reshape(-1, C)
            t_flat = F.normalize(t_flat, dim=1)
            
            reliability_flat = F.interpolate(reliability_map, size=(H, W), mode='bilinear', align_corners=False)
            reliability_flat = reliability_flat.reshape(-1)
            
            high_mask = reliability_flat >= high_reliability_threshold
            if not high_mask.any():
                continue
            
            s_high = s_flat[high_mask]
            t_high = t_flat[high_mask]
            
            all_prototypes = []
            for c in range(self.num_classes):
                core_proto, trans_proto = self.prototype_bank.get_prototypes(stage, c)
                all_prototypes.append(torch.cat([core_proto.detach(), trans_proto.detach()], dim=0))
            all_prototypes = torch.cat(all_prototypes, dim=0)
            
            s_sims = torch.mm(s_high, all_prototypes.t())
            t_sims = torch.mm(t_high, all_prototypes.t())
            
            s_probs = F.softmax(s_sims / self.temperature, dim=1)
            t_probs = F.softmax(t_sims / self.temperature, dim=1)
            
            kl_loss = F.kl_div(
                F.log_softmax(s_sims / self.temperature, dim=1),
                t_probs.detach(),
                reduction='batchmean'
            )
            
            total_loss = total_loss + kl_loss
            num_valid_stages += 1
        
        return total_loss / max(num_valid_stages, 1)
    
    def forward(self,
                student_projections: List[torch.Tensor],
                teacher_projections: List[torch.Tensor],
                pseudo_labels: torch.Tensor,
                reliability_map: torch.Tensor,
                boundary_map: torch.Tensor,
                high_reliability_threshold: float = 0.75,
                low_reliability_threshold: float = 0.45,
                pgpc_weight: float = 1.0,
                apc_weight: float = 0.5,
                pac_weight: float = 0.3) -> Dict[str, torch.Tensor]:
        """
        计算所有 prototype 损失
        
        Args:
            student_projections: student 的 embedding 列表
            teacher_projections: teacher 的 embedding 列表
            pseudo_labels: 伪标签 [B, H, W]
            reliability_map: 可靠性图 [B, 1, H, W]
            boundary_map: 边界图 [B, 1, H, W]
            high_reliability_threshold: 高可靠性阈值
            low_reliability_threshold: 低可靠性阈值
            pgpc_weight: PGPC 损失权重
            apc_weight: APC 损失权重
            pac_weight: PAC 损失权重
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        pgpc = self.pgpc_loss(
            student_projections, pseudo_labels, reliability_map,
            high_reliability_threshold=high_reliability_threshold,
            low_reliability_threshold=low_reliability_threshold
        )
        losses['pgpc_loss'] = pgpc * pgpc_weight
        
        apc = self.apc_loss(
            student_projections, pseudo_labels, boundary_map, reliability_map,
            high_reliability_threshold=high_reliability_threshold
        )
        losses['apc_loss'] = apc * apc_weight
        
        pac = self.pac_loss(
            student_projections, teacher_projections, reliability_map,
            high_reliability_threshold=high_reliability_threshold
        )
        losses['pac_loss'] = pac * pac_weight
        
        losses['total'] = losses['pgpc_loss'] + losses['apc_loss'] + losses['pac_loss']
        
        return losses


if __name__ == '__main__':
    print("Testing Prototype Modules...")
    
    num_stages = 3
    num_classes = 3
    projection_dim = 64
    batch_size = 2
    H, W = 32, 32
    
    input_dims = [64, 128, 256]
    proj_heads = MultiStageProjectionHeads(input_dims, projection_dim)
    
    skip_features = [
        torch.randn(batch_size, 64, H, W),
        torch.randn(batch_size, 128, H // 2, W // 2),
        torch.randn(batch_size, 256, H // 4, W // 4),
    ]
    
    projections = proj_heads(skip_features)
    print(f"Projections: {[p.shape for p in projections]}")
    
    prototype_bank = PrototypeBank(
        num_stages=num_stages,
        num_classes=num_classes,
        num_prototypes_per_class=5,
        projection_dim=projection_dim
    )
    
    labels = torch.randint(0, num_classes, (batch_size, H, W))
    reliability_map = torch.rand(batch_size, 1, H, W)
    boundary_map = torch.rand(batch_size, 1, H, W)
    
    prototype_bank.update(projections, labels, reliability_map, boundary_map)
    
    prototype_loss = PrototypeLoss(prototype_bank, num_classes)
    
    teacher_projections = [p.clone() for p in projections]
    
    losses = prototype_loss(
        student_projections=projections,
        teacher_projections=teacher_projections,
        pseudo_labels=labels,
        reliability_map=reliability_map,
        boundary_map=boundary_map
    )
    
    print(f"Losses: {losses}")
    
    print("\n✅ All tests passed!")
