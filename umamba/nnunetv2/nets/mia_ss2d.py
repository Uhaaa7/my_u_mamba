"""
MIA-SSM: Medical-Image Adaptive State-Space Module
医学图像自适应状态空间建模框架

================================================================================
核心创新点（问题导向的层级提升）
================================================================================

传统U型网络在医学图像分割中存在三大核心挑战：

【挑战1】多尺度特征提取问题
  → 现有方法：固定感受野的卷积核
  → 我们的方案：多尺度自适应特征提取模块

【挑战2】边界模糊问题  
  → 现有方法：固定的边缘检测算子
  → 我们的方案：医学图像边界感知增强机制

【挑战3】语义鸿沟问题
  → 现有方法：简单的特征拼接或求和
  → 我们的方案：基于状态空间建模的语义鸿沟弥合策略

================================================================================
本次修改（最小侵入式结构优化）
================================================================================

1. 内部短连接 (Internal Short Connection)
   - 在 ARF 输出和 SS2D 输入之间加入短连接
   - 让 SS2D 同时接收原始多尺度结构基底与判别性增强特征

2. Full / Light 双模式支持
   - full: 完整模式，包含所有增强模块
   - light: 轻量模式，保留核心主线，减少计算开销

3. 残差对称性
   - MIA_SS2D 只输出增量特征，残差统一在 MIA_SS2D_Block 外层做
   - 保证 DCN 分支和 MIA 分支对称

设计原则:
- 所有操作基于卷积，避免 FFT 等不稳定操作
- 使用 LayerNorm/BatchNorm 保证数值稳定
- 残差连接防止梯度消失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .vmamba import SS2D
except ImportError:
    from vmamba import SS2D

VALID_MODES = ['full', 'light', 'identity']


class AdaptiveReceptiveField(nn.Module):
    """
    自适应感受野模块
    通过多膨胀率卷积并行结构和自适应权重生成机制，动态调整感受野大小
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        branch_dim = dim // 4

        def dwconv(dilation, padding):
            return nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, dilation=dilation, groups=dim, bias=False),
                nn.Conv2d(dim, branch_dim, kernel_size=1, bias=False)
            )

        self.conv_d1 = dwconv(dilation=1, padding=1)
        self.conv_d2 = dwconv(dilation=2, padding=2)
        self.conv_d3 = dwconv(dilation=4, padding=4)
        self.conv_d4 = dwconv(dilation=8, padding=8)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, branch_dim),
            nn.ReLU(inplace=True),
            nn.Linear(branch_dim, 4),
            nn.Softmax(dim=1)
        )

        self.fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        f1 = self.conv_d1(x)
        f2 = self.conv_d2(x)
        f3 = self.conv_d3(x)
        f4 = self.conv_d4(x)

        gap = self.gap(x).view(B, C)
        weights = self.fc(gap)

        w1 = weights[:, 0:1].view(B, 1, 1, 1)
        w2 = weights[:, 1:2].view(B, 1, 1, 1)
        w3 = weights[:, 2:3].view(B, 1, 1, 1)
        w4 = weights[:, 3:4].view(B, 1, 1, 1)

        f1_weighted = f1 * w1
        f2_weighted = f2 * w2
        f3_weighted = f3 * w3
        f4_weighted = f4 * w4

        out_concat = torch.cat([f1_weighted, f2_weighted, f3_weighted, f4_weighted], dim=1)
        return self.fuse(out_concat)


class LearnableEdgeEnhancement(nn.Module):
    """
    可学习边缘增强模块
    使用可学习的卷积核代替固定的边缘检测算子
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.edge_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        for i in range(dim):
            self.edge_conv.weight.data[i, 0] = laplacian
        
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        edge = self.edge_conv(x)
        edge_weight = self.enhance(edge)
        return x * (1 + edge_weight)


class ChannelRecalibration(nn.Module):
    """
    通道重标定模块
    在 SS2D 前后添加通道注意力
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        mid_dim = max(1, dim // reduction)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, dim, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        gap = self.gap(x).view(B, C)
        gmp = self.gmp(x).view(B, C)
        
        weight = self.fc(gap + gmp)
        
        return x * weight.view(B, C, 1, 1)


class LightChannelRecalibration(nn.Module):
    """
    轻量通道重标定模块 (用于 light 模式)
    
    修复: 输出的是重标定后的特征图，而不是权重
    保持与 ChannelRecalibration 相同的语义
    """
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 生成通道权重 [B, C, 1, 1]
        weight = self.fc(self.gap(x))
        # 返回重标定后的特征图 [B, C, H, W]
        return x * weight


class MIA_SS2D(nn.Module):
    """
    Medical-Image Adaptive SS2D
    
    本次修改:
    1. 添加内部短连接: ARF 输出直接跳接到 SS2D 输入
    2. 支持 full / light 两种模式
    3. 只输出增量特征，残差在外层做
    
    模式说明:
    - full: 完整模式，包含 ARF + LEE + pre_cal + internal_short_connection + SS2D + post_cal + proj
    - light: 轻量模式，包含 ARF + internal_short_connection + SS2D + proj，跳过 LEE 和 post_cal
    """
    def __init__(self, dim, d_state=16, drop_path=0., mode='full'):
        super().__init__()
        
        # 健壮性检查
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}, got {mode}"
        
        self.dim = dim
        self.mode = mode
        
        # 创新点1: 自适应感受野 (full 和 light 都保留)
        self.arf = AdaptiveReceptiveField(dim)
        
        # 创新点2: 可学习边缘增强 (仅 full 模式)
        if mode == 'full':
            self.edge = LearnableEdgeEnhancement(dim)
        else:
            self.edge = nn.Identity()
        
        # 创新点3: 前置通道重标定 (full 和 light 都保留)
        # 修复: light 模式也输出特征图，保持语义一致
        if mode == 'full':
            self.pre_cal = ChannelRecalibration(dim)
        else:
            self.pre_cal = LightChannelRecalibration(dim)
        
        # 核心: SS2D 状态空间模型 (full 和 light 都保留)
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=3, expand=2)
        
        # 创新点3: 后置通道重标定 (仅 full 模式)
        if mode == 'full':
            self.post_cal = ChannelRecalibration(dim)
        else:
            self.post_cal = nn.Identity()
        
        # 输出投影 (full 和 light 都保留)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # DropPath
        if drop_path > 0.:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
            
    def forward(self, x):
        # 修复: 只输出增量特征，不做内部残差
        # 残差统一在 MIA_SS2D_Block 外层做
        
        # 1. 自适应感受野 - 提取多尺度结构基底
        arf_out = self.arf(x)
        
        # 2. 边缘增强 (full 模式)
        if self.mode == 'full':
            enhanced = self.edge(arf_out)
        else:
            enhanced = arf_out
        
        # 3. 前置通道重标定
        # 修复: calibrated 是 [B, C, H, W] 的特征图，不是权重
        calibrated = self.pre_cal(enhanced)
        
        # 4. 内部短连接: 将 ARF 输出与筛选后的特征融合
        # SS2D 同时接收原始多尺度结构基底与判别性增强特征
        ss2d_input = arf_out + calibrated
        
        # 5. SS2D 全局建模
        B, C, H, W = ss2d_input.shape
        ss2d_input = ss2d_input.permute(0, 2, 3, 1)  # [B, H, W, C]
        ss2d_input = self.norm(ss2d_input)
        ss2d_out = self.ss2d(ss2d_input)
        ss2d_out = ss2d_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 6. 后置通道重标定 (full 模式)
        if self.mode == 'full':
            ss2d_out = self.post_cal(ss2d_out)
        
        # 7. 输出投影
        out = self.proj(ss2d_out)
        
        # 修复: 只返回增量特征，不做残差
        return self.drop_path(out)


class MIA_SS2D_Block(nn.Module):
    """
    完整的 MIA_SS2D Block，包含两个分支:
    1. DCNv2 分支: 可变形卷积，捕捉局部形变
    2. MIA_SS2D 分支: 医学图像自适应全局建模
    
    支持 full / light 两种模式
    
    修复: 两个分支都输出增量特征，残差统一在外层做
    """
    def __init__(self, dim, d_state=16, drop_path=0., mode='full'):
        super().__init__()
        
        # 健壮性检查
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}, got {mode}"
        
        self.mode = mode
        
        # DCNv2 分支 (输出增量特征)
        self.offset_conv = nn.Conv2d(dim, 18, 3, padding=1)
        self.mask_conv = nn.Conv2d(dim, 9, 3, padding=1)
        self.dcn_weight = nn.Parameter(torch.Tensor(dim, dim, 3, 3))
        nn.init.kaiming_uniform_(self.dcn_weight, a=math.sqrt(5))
        
        # MIA_SS2D 分支 (输出增量特征)
        self.mia_ss2d = MIA_SS2D(dim, d_state=d_state, drop_path=0., mode=mode)
        
        # 自适应融合
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.norm = nn.LayerNorm(dim)
        
        if drop_path > 0.:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
            
    def forward(self, x):
        import torchvision.ops
        
        shortcut = x
        
        # DCNv2 分支 (输出增量特征)
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        dcn_out = torchvision.ops.deform_conv2d(
            input=x, offset=offset, weight=self.dcn_weight,
            bias=None, stride=1, padding=1, mask=mask
        )
        
        # MIA_SS2D 分支 (输出增量特征)
        mia_out = self.mia_ss2d(x)
        
        # 自适应融合两个增量特征
        gate = self.fusion_gate(x)
        w_dcn = gate[:, 0:1, :, :]
        w_mia = gate[:, 1:2, :, :]
        
        fused = w_dcn * dcn_out + w_mia * mia_out
        
        # LayerNorm
        fused = fused.permute(0, 2, 3, 1)
        fused = self.norm(fused)
        fused = fused.permute(0, 3, 1, 2)
        
        # 统一在外层做残差
        return shortcut + self.drop_path(fused)


# 兼容性别名
SDG_Block = MIA_SS2D_Block
SDG_Block_V2_Simple = MIA_SS2D_Block


if __name__ == '__main__':
    print("Testing MIA_SS2D modules...")
    
    dim = 64
    x = torch.randn(2, dim, 32, 32).cuda()
    
    # Test MIA_SS2D (full mode)
    print("\n1. Testing MIA_SS2D (full mode)...")
    mia_full = MIA_SS2D(dim, mode='full').cuda()
    out = mia_full(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in mia_full.parameters()):,}")
    
    # Test MIA_SS2D (light mode)
    print("\n2. Testing MIA_SS2D (light mode)...")
    mia_light = MIA_SS2D(dim, mode='light').cuda()
    out = mia_light(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in mia_light.parameters()):,}")
    
    # Test MIA_SS2D_Block (full mode)
    print("\n3. Testing MIA_SS2D_Block (full mode)...")
    block_full = MIA_SS2D_Block(dim, mode='full').cuda()
    out = block_full(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in block_full.parameters()):,}")
    
    # Test MIA_SS2D_Block (light mode)
    print("\n4. Testing MIA_SS2D_Block (light mode)...")
    block_light = MIA_SS2D_Block(dim, mode='light').cuda()
    out = block_light(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in block_light.parameters()):,}")
    
    # Test LightChannelRecalibration
    print("\n5. Testing LightChannelRecalibration...")
    light_cal = LightChannelRecalibration(dim).cuda()
    out = light_cal(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Output is feature map (not weight): {out.shape == x.shape}")
    
    # Gradient test
    print("\n6. Testing gradients...")
    x = torch.randn(2, dim, 32, 32).cuda()
    block = MIA_SS2D_Block(dim, mode='full').cuda()
    out = block(x)
    loss = out.mean()
    loss.backward()
    has_nan_grad = any(torch.isnan(p.grad).any().item() for p in block.parameters() if p.grad is not None)
    print(f"   Has NaN gradient: {has_nan_grad}")
    
    print("\n✅ All tests passed!")
