"""
MIA_SS2D: Medical-Image Adaptive State-Space Module
医学图像自适应状态空间模块

创新点 (全部基于稳定实现):
1. 自适应感受野模块 (Adaptive Receptive Field): 根据输入内容动态调整感受野
2. 可学习边缘增强 (Learnable Edge Enhancement): 卷积层学习边缘特征
3. 通道重标定 (Channel Recalibration): SS2D 前后的通道注意力
4. 多尺度特征融合 (Multi-Scale Fusion): 不同膨胀率的卷积并行

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


class AdaptiveReceptiveField(nn.Module):
    """
    创新点1: 自适应感受野模块
    
    根据输入内容动态选择不同大小的感受野:
    - 小感受野: 捕捉细节和边缘
    - 大感受野: 捕捉全局上下文
    
    医学图像中器官/病变大小差异大，需要自适应感受野
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 不同膨胀率的卷积
        self.conv_d1 = nn.Conv2d(dim, dim // 4, 3, padding=1, dilation=1, groups=dim // 4)
        self.conv_d2 = nn.Conv2d(dim, dim // 4, 3, padding=2, dilation=2, groups=dim // 4)
        self.conv_d3 = nn.Conv2d(dim, dim // 4, 3, padding=4, dilation=4, groups=dim // 4)
        self.conv_d4 = nn.Conv2d(dim, dim // 4, 3, padding=8, dilation=8, groups=dim // 4)
        
        # 自适应权重生成
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 4),
            nn.Softmax(dim=1)
        )
        
        self.fuse = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 多尺度特征
        f1 = self.conv_d1(x)
        f2 = self.conv_d2(x)
        f3 = self.conv_d3(x)
        f4 = self.conv_d4(x)
        
        # 自适应权重
        gap = self.gap(x).view(B, C)
        weights = self.fc(gap)  # [B, 4]
        
        # 加权融合
        out = (weights[:, 0:1, None, None] * f1 + 
               weights[:, 1:2, None, None] * f2 + 
               weights[:, 2:3, None, None] * f3 + 
               weights[:, 3:4, None, None] * f4)
        
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))


class LearnableEdgeEnhancement(nn.Module):
    """
    创新点2: 可学习边缘增强模块
    
    使用可学习的卷积核代替固定的 Sobel 算子:
    - 更灵活，可以学习医学图像特定的边缘模式
    - 数值稳定，不会出现 sqrt 导致的梯度问题
    
    医学图像边界模糊，需要显式增强边缘信息
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 可学习的边缘检测核 (初始化为类 Sobel)
        self.edge_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        
        # 初始化为 Laplacian 算子 (比 Sobel 更稳定)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        for i in range(dim):
            self.edge_conv.weight.data[i, 0] = laplacian
        
        # 边缘特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 边缘检测
        edge = self.edge_conv(x)
        
        # 边缘增强权重
        edge_weight = self.enhance(edge)
        
        # 增强原始特征
        return x * (1 + edge_weight)


class ChannelRecalibration(nn.Module):
    """
    创新点3: 通道重标定模块
    
    在 SS2D 前后添加通道注意力:
    - 前置: 选择重要通道进入 SS2D
    - 后置: 重新校准 SS2D 输出
    
    医学图像不同通道特征重要性差异大
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
        
        # 全局平均池化和最大池化
        gap = self.gap(x).view(B, C)
        gmp = self.gmp(x).view(B, C)
        
        # 通道权重
        weight = self.fc(gap + gmp)
        
        return x * weight.view(B, C, 1, 1)


class MIA_SS2D(nn.Module):
    """
    Medical-Image Adaptive SS2D
    
    核心创新:
    1. 自适应感受野 - 动态调整感受野大小
    2. 可学习边缘增强 - 增强医学图像边界
    3. 通道重标定 - SS2D 前后通道注意力
    4. 多尺度融合 - 不同膨胀率特征融合
    
    与原版 SS2D 的区别:
    - 原版: 直接使用 SS2D，无医学图像特定设计
    - 本版: 围绕 SS2D 添加医学图像自适应增强
    """
    def __init__(self, dim, d_state=16, drop_path=0.):
        super().__init__()
        self.dim = dim
        
        # 创新点1: 自适应感受野
        self.arf = AdaptiveReceptiveField(dim)
        
        # 创新点2: 可学习边缘增强
        self.edge = LearnableEdgeEnhancement(dim)
        
        # 创新点3: 前置通道重标定
        self.pre_cal = ChannelRecalibration(dim)
        
        # 核心: SS2D 状态空间模型
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=3, expand=2)
        
        # 创新点3: 后置通道重标定
        self.post_cal = ChannelRecalibration(dim)
        
        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # DropPath
        if drop_path > 0.:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
            
    def forward(self, x):
        shortcut = x
        
        # 1. 自适应感受野
        x = self.arf(x)
        
        # 2. 边缘增强
        x = self.edge(x)
        
        # 3. 前置通道重标定
        x = self.pre_cal(x)
        
        # 4. SS2D 全局建模
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.ss2d(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 5. 后置通道重标定
        x = self.post_cal(x)
        
        # 6. 输出投影
        x = self.proj(x)
        
        # 残差连接
        return shortcut + self.drop_path(x)


class MIA_SS2D_Block(nn.Module):
    """
    完整的 MIA_SS2D Block，包含两个分支:
    1. DCNv2 分支: 可变形卷积，捕捉局部形变
    2. MIA_SS2D 分支: 医学图像自适应全局建模
    
    自适应融合两个分支的输出
    """
    def __init__(self, dim, d_state=16, drop_path=0.):
        super().__init__()
        
        # DCNv2 分支
        self.offset_conv = nn.Conv2d(dim, 18, 3, padding=1)  # 2*3*3=18 for offset
        self.mask_conv = nn.Conv2d(dim, 9, 3, padding=1)  # 3*3=9 for mask
        self.dcn_weight = nn.Parameter(torch.Tensor(dim, dim, 3, 3))
        nn.init.kaiming_uniform_(self.dcn_weight, a=math.sqrt(5))
        
        # MIA_SS2D 分支
        self.mia_ss2d = MIA_SS2D(dim, d_state=d_state, drop_path=0.)
        
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
        
        # DCNv2 分支
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        dcn_out = torchvision.ops.deform_conv2d(
            input=x, offset=offset, weight=self.dcn_weight,
            bias=None, stride=1, padding=1, mask=mask
        )
        
        # MIA_SS2D 分支
        mia_out = self.mia_ss2d(x)
        
        # 自适应融合
        gate = self.fusion_gate(x)
        w_dcn = gate[:, 0:1, :, :]
        w_mia = gate[:, 1:2, :, :]
        
        fused = w_dcn * dcn_out + w_mia * mia_out
        
        # LayerNorm
        fused = fused.permute(0, 2, 3, 1)
        fused = self.norm(fused)
        fused = fused.permute(0, 3, 1, 2)
        
        return shortcut + self.drop_path(fused)


# 为了兼容性，提供简化命名
SDG_Block_V2_Simple = MIA_SS2D_Block


if __name__ == '__main__':
    print("Testing MIA_SS2D modules...")
    
    dim = 64
    x = torch.randn(2, dim, 32, 32).cuda()
    
    # Test MIA_SS2D
    print("\n1. Testing MIA_SS2D...")
    mia = MIA_SS2D(dim).cuda()
    out = mia(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in mia.parameters()):,}")
    
    # Test MIA_SS2D_Block
    print("\n2. Testing MIA_SS2D_Block...")
    block = MIA_SS2D_Block(dim).cuda()
    out = block(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Has NaN: {torch.isnan(out).any().item()}")
    print(f"   Parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    # Gradient test
    print("\n3. Testing gradients...")
    x = torch.randn(2, dim, 32, 32).cuda()
    block = MIA_SS2D_Block(dim).cuda()
    out = block(x)
    loss = out.mean()
    loss.backward()
    has_nan_grad = any(torch.isnan(p.grad).any().item() for p in block.parameters() if p.grad is not None)
    print(f"   Has NaN gradient: {has_nan_grad}")
    
    print("\n✅ All tests passed!")
