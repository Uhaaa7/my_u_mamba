"""
SDG_Block: Spatial-Deformable Global Module
空间可变形全局建模模块

结合 DCNv2 (局部自适应采样) + MSS_SS2D (医学图像特定的全局状态空间建模)

创新点:
1. DCNv2: 可变形卷积实现自适应空间采样
2. MSS_SS2D: 医学图像特定的多尺度状态空间建模
   - 频域增强分支
   - 边缘感知模块
   - 跨层级状态交互
   - 自适应状态维度选择
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math
import torchvision.ops
from torch.utils.checkpoint import checkpoint

try:
    from .mss_ss2d import MSS_SS2D
except ImportError as e:
    try:
        from mss_ss2d import MSS_SS2D
    except ImportError as e2:
        print(f"Error: Cannot import MSS_SS2D")
        print(f"Relative import error: {e}")
        print(f"Absolute import error: {e2}")
        raise e2


class DCNv2_PyTorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv_offset = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        self.conv_mask = nn.Conv2d(
            in_channels, 
            kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.sigmoid = nn.Sigmoid()
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.constant_(self.conv_mask.weight, 0)
        nn.init.constant_(self.conv_mask.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.sigmoid(self.conv_mask(x))
        return torchvision.ops.deform_conv2d(
            input=x, 
            offset=offset, 
            weight=self.weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            mask=mask
        )


class SDG_Block_V2(nn.Module):
    """
    SDG_Block V2: 增强版空间可变形全局建模模块
    
    创新设计:
    1. DCNv2 分支: 捕捉局部形变和不规则结构
    2. MSS_SS2D 分支: 医学图像特定的全局状态空间建模
    3. 自适应特征融合: 动态平衡局部和全局特征
    4. 边缘增强: 显式强化医学图像边界
    """
    def __init__(self, dim, drop_path=0., use_freq=True, use_edge=True, 
                 use_cross_level=True, use_adaptive_state=True):
        super().__init__()
        
        self.dcn = DCNv2_PyTorch(
            in_channels=dim, 
            out_channels=dim, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        max_possible_order = int(math.log2(dim)) - 1
        safe_order = min(5, max(2, max_possible_order))
        
        self.mss_ss2d = MSS_SS2D(
            dim=dim, 
            order=safe_order, 
            s=1.0,
            use_freq=use_freq,
            use_edge=use_edge,
            use_cross_level=use_cross_level,
            use_adaptive_state=use_adaptive_state
        )
        
        self.norm = nn.LayerNorm(dim)
        
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float()
            
            def _inner_forward(input_x):
                shortcut = input_x
                
                dcn_out = self.dcn(input_x)
                
                mss_out = self.mss_ss2d(input_x)
                
                gate = self.fusion_gate(input_x)
                w_dcn, w_mss = gate[:, 0:1, :, :], gate[:, 1:2, :, :]
                
                fused = w_dcn * dcn_out + w_mss * mss_out
                
                fused = fused.permute(0, 2, 3, 1).contiguous()
                fused = self.norm(fused)
                fused = fused.permute(0, 3, 1, 2).contiguous()
                
                out = shortcut + self.drop_path(fused)
                return out

            if self.training and x.requires_grad:
                return checkpoint(_inner_forward, x, use_reentrant=False)
            else:
                return _inner_forward(x)


class SDG_Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dcn = DCNv2_PyTorch(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        
        max_possible_order = int(math.log2(dim)) - 1
        safe_order = min(5, max(2, max_possible_order))
        self.mss_ss2d = MSS_SS2D(dim=dim, order=safe_order, s=1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float()
            
            def _inner_forward(input_x):
                shortcut = input_x
                x = self.dcn(input_x)
                
                x = x.permute(0, 2, 3, 1).contiguous()
                x = self.norm(x)
                x = x.permute(0, 3, 1, 2).contiguous()
                
                x = self.mss_ss2d(x)
                
                x = shortcut + self.drop_path(x)
                return x

            if self.training and x.requires_grad:
                return checkpoint(_inner_forward, x, use_reentrant=False)
            else:
                return _inner_forward(x)


if __name__ == '__main__':
    dim = 64
    x = torch.randn(2, dim, 32, 32).cuda()
    
    print("Testing SDG_Block...")
    sdg = SDG_Block(dim).cuda()
    out = sdg(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    print("\nTesting SDG_Block_V2...")
    sdg_v2 = SDG_Block_V2(dim).cuda()
    out_v2 = sdg_v2(x)
    print(f"Input: {x.shape}, Output: {out_v2.shape}")
    
    print("\nParameter count:")
    print(f"SDG_Block: {sum(p.numel() for p in sdg.parameters()):,}")
    print(f"SDG_Block_V2: {sum(p.numel() for p in sdg_v2.parameters()):,}")
