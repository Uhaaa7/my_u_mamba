import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math
import torchvision.ops

# 尝试导入 H-SS2D (既然你已经把文件拷进去了，这里应该能直接导入)
try:
    from .H_vmunet import H_SS2D
except ImportError:
    try:
        from H_vmunet import H_SS2D
    except ImportError:
        pass 

class DCNv2_PyTorch(nn.Module):
    """
    【零编译版】使用 PyTorch 原生 torchvision 实现的 DCNv2
    完全不需要编译 DCNv4，即插即用，专门拯救环境配置失败
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 1. 生成偏移量 (Offsets) 的卷积层
        # 输出通道 = 2 * kernel_size * kernel_size (对应 x, y 坐标偏移)
        self.conv_offset = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # 2. 生成掩码 (Mask) 的卷积层 (DCNv2 特性)
        # 输出通道 = kernel_size * kernel_size (对应每个点的权重)
        self.conv_mask = nn.Conv2d(
            in_channels, 
            kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.sigmoid = nn.Sigmoid()
        
        # 3. 标准卷积权重
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
        # 偏移和掩码初始化为0，保证初始状态像普通卷积一样，训练更稳定
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.constant_(self.conv_mask.weight, 0)
        nn.init.constant_(self.conv_mask.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.sigmoid(self.conv_mask(x))
        
        # 调用 PyTorch 官方算子，无需编译
        return torchvision.ops.deform_conv2d(
            input=x, 
            offset=offset, 
            weight=self.weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            mask=mask
        )

class SDG_Block(nn.Module):
    """
    SDG-Block: Shape-Deformable Global-Gating Block
    【最终方案】: DCNv2 + H-SS2D
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        
        # === 1. 使用原生 DCNv2 (核心替换) ===
        self.dcn = DCNv2_PyTorch(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # === 2. H-SS2D 配置 ===
        # 自动计算 order，防止通道太少报错
        max_possible_order = int(math.log2(dim)) - 1
        safe_order = min(5, max(2, max_possible_order))
        
        self.h_ss2d = H_SS2D(dim=dim, order=safe_order, s=1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        输入: (B, C, H, W)
        """
        shortcut = x

        # --- DCNv2 部分 ---
        # 它可以直接吃 (B, C, H, W)
        x = self.dcn(x)
        
        # --- H-SS2D 部分 ---
        # H-SS2D 需要 (B, C, H, W)
        # 只有 LayerNorm 需要转一下维度
        x = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() # 转回 (B, C, H, W)

        x = self.h_ss2d(x)

        x = shortcut + self.drop_path(x)
        return x