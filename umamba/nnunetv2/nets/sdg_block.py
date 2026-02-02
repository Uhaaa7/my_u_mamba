import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math
import torchvision.ops

# ==========================================
# 修复后的导入逻辑：不再隐藏报错
# ==========================================
try:
    # 尝试相对导入 (用于 nnU-Net 训练时)
    from .H_vmunet import H_SS2D
except ImportError as e:
    try:
        # 尝试绝对导入 (用于单独 python test_model.py 测试时)
        from H_vmunet import H_SS2D
    except ImportError as e2:
        # 如果两次都失败，打印详细错误并抛出异常，不再沉默！
        print(f"❌ 严重错误: 无法导入 H_vmunet.H_SS2D")
        print(f"   相对导入错误: {e}")
        print(f"   绝对导入错误: {e2}")
        raise e2 

class DCNv2_PyTorch(nn.Module):
    """
    【零编译版】使用 PyTorch 原生 torchvision 实现的 DCNv2
    完全不需要编译 DCNv4，即插即用
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 1. 生成偏移量 (Offsets)
        self.conv_offset = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # 2. 生成掩码 (Mask)
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

class SDG_Block(nn.Module):
    """
    SDG-Block: Shape-Deformable Global-Gating Block
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        
        # === 1. DCNv2 ===
        self.dcn = DCNv2_PyTorch(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # === 2. H-SS2D ===
        # 自动计算 order 防止报错
        max_possible_order = int(math.log2(dim)) - 1
        safe_order = min(5, max(2, max_possible_order))
        
        # 这里之前报错是因为 H_SS2D 没定义，现在如果导入失败会直接崩在 import 处，方便调试
        self.h_ss2d = H_SS2D(dim=dim, order=safe_order, s=1.0)
        
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dcn(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.h_ss2d(x)
        x = shortcut + self.drop_path(x)
        return x