"""
MSS-SS2D: Medical-Specific State-Space Module with Cross-Level Interaction
医学图像自适应的跨层级交互状态空间模块

创新点:
1. 跨层级状态交互 (Cross-Level State Interaction): 不同层级的状态信息流动
2. 频域增强 (Frequency Enhancement): 结合频域信息增强边缘和纹理感知
3. 自适应状态维度 (Adaptive State Dimension): 根据内容复杂度动态调整
4. 医学图像特定的边缘感知 (Edge-Aware for Medical Images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, DropPath

try:
    from .vmamba import SS2D
except ImportError:
    from vmamba import SS2D


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FrequencyBranch(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        mid_dim = max(1, dim // reduction)
        
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_dim, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, dim, 1),
            nn.Sigmoid()
        )
        
        self.high_freq_conv = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim) if dim > 0 else None
        self.low_freq_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if dim > 0 else None
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.dim <= 0 or self.high_freq_conv is None:
            return x
        
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        x_mag = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        
        center_h, center_w = H // 2, W // 2
        mask_low = torch.zeros_like(x_mag)
        mask_low[:, :, :max(1, center_h//4), :max(1, center_w//4)] = 1
        mask_high = 1 - mask_low
        
        low_freq = x_mag * mask_low
        high_freq = x_mag * mask_high
        
        low_feat = torch.fft.irfft2(low_freq * torch.exp(1j * x_phase), s=(H, W), norm='ortho')
        high_feat = torch.fft.irfft2(high_freq * torch.exp(1j * x_phase), s=(H, W), norm='ortho')
        
        low_feat = self.low_freq_conv(low_feat)
        high_feat = self.high_freq_conv(high_feat)
        
        freq_feat = low_feat + high_feat
        attention = self.freq_attention(freq_feat)
        
        return attention * freq_feat


class EdgeAwareModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.sobel_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False) if dim > 0 else None
        self.sobel_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False) if dim > 0 else None
        
        if dim > 0:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            for i in range(dim):
                self.sobel_x.weight.data[i, 0] = sobel_x
                self.sobel_y.weight.data[i, 0] = sobel_y
                
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False
        
        mid_dim = max(1, dim // 4)
        self.edge_attention = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if self.dim <= 0 or self.sobel_x is None:
            return torch.ones_like(x)
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge_attention = self.edge_attention(edge)
        return edge_attention


class CrossLevelStateInteraction(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.num_levels = len(dims)
        
        self.state_gates = nn.ModuleList()
        self.state_projections = nn.ModuleList()
        
        for i in range(self.num_levels):
            mid_dim = max(1, dims[i] // 4)
            self.state_gates.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dims[i], mid_dim, 1),
                    nn.ReLU(),
                    nn.Conv2d(mid_dim, 2, 1),
                    nn.Softmax(dim=1)
                )
            )
            
            if i > 0:
                self.state_projections.append(
                    nn.Conv2d(dims[i-1], dims[i], 1)
                )
                
    def forward(self, states):
        if not isinstance(states, list):
            states = [states]
            
        gated_states = []
        for i, state in enumerate(states):
            gate = self.state_gates[i](state)
            w_self, w_cross = gate[:, 0:1, :, :], gate[:, 1:2, :, :]
            
            if i > 0 and i <= len(self.state_projections):
                prev_state = states[i-1]
                if prev_state.shape[2:] != state.shape[2:]:
                    prev_state = F.interpolate(prev_state, size=state.shape[2:], mode='bilinear', align_corners=True)
                cross_info = self.state_projections[i-1](prev_state)
                gated_state = w_self * state + w_cross * cross_info
            else:
                gated_state = state
                
            gated_states.append(gated_state)
            
        return gated_states


class AdaptiveStateSelector(nn.Module):
    def __init__(self, dim, d_state_options=[8, 16, 32]):
        super().__init__()
        self.dim = dim
        self.d_state_options = d_state_options
        
        mid_dim = max(1, dim // 4)
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_dim, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, len(d_state_options), 1),
            nn.Softmax(dim=1)
        )
        
        self.ss2d_modules = nn.ModuleList([
            SS2D(d_model=dim, d_state=ds) for ds in d_state_options
        ])
        
    def forward(self, x):
        B, H, W, C = x.shape
        x_conv = x.permute(0, 3, 1, 2)
        
        complexity_weights = self.complexity_estimator(x_conv)
        
        outputs = []
        for i, ss2d in enumerate(self.ss2d_modules):
            out = ss2d(x)
            outputs.append(out)
        
        final_out = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            weight = complexity_weights[:, i:i+1, :, :].permute(0, 2, 3, 1)
            final_out = final_out + weight * out
            
        return final_out


class MSS_SS2D(nn.Module):
    """
    Medical-Specific State-Space Module with Cross-Level Interaction
    
    创新点总结:
    1. 频域增强分支 - 捕捉医学图像的纹理和边缘特征
    2. 边缘感知模块 - 显式增强器官边界和病变边缘
    3. 跨层级状态交互 - 不同尺度特征之间的信息流动
    4. 自适应状态选择 - 根据内容复杂度动态调整计算资源
    """
    def __init__(self, dim, order=5, d_state=16, use_freq=True, use_edge=True, 
                 use_cross_level=True, use_adaptive_state=True, s=1.0):
        super().__init__()
        self.dim = dim
        
        max_possible_order = int(math.log2(dim)) - 1 if dim > 4 else 2
        safe_order = min(order, max(2, max_possible_order))
        self.order = safe_order
        
        self.use_freq = use_freq
        self.use_edge = use_edge
        self.use_cross_level = use_cross_level
        self.use_adaptive_state = use_adaptive_state
        
        self.dims = [max(2, dim // 2 ** i) for i in range(safe_order)]
        self.dims.reverse()
        
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        self.dwconv = nn.Conv2d(
            sum(self.dims), sum(self.dims), 
            kernel_size=7, padding=3, groups=sum(self.dims), bias=True
        )
        
        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i+1], 1) 
            for i in range(safe_order - 1)
        ])
        
        if use_adaptive_state:
            self.ss2d_modules = nn.ModuleList([
                AdaptiveStateSelector(d, d_state_options=[8, 16, 24])
                for d in self.dims
            ])
        else:
            self.ss2d_modules = nn.ModuleList([
                SS2D(d_model=d, d_state=d_state) for d in self.dims
            ])
        
        if use_freq:
            self.freq_branch = FrequencyBranch(dim)
            self.freq_fusion = nn.Conv2d(dim * 2, dim, 1)
            
        if use_edge:
            self.edge_module = EdgeAwareModule(dim)
            
        if use_cross_level:
            self.cross_level = CrossLevelStateInteraction(self.dims)
        
        self.scale = s
        print(f'[MSS_SS2D] order={safe_order}, dims={self.dims}, '
              f'freq={use_freq}, edge={use_edge}, cross_level={use_cross_level}, '
              f'adaptive_state={use_adaptive_state}')
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        freq_feat = None
        if self.use_freq:
            freq_feat = self.freq_branch(x)
            
        edge_attention = None
        if self.use_edge:
            edge_attention = self.edge_module(x)
        
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        
        states = []
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_modules[0](x)
        x = x.permute(0, 3, 1, 2)
        states.append(x)
        
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
            x = x.permute(0, 2, 3, 1)
            x = self.ss2d_modules[i + 1](x)
            x = x.permute(0, 3, 1, 2)
            states.append(x)
        
        if self.use_cross_level and len(states) > 1:
            states = self.cross_level(states)
            x = states[-1]
        
        x = self.proj_out(x)
        
        if self.use_freq and freq_feat is not None:
            x = self.freq_fusion(torch.cat([x, freq_feat], dim=1))
            
        if self.use_edge and edge_attention is not None:
            x = x * edge_attention + x
            
        return x


class MSS_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, 
                 use_freq=True, use_edge=True, use_cross_level=True, use_adaptive_state=True):
        super().__init__()
        
        self.norm1 = LayerNorm2d(dim, eps=1e-6)
        self.mss_ss2d = MSS_SS2D(
            dim, 
            order=max(2, min(5, int(math.log2(dim)) - 1)),
            use_freq=use_freq,
            use_edge=use_edge,
            use_cross_level=use_cross_level,
            use_adaptive_state=use_adaptive_state
        )
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
            
        x = x + self.drop_path(gamma1 * self.mss_ss2d(self.norm1(x)))
        
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma2 is not None:
            x = self.gamma2 * x
            
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        
        return x


if __name__ == '__main__':
    dim = 64
    x = torch.randn(2, dim, 32, 32).cuda()
    
    print("Testing MSS_SS2D...")
    mss = MSS_SS2D(dim, order=4).cuda()
    out = mss(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\nTesting MSS_Block...")
    block = MSS_Block(dim).cuda()
    out = block(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    print("\nParameter count:")
    print(f"MSS_SS2D: {sum(p.numel() for p in mss.parameters()):,}")
    print(f"MSS_Block: {sum(p.numel() for p in block.parameters()):,}")
