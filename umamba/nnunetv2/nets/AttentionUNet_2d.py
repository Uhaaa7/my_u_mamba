import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple, Optional

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He


class GridAttentionBlock2D(nn.Module):
    """
    2D Grid Attention Block
    
    基于论文: "Attention U-Net: Learning Where to Look for the Pancreas"
    
    核心思想: 使用 gating signal 来对 encoder features 进行注意力加权
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None, 
                 sub_sample_factor=2, norm_op=None, norm_op_kwargs=None):
        super().__init__()
        
        self.sub_sample_factor = sub_sample_factor
        
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        
        self.theta = nn.Conv2d(in_channels, inter_channels, 
                               kernel_size=sub_sample_factor, 
                               stride=sub_sample_factor, 
                               padding=0, bias=False)
        
        self.phi = nn.Conv2d(gating_channels, inter_channels, 
                             kernel_size=1, stride=1, padding=0, bias=True)
        
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, g):
        """
        Args:
            x: encoder features, shape (B, C, H, W)
            g: gating signal from decoder, shape (B, C_g, H', W')
        
        Returns:
            attention weighted features and attention map
        """
        input_size = x.size()
        
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        
        f = F.relu(theta_x + phi_g, inplace=True)
        
        sigm_psi_f = torch.sigmoid(self.psi(f))
        
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)
        
        y = sigm_psi_f * x
        W_y = self.W(y)
        
        return W_y, sigm_psi_f


class MultiAttentionBlock2D(nn.Module):
    """
    Multi Attention Block for 2D
    
    封装 GridAttentionBlock2D 用于 UNet decoder
    """
    def __init__(self, in_channels, gate_channels, inter_channels=None, 
                 sub_sample_factor=2, norm_op=None, norm_op_kwargs=None):
        super().__init__()
        
        self.attention = GridAttentionBlock2D(
            in_channels=in_channels,
            gating_channels=gate_channels,
            inter_channels=inter_channels,
            sub_sample_factor=sub_sample_factor,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs
        )
        
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, g):
        att_out, att_map = self.attention(x, g)
        return self.combine(att_out), att_map


class ConvDropoutNormNonlin(nn.Module):
    """
    conv -> dropout -> norm -> nonlin
    """
    def __init__(self, input_channels, output_channels, 
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        
        ops = []
        ops.append(conv_op(input_channels, output_channels, **conv_kwargs))
        
        if dropout_op is not None:
            ops.append(dropout_op(**dropout_op_kwargs))
        
        if norm_op is not None:
            ops.append(norm_op(output_channels, **norm_op_kwargs))
        
        if nonlin is not None:
            ops.append(nonlin(**nonlin_kwargs))
        
        self.ops = nn.Sequential(*ops)
    
    def forward(self, x):
        return self.ops(x)


class StackedConvLayers(nn.Module):
    """
    堆叠的卷积层
    """
    def __init__(self, input_channels, output_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        super().__init__()
        
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        
        self.convs = nn.ModuleList()
        
        for i in range(num_convs):
            if i == 0 and first_stride is not None:
                conv_kwargs['stride'] = first_stride
            else:
                conv_kwargs['stride'] = 1
            
            self.convs.append(
                ConvDropoutNormNonlin(
                    input_channels if i == 0 else output_channels,
                    output_channels,
                    conv_op=conv_op,
                    conv_kwargs=conv_kwargs.copy(),
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            )
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, 
                 pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class AttentionUNetEncoder(nn.Module):
    """
    Attention UNet Encoder
    
    标准 UNet 编码器结构
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 pool_type: str = 'conv'):
        super().__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_conv_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        
        self.input_channels = input_channels
        self.output_channels = features_per_stage
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        conv_pad_sizes = []
        for krnl in kernel_sizes:
            if isinstance(krnl, (list, tuple)):
                conv_pad_sizes.append([i // 2 for i in krnl])
            else:
                conv_pad_sizes.append([krnl // 2] * 2)
        self.conv_pad_sizes = conv_pad_sizes
        
        self.stages = nn.ModuleList()
        self.pool_ops = nn.ModuleList()
        
        for s in range(n_stages):
            first_stride = strides[s] if s > 0 else 1
            
            stage = StackedConvLayers(
                input_channels if s == 0 else features_per_stage[s - 1],
                features_per_stage[s],
                n_conv_per_stage[s],
                conv_op=conv_op,
                conv_kwargs={'kernel_size': kernel_sizes[s], 'padding': conv_pad_sizes[s]},
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                first_stride=first_stride
            )
            self.stages.append(stage)
            
            if s < n_stages - 1:
                if pool_type == 'max':
                    self.pool_ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pool_type == 'avg':
                    self.pool_ops.append(nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.pool_ops.append(nn.Identity())
            else:
                self.pool_ops.append(nn.Identity())

        self.return_skips = return_skips

    def forward(self, x):
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)
            if s < len(self.stages) - 1:
                x = self.pool_ops[s](x)
        
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class AttentionUNetDecoder(nn.Module):
    """
    Attention UNet Decoder
    
    核心特点: 在每个 skip connection 处使用 attention gate
    """
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 attention_dsample: int = 2):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        
        n_stages_encoder = len(encoder.output_channels)
        n_stages_decoder = n_stages_encoder - 1
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages_decoder
        assert len(n_conv_per_stage) == n_stages_decoder

        self.stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for s in range(n_stages_decoder):
            input_features_below = encoder.output_channels[-(s + 1)]
            input_features_skip = encoder.output_channels[-(s + 2)]
            
            self.attention_blocks.append(
                MultiAttentionBlock2D(
                    in_channels=input_features_skip,
                    gate_channels=input_features_below,
                    inter_channels=input_features_skip // 2,
                    sub_sample_factor=attention_dsample,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs
                )
            )
            
            stride_for_upsampling = encoder.strides[-(s + 1)]
            self.upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode='nearest'
                )
            )
            
            self.stages.append(
                StackedConvLayers(
                    2 * input_features_skip,
                    input_features_skip,
                    n_conv_per_stage[s],
                    conv_op=encoder.conv_op,
                    conv_kwargs={'kernel_size': encoder.kernel_sizes[-(s + 2)], 
                                'padding': encoder.conv_pad_sizes[-(s + 2)]},
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    dropout_op=encoder.dropout_op,
                    dropout_op_kwargs=encoder.dropout_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs
                )
            )
            
            self.seg_layers.append(
                encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            )

    def forward(self, skips, input_size=None):
        lres_input = skips[-1]
        seg_outputs = []
        
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            
            skip_feature = skips[-(s + 2)]
            skip_feature, _ = self.attention_blocks[s](skip_feature, lres_input)
            
            if x.shape[2:] != skip_feature.shape[2:]:
                skip_feature = F.interpolate(
                    skip_feature, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            x = torch.cat((x, skip_feature), 1)
            x = self.stages[s](x)
            
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[s](x))
            
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        
        return seg_outputs


class AttentionUNet(nn.Module):
    """
    Attention U-Net for 2D Medical Image Segmentation
    
    基于论文: "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., MIDL 2018)
    
    核心创新:
    - 在 skip connection 处引入 attention gate
    - 让模型学会"关注什么"，自动抑制无关区域
    - 无需额外的监督信号
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 attention_dsample: int = 2):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("🚀 Attention UNet 初始化")
        print(f"   n_stages: {n_stages}")
        print(f"   features_per_stage: {features_per_stage}")
        print(f"   deep_supervision: {deep_supervision}")
        print(f"   attention_dsample: {attention_dsample}")
        print("=" * 60 + "\n")
        
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)
        
        self.encoder = AttentionUNetEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
        )

        self.decoder = AttentionUNetDecoder(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision,
            attention_dsample=attention_dsample
        )
        
        self.deep_supervision = deep_supervision

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips, input_size=x.shape[2:])
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == 2, "Input size must be [H, W]"
        
        output = 0
        current_size = input_size
        
        for s in range(len(self.encoder.stages)):
            output += self.encoder.output_channels[s] * np.prod(current_size)
            if s < len(self.encoder.stages) - 1:
                current_size = [i // self.encoder.strides[s + 1] for i in current_size]
        
        for s in range(len(self.decoder.stages)):
            current_size = [i * self.encoder.strides[-(s + 1)] for i in current_size]
            output += 2 * self.encoder.output_channels[-(s + 2)] * np.prod(current_size)
        
        return output


import math


def get_attention_unet_from_plans(plans_manager: PlansManager,
                                   dataset_json: dict,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   deep_supervision: bool = True):
    """
    从 plans 配置创建 Attention UNet
    """
    num_stages = configuration_manager.num_conv_per_stage_encoder.__len__()
    
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    
    norm_op = get_matching_instancenorm(conv_op=conv_op)
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    
    dropout_op = None
    dropout_op_kwargs = None
    
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {'inplace': True}
    
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads
    
    model = AttentionUNet(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=configuration_manager.unet_base_num_features,
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        n_conv_per_stage=configuration_manager.num_conv_per_stage_encoder,
        num_classes=num_classes,
        n_conv_per_stage_decoder=configuration_manager.num_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        dropout_op=dropout_op,
        dropout_op_kwargs=dropout_op_kwargs,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        deep_supervision=deep_supervision,
        attention_dsample=2
    )
    
    return model
