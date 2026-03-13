try:
    from .mia_ss2d import MIA_SS2D_Block as SDG_Block, VALID_MODES
except ImportError:
    from mia_ss2d import MIA_SS2D_Block as SDG_Block, VALID_MODES

import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple, Optional

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class LightAuxiliaryHead(nn.Module):
    """
    轻量辅助分割头
    
    设计原则:
    - 极简结构：仅一个 1x1 卷积 + 上采样
    - 不引入额外复杂分支
    - 用于训练期辅助监督
    """
    def __init__(self, in_channels, num_classes, target_size=None):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        
    def forward(self, x, target_size=None):
        out = self.conv(x)
        
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        elif self.target_size is not None:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
            
        return out
    
    def compute_conv_feature_map_size(self, input_size):
        # 1x1 卷积的特征图大小
        return np.prod([self.conv.out_channels, *input_size], dtype=np.int64)


class UNetResDecoder(nn.Module):
    """
    UNet 解码器
    
    本次修改:
    1. 分层异构配置: 不同 skip 层使用不同模式 (full / light / identity)
    2. 支持轻量 auxiliary head
    3. 修复: skip_modes 只配置实际使用的 skip 层 (n_stages_decoder - 1 层)
    4. 添加健壮性检查
    """
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, 
                 nonlin_first: bool = False, 
                 enable_sdg: bool = False,
                 skip_modes: Optional[List[str]] = None,
                 enable_aux_head: bool = False,
                 aux_head_stage: int = 1):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.enable_aux_head = enable_aux_head
        self.aux_head_stage = aux_head_stage
        
        n_stages_encoder = len(encoder.output_channels)
        n_stages_decoder = n_stages_encoder - 1
        
        # === 分层异构配置 ===
        # 修复: skip_modes 只配置实际使用的 skip 层
        # 在 forward 中，只有 s < (len(self.stages) - 1) 时才使用 skip
        # 所以实际使用的 skip 层数 = n_stages_decoder - 1
        
        n_used_skips = n_stages_decoder - 1  # 实际使用的 skip 层数
        
        if skip_modes is None:
            skip_modes = self._get_default_skip_modes(n_used_skips)
        
        # 健壮性检查: skip_modes 长度必须与实际使用的 skip 层数匹配
        assert len(skip_modes) == n_used_skips, \
            f"skip_modes length ({len(skip_modes)}) must match number of used skips ({n_used_skips})"
        
        # 健壮性检查: 所有 mode 必须合法
        for i, mode in enumerate(skip_modes):
            assert mode in VALID_MODES, \
                f"skip_modes[{i}] = '{mode}' is invalid, must be one of {VALID_MODES}"
        
        self.skip_modes = skip_modes
        print(f"📋 Skip connection modes (for {n_used_skips} used skips): {skip_modes}")
        
        self.sdg_blocks = nn.ModuleList()
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages_decoder
        assert len(n_conv_per_stage) == n_stages_decoder

        stages = []
        upsample_layers = []
        seg_layers = []

        for s in range(n_stages_decoder):
            input_features_below = encoder.output_channels[-(s + 1)]
            input_features_skip = encoder.output_channels[-(s + 2)]
            
            # === 根据 skip_modes 配置 SDG 模块 ===
            # 只有 s < n_used_skips 时才需要 SDG 模块
            if s < n_used_skips:
                mode = skip_modes[s]
                if enable_sdg and mode != 'identity':
                    self.sdg_blocks.append(SDG_Block(dim=input_features_skip, mode=mode))
                else:
                    self.sdg_blocks.append(nn.Identity())
            # 最后一个 stage 不使用 skip，不需要 SDG 模块

            stride_for_upsampling = encoder.strides[-(s + 1)]
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip if s < n_stages_decoder - 1 else input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 2)],
                    padding=encoder.conv_pad_sizes[-(s + 2)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 2)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        
        # === 轻量 Auxiliary Head ===
        if enable_aux_head:
            # aux_head_stage 指定从哪个 decoder stage 接入
            # aux_head_stage=0 表示最深层，aux_head_stage=n_stages_decoder-1 表示最浅层
            # 默认选择中间偏上的层 (aux_head_stage=1)
            aux_stage_idx = min(aux_head_stage, n_stages_decoder - 1)
            aux_channels = encoder.output_channels[-(aux_stage_idx + 2)]
            self.aux_head = LightAuxiliaryHead(
                in_channels=aux_channels,
                num_classes=num_classes
            )
            self.aux_stage_idx = aux_stage_idx
            print(f"🔧 Auxiliary head enabled at decoder stage {aux_stage_idx} (channels={aux_channels})")
        else:
            self.aux_head = None
            self.aux_stage_idx = -1

    def _get_default_skip_modes(self, n_used_skips):
        """
        获取默认的分层异构配置
        
        策略 (针对实际使用的 skip 层):
        - 最深层 (s=0): light - 语义强但空间粗，使用轻量模式
        - 中间层 (s=1, s=2, ...): full - 最适合跳跃连接重构
        - 最高分辨率层 (s=n-1): identity - 细节丰富，不做重增强
        
        例如 n_used_skips=4 时:
        skip_modes = ['light', 'full', 'full', 'identity']
        """
        if n_used_skips <= 1:
            return ['light']
        
        if n_used_skips == 2:
            return ['light', 'identity']
        
        modes = []
        for s in range(n_used_skips):
            if s == 0:
                modes.append('light')
            elif s == n_used_skips - 1:
                modes.append('identity')
            else:
                modes.append('full')
        
        return modes

    def forward(self, skips, input_size=None):
        """
        Args:
            skips: encoder 输出的 skip 特征列表
            input_size: 原始输入尺寸，用于 auxiliary head 上采样
        
        Returns:
            如果 enable_aux_head=False: 返回主分割输出 (与之前一致)
            如果 enable_aux_head=True: 返回 (主输出, aux输出) 或 [主输出列表, aux输出]
        """
        lres_input = skips[-1]
        seg_outputs = []
        aux_output = None
        
        n_used_skips = len(self.skip_modes)
        
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            
            if s < n_used_skips:
                skip_feature = skips[-(s + 2)]
                skip_feature = self.sdg_blocks[s](skip_feature)
                x = torch.cat((x, skip_feature), 1)
            
            x = self.stages[s](x)
            
            # === Auxiliary Head ===
            if self.enable_aux_head and self.aux_head is not None and s == self.aux_stage_idx:
                aux_output = self.aux_head(x, target_size=input_size)
            
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        # === 返回值处理 ===
        if self.enable_aux_head and aux_output is not None:
            if not self.deep_supervision:
                return seg_outputs[0], aux_output
            else:
                return seg_outputs, aux_output
        else:
            if not self.deep_supervision:
                return seg_outputs[0]
            else:
                return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        
        # 修复: 计入 auxiliary head 的特征图大小
        if self.enable_aux_head and self.aux_head is not None:
            aux_stage_size = skip_sizes[-(self.aux_stage_idx + 1)]
            output += self.aux_head.compute_conv_feature_map_size(aux_stage_size)
        
        return output


class UMambaBot(nn.Module):
    """
    UMambaBot 网络
    
    本次修改:
    1. 支持分层异构配置
    2. 支持轻量 auxiliary head
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
                 stem_channels: int = None,
                 enable_sdg: bool = False,
                 skip_modes: Optional[List[str]] = None,
                 enable_aux_head: bool = False,
                 aux_head_stage: int = 1
                 ):
        super().__init__()
        
        print("\n" + "=" * 60)
        print("🚀 UMambaBot 初始化")
        print(f"   enable_sdg: {enable_sdg}")
        print(f"   enable_aux_head: {enable_aux_head}")
        if enable_sdg and skip_modes is None:
            print("   skip_modes: 自动配置 (分层异构)")
        elif skip_modes is not None:
            print(f"   skip_modes: {skip_modes}")
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
        
        self.encoder = UNetResEncoder(
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
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels,
        )

        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])

        self.decoder = UNetResDecoder(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision, 
            enable_sdg=enable_sdg,
            skip_modes=skip_modes,
            enable_aux_head=enable_aux_head,
            aux_head_stage=aux_head_stage
        )
        
        self.enable_aux_head = enable_aux_head

    def forward(self, x):
        input_size = x.shape[2:]
        
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        
        return self.decoder(skips, input_size=input_size)


def get_umamba_bot_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        enable_sdg: bool = False,
        skip_modes: Optional[List[str]] = None,
        enable_aux_head: bool = False,
        aux_head_stage: int = 1
    ):
    """
    从 plans 配置构建 UMambaBot 网络
    
    新增参数:
        skip_modes: 自定义每个 skip 层的模式列表 ['light', 'full', 'identity', ...]
                    如果为 None，使用默认的分层异构配置
                    注意: 只配置实际使用的 skip 层 (n_stages_decoder - 1 层)
        enable_aux_head: 是否启用轻量辅助分割头
        aux_head_stage: auxiliary head 接入的 decoder stage 索引
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaBot'
    network_class = UMambaBot
    kwargs = {
        'UMambaBot': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        enable_sdg=enable_sdg,
        skip_modes=skip_modes,
        enable_aux_head=enable_aux_head,
        aux_head_stage=aux_head_stage,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model
