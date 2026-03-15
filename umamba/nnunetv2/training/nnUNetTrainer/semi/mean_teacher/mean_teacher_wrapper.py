"""
Mean Teacher 网络包装器

在不修改原有网络结构的前提下:
1. 添加边界分支
2. 输出增强后的 skip 特征
3. 同时输出主分割结果、辅助分割结果、边界预测和 skip 特征

设计原则:
- 最小侵入式: 不修改原有 UMambaBot 结构
- 通过 forward hook 拦截 skip 特征
- 边界分支接入高分辨率 decoder 特征

修复:
1. 边界分支通道数在初始化时推断，确保 EMA 同步
2. forward_inference 直接调用 base_model，避免递归
3. InferenceOnlyWrapper 正确实现独立推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import copy

try:
    from .semi_supervised_modules import BoundaryHead
except ImportError:
    from nnunetv2.training.nnUNetTrainer.semi.mean_teacher.semi_supervised_modules import BoundaryHead


class MeanTeacherWrapper(nn.Module):
    """
    Mean Teacher 网络包装器
    
    包装原有的 UMambaBot 网络，添加:
    1. 边界分支 (BoundaryHead)
    2. Skip 特征收集机制
    
    输出格式:
    - 有标签模式: (main_output, aux_output, boundary_output, skip_features)
    - 无标签模式: 同上
    
    注意:
    - 原有网络结构完全保持不变
    - 边界分支通过 hook 接入 decoder 特征
    - Skip 特征通过 hook 从 SDG 模块收集
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 num_classes: int,
                 boundary_head_channels: int = None,
                 boundary_stage: int = -1):
        """
        Args:
            base_model: 原有的 UMambaBot 网络
            num_classes: 分割类别数
            boundary_head_channels: 边界分支隐藏通道数
            boundary_stage: 边界分支接入的 decoder stage (-1 表示最后一个 stage)
        """
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.boundary_stage = boundary_stage


        # 代理底层网络的关键属性，兼容 nnUNetTrainer 对 self.network.decoder / encoder 的访问
        self.decoder = self.base_model.decoder
        self.encoder = self.base_model.encoder

        
        decoder = base_model.decoder
        n_stages_decoder = len(decoder.stages)
        
        if boundary_stage < 0:
            boundary_stage = n_stages_decoder + boundary_stage
        self.boundary_stage_idx = min(boundary_stage, n_stages_decoder - 1)
        
        self.skip_features_buffer: List[torch.Tensor] = []
        self.decoder_features_buffer: List[torch.Tensor] = []
        
        self._register_hooks()
        
        self._has_aux_head = hasattr(decoder, 'enable_aux_head') and decoder.enable_aux_head
        self._has_deep_supervision = decoder.deep_supervision
        
        boundary_in_channels = self._infer_boundary_in_channels()
        if boundary_head_channels is None:
            boundary_head_channels = max(boundary_in_channels // 4, 16)
        
        self.boundary_head = BoundaryHead(
            in_channels=boundary_in_channels,
            hidden_channels=boundary_head_channels
        )
        
        print(f"🔧 MeanTeacherWrapper initialized:")
        print(f"   - Boundary head at decoder stage {self.boundary_stage_idx} (channels={boundary_in_channels})")
        print(f"   - Aux head enabled: {self._has_aux_head}")
        print(f"   - Deep supervision: {self._has_deep_supervision}")
    
    def _infer_boundary_in_channels(self) -> int:
        """
        推断边界分支的输入通道数
        
        根据 decoder stage 的结构推断输出通道数：
        - decoder.stages[i] 的输入来自 encoder 的对应层或上一级 decoder
        - 我们需要知道该 stage 的输出通道数
        
        对于 UMambaBot decoder:
        - stages[0] 输出通道 = encoder.output_channels[-1] (最深层 encoder 特征)
        - stages[1] 输出通道 = stages[0] 的输出通道 (通常保持或减半)
        - 以此类推...
        
        更可靠的方法：从 decoder 的配置推断
        """
        decoder = self.base_model.decoder
        encoder = decoder.encoder
        encoder_output_channels = encoder.output_channels
        
        n_stages_decoder = len(decoder.stages)
        stage_idx = self.boundary_stage_idx
        
        if hasattr(decoder, 'transpconvs') and len(decoder.transpconvs) > 0:
            transpconv_channels = []
            for tc in decoder.transpconvs:
                if hasattr(tc, 'conv') and hasattr(tc.conv, 'out_channels'):
                    transpconv_channels.append(tc.conv.out_channels)
                elif hasattr(tc, 'out_channels'):
                    transpconv_channels.append(tc.out_channels)
            
            if stage_idx < len(transpconv_channels):
                return transpconv_channels[stage_idx]
        
        if hasattr(decoder, 'stages') and len(decoder.stages) > stage_idx:
            stage = decoder.stages[stage_idx]
            if hasattr(stage, 'out_channels'):
                return stage.out_channels
            
            for module in stage.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    return module.out_channels
        
        if stage_idx < n_stages_decoder - 1:
            return encoder_output_channels[-(stage_idx + 2)]
        else:
            return encoder_output_channels[0]
    
    def _register_hooks(self):
        """注册 forward hooks 收集 skip 特征和 decoder 特征"""
        
        decoder = self.base_model.decoder
        
        def make_skip_hook(idx):
            def hook(module, input, output):
                if len(self.skip_features_buffer) <= idx:
                    while len(self.skip_features_buffer) <= idx:
                        self.skip_features_buffer.append(None)
                self.skip_features_buffer[idx] = output
            return hook
        
        n_used_skips = len(decoder.skip_modes) if hasattr(decoder, 'skip_modes') else 0
        for i in range(n_used_skips):
            if hasattr(decoder, 'sdg_blocks') and i < len(decoder.sdg_blocks):
                decoder.sdg_blocks[i].register_forward_hook(make_skip_hook(i))
        
        def make_decoder_hook(idx):
            def hook(module, input, output):
                if len(self.decoder_features_buffer) <= idx:
                    while len(self.decoder_features_buffer) <= idx:
                        self.decoder_features_buffer.append(None)
                self.decoder_features_buffer[idx] = output
            return hook
        
        for i, stage in enumerate(decoder.stages):
            stage.register_forward_hook(make_decoder_hook(i))
    
    def _clear_buffers(self):
        """清空特征缓冲区"""
        self.skip_features_buffer = []
        self.decoder_features_buffer = []
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C_in, H, W]
        
        Returns:
            outputs: 包含以下键的字典
                - 'main_output': 主分割输出 [B, C, H, W] 或 List[Tensor]
                - 'aux_output': 辅助分割输出 [B, C, H, W] (如果启用)
                - 'boundary_output': 边界预测 [B, 1, H, W]
                - 'skip_features': 增强后的 skip 特征列表
        """
        self._clear_buffers()
        
        input_size = x.shape[2:]
        
        base_output = self.base_model(x)
        
        outputs = {}
        
        if self._has_aux_head and isinstance(base_output, tuple):
            main_output, aux_output = base_output
            outputs['main_output'] = main_output
            outputs['aux_output'] = aux_output
        else:
            outputs['main_output'] = base_output
            outputs['aux_output'] = None
        
        if self.boundary_stage_idx < len(self.decoder_features_buffer):
            decoder_feat = self.decoder_features_buffer[self.boundary_stage_idx]
            if decoder_feat is not None:
                boundary_output = self.boundary_head(decoder_feat, target_size=input_size)
                outputs['boundary_output'] = boundary_output
            else:
                outputs['boundary_output'] = None
        else:
            outputs['boundary_output'] = None
        
        valid_skip_features = [f for f in self.skip_features_buffer if f is not None]
        outputs['skip_features'] = valid_skip_features if valid_skip_features else None
        
        return outputs
    
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理专用前向传播，只返回主分割输出
        
        重要：这个方法直接调用 base_model，不经过 self.forward，
        因此不会触发 boundary head 和 skip feature 收集，
        也不会受到 monkey patch 的影响。
        
        Args:
            x: 输入图像 [B, C_in, H, W]
        
        Returns:
            main_output: 主分割输出 [B, C, H, W]
        """
        base_output = self.base_model(x)
        
        if self._has_aux_head and isinstance(base_output, tuple):
            main_output = base_output[0]
        else:
            main_output = base_output
        
        if isinstance(main_output, (list, tuple)):
            return main_output[0]
        return main_output
    
    def get_main_output_tensor(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从输出字典中获取主分割输出张量
        
        Args:
            outputs: forward() 返回的输出字典
        
        Returns:
            main_tensor: [B, C, H, W] 形状的主输出张量
        """
        main_output = outputs['main_output']
        
        if isinstance(main_output, (list, tuple)):
            return main_output[0]
        return main_output
    
    def get_all_main_outputs(self, outputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        获取所有尺度的主分割输出（用于 deep supervision）
        
        Args:
            outputs: forward() 返回的输出字典
        
        Returns:
            main_outputs: 主分割输出列表
        """
        main_output = outputs['main_output']
        
        if isinstance(main_output, (list, tuple)):
            return list(main_output)
        return [main_output]
    
    def get_aux_output_tensor(self, outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """获取辅助输出张量"""
        return outputs.get('aux_output')
    
    def get_boundary_output_tensor(self, outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """获取边界输出张量"""
        return outputs.get('boundary_output')
    
    def get_skip_features(self, outputs: Dict[str, torch.Tensor]) -> Optional[List[torch.Tensor]]:
        """获取 skip 特征列表"""
        return outputs.get('skip_features')


def wrap_model_for_mean_teacher(base_model: nn.Module,
                                num_classes: int,
                                boundary_head_channels: int = None,
                                boundary_stage: int = -1) -> MeanTeacherWrapper:
    """
    将基础模型包装为 Mean Teacher 兼容格式
    
    Args:
        base_model: 原有的 UMambaBot 网络
        num_classes: 分割类别数
        boundary_head_channels: 边界分支隐藏通道数
        boundary_stage: 边界分支接入的 decoder stage
    
    Returns:
        wrapped_model: 包装后的模型
    """
    return MeanTeacherWrapper(
        base_model=base_model,
        num_classes=num_classes,
        boundary_head_channels=boundary_head_channels,
        boundary_stage=boundary_stage
    )


class InferenceOnlyWrapper(nn.Module):
    """
    推理专用包装器
    
    这个包装器持有对 wrapped_model 的引用，只暴露主分割输出。
    通过调用 forward_inference 方法避免递归调用问题。
    
    注意：这个类不进行深拷贝，它只是一个轻量级的推理接口包装。
    """
    
    def __init__(self, wrapped_model: MeanTeacherWrapper):
        super().__init__()
        self.wrapped_model = wrapped_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理前向传播
        
        Args:
            x: 输入图像 [B, C_in, H, W]
        
        Returns:
            main_output: 主分割输出 [B, C, H, W]
        """
        return self.wrapped_model.forward_inference(x)


if __name__ == '__main__':
    print("Testing MeanTeacherWrapper...")
    
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_channels = [32, 64, 128, 256]
            self.strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
            self.conv_op = nn.Conv2d
            self.norm_op = nn.BatchNorm2d
            self.norm_op_kwargs = {}
            self.nonlin = nn.ReLU
            self.nonlin_kwargs = {'inplace': True}
            self.conv_bias = True
            self.kernel_sizes = [3, 3, 3, 3]
            self.conv_pad_sizes = [[1, 1], [1, 1], [1, 1], [1, 1]]
            
            self.stages = nn.ModuleList([
                nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()),
            ])
        
        def forward(self, x):
            skips = []
            for stage in self.stages:
                x = stage(x)
                skips.append(x)
            return skips
    
    class DummySDGBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    class DummyDecoder(nn.Module):
        def __init__(self, encoder, num_classes):
            super().__init__()
            self.encoder = encoder
            self.num_classes = num_classes
            self.deep_supervision = True
            self.enable_aux_head = True
            self.aux_head_stage = 1
            self.skip_modes = ['light', 'full', 'full']
            
            self.sdg_blocks = nn.ModuleList([
                DummySDGBlock(128),
                DummySDGBlock(64),
                DummySDGBlock(32),
            ])
            
            self.stages = nn.ModuleList([
                nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()),
            ])
            
            self.transpconvs = nn.ModuleList([
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
            ])
            
            self.seg_outputs = nn.ModuleList([
                nn.Conv2d(128, num_classes, 1),
                nn.Conv2d(64, num_classes, 1),
                nn.Conv2d(32, num_classes, 1),
            ])
            
            self.aux_head = nn.Conv2d(64, num_classes, 1)
        
        def forward(self, skips):
            x = skips[-1]
            outputs = []
            
            for i, (transpconv, stage, seg_out) in enumerate(zip(self.transpconvs, self.stages, self.seg_outputs)):
                x = transpconv(x)
                if i < len(skips) - 1:
                    skip_idx = len(skips) - 2 - i
                    if skip_idx >= 0 and skip_idx < len(self.sdg_blocks):
                        x = x + self.sdg_blocks[i](skips[skip_idx])
                x = stage(x)
                outputs.append(seg_out(x))
            
            aux_output = self.aux_head(self.stages[1](self.transpconvs[1](self.stages[0](self.transpconvs[0](skips[-1])))))
            
            return outputs[::-1], aux_output
    
    class DummyUMambaBot(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.encoder = DummyEncoder()
            self.decoder = DummyDecoder(self.encoder, num_classes)
        
        def forward(self, x):
            skips = self.encoder(x)
            return self.decoder(skips)
    
    base_model = DummyUMambaBot(num_classes=3)
    wrapped = MeanTeacherWrapper(base_model, num_classes=3, boundary_stage=-1)
    
    x = torch.randn(2, 3, 64, 64)
    
    print("\nTesting forward()...")
    outputs = wrapped(x)
    print(f"  main_output shape: {outputs['main_output'][0].shape if isinstance(outputs['main_output'], list) else outputs['main_output'].shape}")
    print(f"  aux_output shape: {outputs['aux_output'].shape if outputs['aux_output'] is not None else None}")
    print(f"  boundary_output shape: {outputs['boundary_output'].shape if outputs['boundary_output'] is not None else None}")
    print(f"  skip_features count: {len(outputs['skip_features']) if outputs['skip_features'] else 0}")
    
    print("\nTesting forward_inference()...")
    main_out = wrapped.forward_inference(x)
    print(f"  main_output shape: {main_out.shape}")
    
    print("\nTesting InferenceOnlyWrapper...")
    inference_wrapper = InferenceOnlyWrapper(wrapped)
    main_out2 = inference_wrapper(x)
    print(f"  main_output shape: {main_out2.shape}")
    
    print("\n✅ All tests passed!")
