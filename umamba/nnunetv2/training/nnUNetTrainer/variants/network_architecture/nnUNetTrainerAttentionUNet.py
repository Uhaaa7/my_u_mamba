from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from torch import nn

from nnunetv2.nets.AttentionUNet_2d import AttentionUNet


class nnUNetTrainerAttentionUNet(nnUNetTrainer):
    """
    Attention U-Net Trainer for 2D Medical Image Segmentation
    
    基于论文: "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., MIDL 2018)
    
    核心特点:
    - 在 skip connection 处使用 attention gate
    - 自动学习关注重要区域，抑制无关背景
    - 适用于小目标分割任务
    """
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)
        
        norm_op = get_matching_instancenorm(conv_op)
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        
        dropout_op = None
        dropout_op_kwargs = None
        
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'inplace': True}
        
        features_per_stage = [
            min(configuration_manager.UNet_base_num_features * 2 ** i,
                configuration_manager.unet_max_num_features) 
            for i in range(num_stages)
        ]

        model = AttentionUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=enable_deep_supervision,
            attention_dsample=2
        )
        
        model.apply(InitWeights_He(1e-2))
        
        return model
