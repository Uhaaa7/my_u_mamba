from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
# 如果需要 3D 也请引入 get_umamba_bot_3d_from_plans

class nnUNetTrainerUMambaSDG(nnUNetTrainerUMambaBot):
    """
    【创新点训练器 Ours】
    继承自原版 Trainer，但在构建网络时强制打开 enable_sdg=True。
    用于跑带有 DCNv2 + H-SS2D 的改进模型。
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(
                plans_manager, 
                dataset_json, 
                configuration_manager,
                num_input_channels, 
                deep_supervision=enable_deep_supervision,
                enable_sdg=True  # <--- ✅ 显式开启 SDG 模块 (跑改进版)
            )
        
        elif len(configuration_manager.patch_size) == 3:
             raise NotImplementedError("SDG-Block 3D version not implemented yet")
        else:
            raise NotImplementedError("Only 2D models are supported for SDG-Block currently")
        
        print("🚀🚀🚀 [Ours Mode] UMambaBot with SDG-Block ENABLED! 🚀🚀🚀")

        return model