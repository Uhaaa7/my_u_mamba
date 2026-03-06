import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans

class nnUNetTrainerUMambaSDG(nnUNetTrainerUMambaBot):
    """
    【创新点训练器 Ours】
    继承自 nnUNetTrainerUMambaBot，唯一差异是开启 SDG 模块。
    所有超参数与 Baseline 完全一致，实现公平对比。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # 调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # ✅ 与 Baseline (nnUNetTrainerUMambaBot) 完全对齐的超参数
        # num_epochs=200, initial_lr=0.01 继承自 nnUNetTrainer
        print("🔥🔥🔥 成功加载了改进版 Trainer (SDG V2)！配置与 Baseline 一致 🔥🔥🔥")
        print(f"📊 当前训练配置: num_epochs={self.num_epochs}, initial_lr={self.initial_lr}")
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
