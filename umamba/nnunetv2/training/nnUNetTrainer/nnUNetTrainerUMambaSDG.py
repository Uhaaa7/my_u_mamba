import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans

class nnUNetTrainerUMambaSDG(nnUNetTrainerUMambaBot):
    """
    【创新点训练器 Ours】
    继承自原版 Trainer，但在构建网络时强制打开 enable_sdg=True。
    并且重写了 train_step 以加入梯度裁剪，防止 NaN。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # 调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 👇👇👇 【核心修改】降低初始学习率 👇👇👇
        # 默认是 1e-2 (0.01)，对于 BS=4 + Mamba 来说太激进了
        # 改为 1e-3 (0.001)，稳扎稳打
        self.initial_lr = 1e-3
        self.num_epochs = 500
        print("🔥🔥🔥 成功加载了我的修改版 Trainer！初始 LR = 1e-3 🔥🔥🔥")
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

    # 👇👇👇 把这个函数加在这里！这就加上了梯度裁剪 👇👇👇
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast 是 nnU-Net 默认开启的
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            
            # 🔥🔥🔥 关键修改：在 scaler.step 之前解包并裁剪梯度 🔥🔥🔥
            # 这就是防止 NaN 的绝对防御
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # 如果没用 scaler (极少见)，也加上裁剪
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}