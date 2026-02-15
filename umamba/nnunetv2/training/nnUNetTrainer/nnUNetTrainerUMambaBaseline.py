import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans

class nnUNetTrainerUMambaBaseline(nnUNetTrainer):
    """
    【Baseline 完美对照组】
    严格控制所有变量与 Ours (SDG) 一致：
    1. LR = 1e-3
    2. Gradient Clipping = 12.0
    3. Patch Size 必须在 plans.json 或 运行时保证是 [384, 384]
    4. 唯一变量：SDG 关闭 (enable_sdg=False)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # ✅ 1. 强制对齐学习率
        self.initial_lr = 1e-3
        self.num_epochs = 500
        print("🐢 [Baseline] Initial LR set to 1e-3 (Matched with Ours)")

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        # ⚠️ 检查 Patch Size 是否一致
        # 如果你的 plans.json 还没改回来，这里应该会自动读到 [384, 384]
        # 如果这里读到的是 [512, 512]，你需要在 plans 文件里改，或者在这里强制报错提醒自己
        current_ps = configuration_manager.patch_size
        print(f"🐢 [Baseline] Current Patch Size: {current_ps}")
        
        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(
                plans_manager, 
                dataset_json, 
                configuration_manager,
                num_input_channels, 
                deep_supervision=enable_deep_supervision,
                # ✅ 2. 唯一的不同：关闭 SDG
                enable_sdg=False 
            )
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(
                plans_manager, 
                dataset_json, 
                configuration_manager,
                num_input_channels, 
                deep_supervision=enable_deep_supervision
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("🐢 [Baseline] SDG Module: DISABLED ❌")
        return model

    # ✅ 3. 强制对齐梯度裁剪
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # 保持 FP16 (Autocast)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            
            # 🔥 必须加！为了公平！
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}