import torch
from torch import nn
from typing import Optional, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerUMambaSDG(nnUNetTrainerUMambaBot):
    """
    UMamba + SDG trainer
    - Hierarchical heterogeneous skip configuration
    - Optional lightweight auxiliary head
    """

    # ====== 实验配置：以后做消融时直接改这里 ======
    DEFAULT_SKIP_MODES: Optional[List[str]] = None   # None 表示自动配置
    DEFAULT_ENABLE_AUX_HEAD: bool = True
    DEFAULT_AUX_HEAD_STAGE: int = 1
    DEFAULT_AUX_LOSS_WEIGHT: float = 0.4

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # 保存配置到实例，供 train/val step 使用
        self.skip_modes = self.DEFAULT_SKIP_MODES
        self.enable_aux_head = self.DEFAULT_ENABLE_AUX_HEAD
        self.aux_head_stage = self.DEFAULT_AUX_HEAD_STAGE
        self.aux_loss_weight = self.DEFAULT_AUX_LOSS_WEIGHT

        # auxiliary head 使用单输出普通损失，不复用 nnUNet deep supervision wrapper
        self.aux_ce_loss = nn.CrossEntropyLoss()

        print("🔥🔥🔥 成功加载了改进版 Trainer (SDG V3 - 分层异构配置)！🔥🔥🔥")
        print(f"📊 当前训练配置: num_epochs={self.num_epochs}, initial_lr={self.initial_lr}")
        print(f"📋 skip_modes: {self.skip_modes if self.skip_modes is not None else '自动配置'}")
        print(f"🔧 enable_aux_head: {self.enable_aux_head}, aux_head_stage: {self.aux_head_stage}")
        print(f"⚖️ aux_loss_weight: {self.aux_loss_weight}")

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
        **kwargs
    ) -> nn.Module:

        skip_modes = nnUNetTrainerUMambaSDG.DEFAULT_SKIP_MODES
        enable_aux_head = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_AUX_HEAD
        aux_head_stage = nnUNetTrainerUMambaSDG.DEFAULT_AUX_HEAD_STAGE

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
                enable_sdg=True,
                skip_modes=skip_modes,
                enable_aux_head=enable_aux_head,
                aux_head_stage=aux_head_stage
            )
        elif len(configuration_manager.patch_size) == 3:
            raise NotImplementedError("SDG-Block 3D version not implemented yet")
        else:
            raise NotImplementedError("Only 2D models are supported for SDG-Block currently")

        print("🚀🚀🚀 [Ours Mode] UMambaBot with SDG-Block ENABLED! 🚀🚀🚀")
        print(f"📋 build_network_architecture -> skip_modes: {skip_modes if skip_modes is not None else '自动配置'}")
        print(f"🔧 build_network_architecture -> enable_aux_head: {enable_aux_head}, aux_head_stage: {aux_head_stage}")

        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output = self.network(data)

            aux_output = None
            if self.enable_aux_head and isinstance(output, tuple):
                main_output, aux_output = output
            else:
                main_output = output

            # 主损失：继续使用 nnUNet 原生 deep supervision loss
            l = self.loss(main_output, target)

            # auxiliary loss：单独使用普通单输出 CE loss
            if self.enable_aux_head and aux_output is not None:
                aux_target = target[0] if isinstance(target, list) else target

                # CE 需要 target 为 [B, H, W] 且 dtype 为 long
                if aux_target.ndim == 4 and aux_target.shape[1] == 1:
                    aux_target = aux_target[:, 0].long()
                else:
                    aux_target = aux_target.long()

                aux_loss = self.aux_ce_loss(aux_output, aux_target)
                l = l + self.aux_loss_weight * aux_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = self.network(data)

                # 如果启用了 auxiliary head，则只取主输出
                if self.enable_aux_head and isinstance(output, tuple):
                    main_output, _ = output
                else:
                    main_output = output

                # ---- 兼容 deep supervision / 非 deep supervision 两种主输出格式 ----
                if isinstance(main_output, (list, tuple)):
                    # 主输出本身就是 deep supervision 多尺度输出
                    l = self.loss(main_output, target)
                    output_for_metrics = main_output[0]
                    target_for_metrics = target[0] if isinstance(target, list) else target
                else:
                    # 主输出是单个 Tensor，但 target 可能仍是 list
                    if isinstance(target, list):
                        l = self.loss((main_output,), (target[0],))
                        target_for_metrics = target[0]
                    else:
                        l = self.loss(main_output, target)
                        target_for_metrics = target
                    output_for_metrics = main_output

            axes = [0] + list(range(2, output_for_metrics.ndim))

            if self.label_manager.has_regions:
                if output_for_metrics.shape[1] == 1:
                    predicted_segmentation_onehot = (torch.sigmoid(output_for_metrics) > 0.5).float()
                else:
                    predicted_segmentation_onehot = torch.sigmoid(output_for_metrics)

                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    predicted_segmentation_onehot,
                    target_for_metrics,
                    axes=axes
                )
            else:
                output_seg = output_for_metrics.argmax(1)[:, None]

                predicted_segmentation_onehot = torch.zeros(
                    output_for_metrics.shape,
                    device=output_for_metrics.device,
                    dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)

                target_onehot = torch.zeros(
                    output_for_metrics.shape,
                    device=output_for_metrics.device,
                    dtype=torch.float32
                )
                target_onehot.scatter_(1, target_for_metrics.long(), 1)

                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    predicted_segmentation_onehot,
                    target_onehot,
                    axes=axes
                )

                # 去掉 background 通道
                tp = tp[1:]
                fp = fp[1:]
                fn = fn[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }
    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        During actual validation/export, nnUNet predictor expects network(x) to return a Tensor.
        Our network may return (main_output, aux_output) when auxiliary head is enabled.
        Here we temporarily attach a forward hook to strip aux output and only keep main output.
        """
        if not self.enable_aux_head:
            # No aux branch -> use parent implementation directly
            return super().perform_actual_validation(save_probabilities)

        def _unwrap_aux_output_hook(module, input, output):
            if isinstance(output, tuple):
                return output[0]  # keep only main output
            return output

        hook_handle = self.network.register_forward_hook(_unwrap_aux_output_hook)

        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            hook_handle.remove()