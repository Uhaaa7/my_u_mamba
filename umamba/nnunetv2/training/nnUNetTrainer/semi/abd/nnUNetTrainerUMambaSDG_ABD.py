"""
ABD (Attention-Based Dual-branch) 半监督训练器

基于 ABD 论文实现的半监督医学图像分割训练器

核心机制:
1. 双分支独立网络 (branch1 和 branch2)
2. Cross Teaching: 两个分支互相提供伪标签
3. ABD-I: 标注样本上的双向 Patch 位移增强
4. ABD-R: 无标注样本上的置信度引导 Patch 位移

骨干网络: UMambaSDG (通过 enable_sdg=True 启用)

与 CPS 的区别:
- ABD 使用 weak/strong 两种增强视图
- ABD 引入了基于置信度的 Patch 位移机制
- ABD 有两阶段训练 (ABD-I 和 ABD-R)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np
from typing import Optional, Dict, List

try:
    from ....nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG
except ModuleNotFoundError:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG

try:
    from .abd_wrapper import ABDDualBranchWrapper
    from .abd_utils import extract_main_logits, ABD_I, ABD_R, get_confidence_map
    from .abd_loss import DiceLoss, compute_supervised_loss, compute_pseudo_supervision_loss
except ImportError:
    from nnunetv2.training.nnUNetTrainer.semi.abd.abd_wrapper import ABDDualBranchWrapper
    from nnunetv2.training.nnUNetTrainer.semi.abd.abd_utils import (
        extract_main_logits, ABD_I, ABD_R, get_confidence_map
    )
    from nnunetv2.training.nnUNetTrainer.semi.abd.abd_loss import (
        DiceLoss, compute_supervised_loss, compute_pseudo_supervision_loss
    )

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


def _identity_transform(**data_dict):
    return data_dict


def _create_crop_transform(target_size):
    """
    创建一个简单的中心裁剪 transform
    
    Args:
        target_size: 目标尺寸 (H, W)
    
    Returns:
        transform 函数
    """
    from batchgenerators.transforms.abstract_transforms import AbstractTransform
    import numpy as np
    
    class CenterCropTransform(AbstractTransform):
        def __init__(self, target_size):
            self.target_size = target_size
        
        def __call__(self, **data_dict):
            data = data_dict.get('data')
            target = data_dict.get('target')
            seg = data_dict.get('seg')
            
            if data is not None:
                b, c, h, w = data.shape
                th, tw = self.target_size
                
                if h != th or w != tw:
                    start_h = (h - th) // 2
                    start_w = (w - tw) // 2
                    data_dict['data'] = data[:, :, start_h:start_h+th, start_w:start_w+tw]
            
            if target is not None:
                if isinstance(target, list):
                    target = target[0]
                if target.ndim == 4:
                    b, c, h, w = target.shape
                    th, tw = self.target_size
                    if h != th or w != tw:
                        start_h = (h - th) // 2
                        start_w = (w - tw) // 2
                        data_dict['target'] = target[:, :, start_h:start_h+th, start_w:start_w+tw]
                elif target.ndim == 3:
                    b, h, w = target.shape
                    th, tw = self.target_size
                    if h != th or w != tw:
                        start_h = (h - th) // 2
                        start_w = (w - tw) // 2
                        data_dict['target'] = target[:, start_h:start_h+th, start_w:start_w+tw]
            
            if seg is not None:
                if isinstance(seg, list):
                    seg = seg[0]
                if seg.ndim == 4:
                    b, c, h, w = seg.shape
                    th, tw = self.target_size
                    if h != th or w != tw:
                        start_h = (h - th) // 2
                        start_w = (w - tw) // 2
                        data_dict['seg'] = seg[:, :, start_h:start_h+th, start_w:start_w+tw]
                elif seg.ndim == 3:
                    b, h, w = seg.shape
                    th, tw = self.target_size
                    if h != th or w != tw:
                        start_h = (h - th) // 2
                        start_w = (w - tw) // 2
                        data_dict['seg'] = seg[:, start_h:start_h+th, start_w:start_w+tw]
            
            return data_dict
    
    return CenterCropTransform(target_size)


class nnUNetTrainerUMambaSDG_ABD(nnUNetTrainerUMambaSDG):
    """
    ABD (Attention-Based Dual-branch) 半监督训练器
    
    继承自 nnUNetTrainerUMambaSDG，确保使用 UMambaSDG 骨干网络
    
    核心特性:
    1. 双分支结构 (branch1 和 branch2)，都是 UMambaSDG
    2. Cross Teaching 机制
    3. ABD-I: 标注样本 Patch 位移
    4. ABD-R: 无标注样本置信度引导 Patch 位移
    """
    
    DEFAULT_LABELED_BS: int = 6
    DEFAULT_CONSISTENCY: float = 0.1
    DEFAULT_WARMUP_EPOCHS: int = 15
    DEFAULT_RAMPUP_EPOCHS: int = 40
    DEFAULT_PATCH_GRID_SIZE: int = 4
    DEFAULT_TOP_NUM: int = 4
    DEFAULT_DISABLE_ABD_I_RATIO: float = 0.8
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 2
    
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
        
        self.batch_size = 6
        self.labeled_bs = 3
        self.consistency = self.DEFAULT_CONSISTENCY
        self.warmup_epochs = self.DEFAULT_WARMUP_EPOCHS
        self.rampup_epochs = self.DEFAULT_RAMPUP_EPOCHS
        self.patch_grid_size = self.DEFAULT_PATCH_GRID_SIZE
        self.top_num = self.DEFAULT_TOP_NUM
        self.disable_abd_i_ratio = self.DEFAULT_DISABLE_ABD_I_RATIO
        self.gradient_accumulation_steps = self.DEFAULT_GRADIENT_ACCUMULATION_STEPS
        
        self.initial_lr = 0.005
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = None
        
        self.global_step = 0
        
        print("🔥🔥🔥 成功加载 ABD 半监督训练器! 🔥🔥🔥")
        print(f"📊 Per-step batch size: {self.batch_size} (labeled={self.labeled_bs}, unlabeled={self.batch_size - self.labeled_bs})")
        print(f"📊 Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"📊 Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"📊 Consistency weight: {self.consistency}")
        print(f"📊 Warmup epochs: {self.warmup_epochs}, Rampup epochs: {self.rampup_epochs}")
        print(f"📊 Patch grid size: {self.patch_grid_size}x{self.patch_grid_size}")
        print(f"📊 Top num: {self.top_num}")
        print(f"📊 Initial LR: {self.initial_lr}")
    
    def _prepare_seg_target(self, target):
        """
        统一把标签整理成 [B, H, W] long tensor
        """
        if isinstance(target, list):
            target = target[0]

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]

        if target.ndim != 3:
            raise ValueError(f"Expected target shape [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")

        return target.long().contiguous()
    
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        构建双分支网络
        两个分支都是 UMambaSDG，参数独立
        这里强制关闭 deep supervision，避免 ABD 训练时 target/output 语义混乱
        """
        skip_modes = nnUNetTrainerUMambaSDG.DEFAULT_SKIP_MODES
        enable_aux_head = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_AUX_HEAD
        aux_head_stage = nnUNetTrainerUMambaSDG.DEFAULT_AUX_HEAD_STAGE
        enable_adaptive_upsample = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_ADAPTIVE_UPSAMPLE

        force_deep_supervision = False

        if len(configuration_manager.patch_size) == 2:
            branch1 = get_umamba_bot_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=force_deep_supervision,
                enable_sdg=True,
                skip_modes=skip_modes,
                enable_aux_head=enable_aux_head,
                aux_head_stage=aux_head_stage,
                enable_adaptive_upsample=enable_adaptive_upsample
            )

            branch2 = get_umamba_bot_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=force_deep_supervision,
                enable_sdg=True,
                skip_modes=skip_modes,
                enable_aux_head=enable_aux_head,
                aux_head_stage=aux_head_stage,
                enable_adaptive_upsample=enable_adaptive_upsample
            )

            model = ABDDualBranchWrapper(
                branch1,
                branch2,
                enable_aux_head=enable_aux_head,
                enable_deep_supervision=force_deep_supervision
            )
        else:
            raise NotImplementedError("ABD only supports 2D models currently")

        print(" [ABD Mode] Dual UMambaSDG with ABD mechanism ENABLED! ")
        print(" [ABD Fix] Deep supervision forced OFF for stable semi-supervised training.")
        return model


    def set_deep_supervision_enabled(self, enabled: bool):
        """
        设置深度监督开关
        
        需要同时设置两个分支的 deep supervision
        """
        if hasattr(self.network, 'branch1') and hasattr(self.network, 'branch2'):
            if hasattr(self.network.branch1, 'decoder'):
                self.network.branch1.decoder.deep_supervision = enabled
            if hasattr(self.network.branch2, 'decoder'):
                self.network.branch2.decoder.deep_supervision = enabled
        elif hasattr(self.network, 'decoder'):
            self.network.decoder.deep_supervision = enabled
    
    def get_dataloaders(self):
        """
        重写数据加载逻辑
        
        构建 labeled 和 unlabeled 两个 dataloader
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        
        tr_keys, val_keys = self.do_split()
        
        labeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'labeled.txt')
        unlabeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'unlabeled.txt')
        
        if not os.path.exists(labeled_txt_path) or not os.path.exists(unlabeled_txt_path):
            raise FileNotFoundError(f"找不到半监督列表 {labeled_txt_path} 或 {unlabeled_txt_path}")
        
        with open(labeled_txt_path, 'r') as f:
            labeled_cases = [line.strip() for line in f.readlines() if line.strip()]
        with open(unlabeled_txt_path, 'r') as f:
            unlabeled_cases = [line.strip() for line in f.readlines() if line.strip()]
        
        tr_labeled_keys = [k for k in labeled_cases if k in tr_keys]
        tr_unlabeled_keys = [k for k in unlabeled_cases if k in tr_keys]
        
        dataset_tr_labeled = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_labeled_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        dataset_tr_unlabeled = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_unlabeled_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        
        labeled_bs = min(self.labeled_bs, self.batch_size)
        unlabeled_bs = self.batch_size - labeled_bs
        
        dl_cls = nnUNetDataLoader2D if dim == 2 else nnUNetDataLoader3D
        
        dl_tr_labeled = dl_cls(
            dataset_tr_labeled,
            labeled_bs,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent
        )
        
        dl_tr_unlabeled = dl_cls(
            dataset_tr_unlabeled,
            unlabeled_bs,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=0.0
        )
        
        dl_val = dl_cls(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent
        )
        
        allowed_num_processes = get_allowed_n_proc_DA()
        
        crop_transform = _create_crop_transform(self.configuration_manager.patch_size)
        
        if allowed_num_processes == 0:
            mt_gen_train_labeled = SingleThreadedAugmenter(dl_tr_labeled, tr_transforms)
            mt_gen_train_unlabeled = SingleThreadedAugmenter(dl_tr_unlabeled, crop_transform)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            num_proc = max(1, allowed_num_processes // 2)
            mt_gen_train_labeled = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr_labeled,
                transform=tr_transforms,
                num_processes=num_proc,
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == 'cuda'
            )
            mt_gen_train_unlabeled = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr_unlabeled,
                transform=crop_transform,
                num_processes=num_proc,
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == 'cuda'
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=num_proc,
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == 'cuda'
            )
        
        self.dataloader_unlabeled = mt_gen_train_unlabeled
        self.dataset_tr_unlabeled = dataset_tr_unlabeled
        self.dl_tr_unlabeled = dl_tr_unlabeled
        
        return mt_gen_train_labeled, mt_gen_val
    
    def on_train_start(self):
        """
        训练开始前的准备工作
        """
        super().on_train_start()
        
        num_classes = self.label_manager.num_segmentation_heads
        self.dice_loss = DiceLoss(num_classes)
        
        self.logger.my_fantastic_logging['loss_branch1'] = list()
        self.logger.my_fantastic_logging['loss_branch2'] = list()
        self.logger.my_fantastic_logging['loss_pseudo'] = list()
        self.logger.my_fantastic_logging['consistency_weight'] = list()
        
        print(f"📊 Labeled dataloader iterations: {len(self.dataloader_train)}")
        print(f"📊 Unlabeled dataloader iterations: {len(self.dataloader_unlabeled)}")
        print(f"📊 Number of classes: {num_classes}")
    
    def get_patch_params(self) -> tuple:
        """
        计算 patch 相关参数
        
        基于 patch_grid_size 动态计算 patch_size，支持非正方形输入
        
        Returns:
            (patch_h, patch_w, h_size, w_size): patch 高宽和网格数
        """
        H, W = self.configuration_manager.patch_size
        h_size = self.patch_grid_size
        w_size = self.patch_grid_size
        patch_h = H // h_size
        patch_w = W // w_size
        return patch_h, patch_w, h_size, w_size
    
    def get_current_consistency_weight(self) -> float:
        """
        获取当前一致性权重 (warmup + rampup 策略)
        
        与其他框架保持一致
        """
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        elif self.current_epoch < self.warmup_epochs + self.rampup_epochs:
            return self.consistency * (self.current_epoch - self.warmup_epochs + 1) / self.rampup_epochs
        return self.consistency
    
    def train_step(self, batch: dict, accumulation_step: int = 0) -> dict:
        """
        修复版 train_step:
        1. labeled supervised loss
        2. unlabeled cross teaching loss
        3. 暂时关闭 ABD-I / ABD-R，先保证训练逻辑正确且稳定
        """
        labeled_batch = batch['labeled']
        unlabeled_batch = batch['unlabeled']

        data_labeled = labeled_batch['data']
        target_labeled = labeled_batch['target']
        data_unlabeled = unlabeled_batch['data']

        if isinstance(data_labeled, np.ndarray):
            data_labeled = torch.from_numpy(data_labeled)
        data_labeled = data_labeled.to(self.device, non_blocking=True)

        if isinstance(target_labeled, list):
            target_labeled = [t.to(self.device, non_blocking=True) for t in target_labeled]
        else:
            target_labeled = target_labeled.to(self.device, non_blocking=True)

        if isinstance(data_unlabeled, np.ndarray):
            data_unlabeled = torch.from_numpy(data_unlabeled)
        data_unlabeled = data_unlabeled.to(self.device, non_blocking=True)

        target_for_loss = self._prepare_seg_target(target_labeled)
        labeled_bs = data_labeled.shape[0]

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            output1_labeled = self.network(data_labeled, branch='branch1')
            output2_labeled = self.network(data_labeled, branch='branch2')
            output1_unlabeled = self.network(data_unlabeled, branch='branch1')
            output2_unlabeled = self.network(data_unlabeled, branch='branch2')

            main_output1_labeled = extract_main_logits(output1_labeled)
            main_output2_labeled = extract_main_logits(output2_labeled)
            main_output1_unlabeled = extract_main_logits(output1_unlabeled)
            main_output2_unlabeled = extract_main_logits(output2_unlabeled)

            loss1 = compute_supervised_loss(
                main_output1_labeled, target_for_loss, self.ce_loss, self.dice_loss
            )
            loss2 = compute_supervised_loss(
                main_output2_labeled, target_for_loss, self.ce_loss, self.dice_loss
            )

            if main_output1_unlabeled.shape[0] > 0:
                pseudo_outputs1_u = torch.argmax(
                    torch.softmax(main_output1_unlabeled.detach(), dim=1), dim=1
                )
                pseudo_outputs2_u = torch.argmax(
                    torch.softmax(main_output2_unlabeled.detach(), dim=1), dim=1
                )

                pseudo_supervision1 = compute_pseudo_supervision_loss(
                    main_output1_unlabeled, pseudo_outputs2_u, self.dice_loss
                )
                pseudo_supervision2 = compute_pseudo_supervision_loss(
                    main_output2_unlabeled, pseudo_outputs1_u, self.dice_loss
                )
            else:
                pseudo_supervision1 = torch.tensor(0.0, device=self.device)
                pseudo_supervision2 = torch.tensor(0.0, device=self.device)

            # 暂时关闭 ABD-I / ABD-R
            loss3 = torch.tensor(0.0, device=self.device)
            loss4 = torch.tensor(0.0, device=self.device)
            pseudo_supervision3 = torch.tensor(0.0, device=self.device)
            pseudo_supervision4 = torch.tensor(0.0, device=self.device)

            consistency_weight = self.get_current_consistency_weight()

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2
            total_loss = model1_loss + model2_loss

        total_loss = total_loss / self.gradient_accumulation_steps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        is_accumulation_step = (accumulation_step + 1) % self.gradient_accumulation_steps != 0
        if not is_accumulation_step:
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        log_dict = {
            'loss': total_loss.detach().cpu().numpy() * self.gradient_accumulation_steps,
            'loss_branch1': model1_loss.detach().cpu().numpy(),
            'loss_branch2': model2_loss.detach().cpu().numpy(),
            'loss_pseudo': (pseudo_supervision1 + pseudo_supervision2).detach().cpu().numpy(),
            'consistency_weight': consistency_weight
        }

        return log_dict
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        处理训练 epoch 结束时的输出
        """
        from nnunetv2.utilities.collate_outputs import collate_outputs
        
        outputs = collate_outputs(train_outputs)
        
        loss = np.mean(outputs['loss'])
        loss_branch1 = np.mean(outputs['loss_branch1'])
        loss_branch2 = np.mean(outputs['loss_branch2'])
        loss_pseudo = np.mean(outputs['loss_pseudo'])
        consistency_weight = np.mean(outputs['consistency_weight'])
        
        self.logger.log('train_losses', loss, self.current_epoch)
        self.logger.log('loss_branch1', loss_branch1, self.current_epoch)
        self.logger.log('loss_branch2', loss_branch2, self.current_epoch)
        self.logger.log('loss_pseudo', loss_pseudo, self.current_epoch)
        self.logger.log('consistency_weight', consistency_weight, self.current_epoch)
        
        print(f"Epoch {self.current_epoch}: total_loss={loss:.4f}, "
              f"branch1={loss_branch1:.4f}, branch2={loss_branch2:.4f}, "
              f"pseudo={loss_pseudo:.4f}, consistency={consistency_weight:.4f}")
    
    def run_training(self):
        """
        重写训练循环以支持双 dataloader 和梯度累加
        """
        self.on_train_start()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            self.on_epoch_start()
            
            train_outputs = []
            labeled_iter = iter(self.dataloader_train)
            unlabeled_iter = iter(self.dataloader_unlabeled)
            
            self.optimizer.zero_grad()
            
            for iter_idx in range(self.num_iterations_per_epoch):
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.dataloader_train)
                    labeled_batch = next(labeled_iter)
                
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.dataloader_unlabeled)
                    unlabeled_batch = next(unlabeled_iter)
                
                batch = {
                    'labeled': labeled_batch,
                    'unlabeled': unlabeled_batch
                }
                
                train_outputs.append(self.train_step(batch, iter_idx))
            
            self.on_train_epoch_end(train_outputs)
            
            with torch.no_grad():
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            
            self.on_epoch_end()
            
            if self._early_stop:
                self.print_to_log_file(f"Early stopping at epoch {self.current_epoch}")
                break
        
        self.on_train_end()
    
    def validation_step(self, batch: dict) -> dict:
        """
        验证步骤
        
        使用 branch1 进行验证
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
                output = self.network(data, branch='branch1')
                
                if self.enable_aux_head and isinstance(output, tuple):
                    main_output, _ = output
                else:
                    main_output = output
                
                if isinstance(main_output, (list, tuple)):
                    l = self.loss(main_output, target)
                    output_for_metrics = main_output[0]
                    target_for_metrics = target[0] if isinstance(target, list) else target
                else:
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
                
                if target_for_metrics.ndim == output_for_metrics.ndim:
                    target_onehot = target_for_metrics
                else:
                    target_onehot = torch.zeros(
                        output_for_metrics.shape,
                        device=output_for_metrics.device,
                        dtype=torch.float32
                    )
                    target_onehot.scatter_(1, target_for_metrics[:, None].long(), 1)
                
                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    predicted_segmentation_onehot,
                    target_onehot,
                    axes=axes
                )
            
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
        
        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """
        保存检查点
        
        保存双分支网络的状态
        """
        super().save_checkpoint(filename)
        
        checkpoint_path = os.path.join(self.output_folder, filename)
        
        if hasattr(self.network, 'branch1') and hasattr(self.network, 'branch2'):
            abd_state = {
                'branch1_state_dict': self.network.branch1.state_dict(),
                'branch2_state_dict': self.network.branch2.state_dict(),
                'epoch': self.current_epoch,
                'global_step': self.global_step
            }
            torch.save(abd_state, checkpoint_path.replace('.pth', '_abd.pth'))
    
    def load_checkpoint(self, filename: str) -> None:
        """
        加载检查点
        """
        super().load_checkpoint(filename)
        
        checkpoint_path = os.path.join(self.output_folder, filename)
        abd_checkpoint_path = checkpoint_path.replace('.pth', '_abd.pth')
        
        if os.path.exists(abd_checkpoint_path):
            abd_state = torch.load(abd_checkpoint_path, map_location=self.device)
            if hasattr(self.network, 'branch1') and hasattr(self.network, 'branch2'):
                self.network.branch1.load_state_dict(abd_state['branch1_state_dict'])
                self.network.branch2.load_state_dict(abd_state['branch2_state_dict'])
                self.global_step = abd_state.get('global_step', 0)
                print(f"✅ Loaded ABD checkpoint from {abd_checkpoint_path}")
