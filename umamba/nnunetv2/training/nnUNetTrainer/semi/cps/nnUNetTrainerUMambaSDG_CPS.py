"""
CPS (Cross Pseudo Supervision) 半监督训练器

基于 CPS 论文实现的半监督医学图像分割训练器

核心机制:
1. 双分支独立网络 (branch1 和 branch2)
2. 有标注样本分别进行监督训练
3. 无标注样本分别经两分支前向
4. 每个分支用对方分支的伪标签进行监督
5. 伪标签为 argmax 得到的 hard pseudo label
6. 总损失 = 两分支监督损失 + lambda_cps * cps_loss

骨干网络: UMambaSDG (通过 enable_sdg=True 启用)
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
    from .cps_wrapper import CPSDualBranchWrapper
    from .cps_loss import extract_main_logits, compute_cps_loss, CPSLoss
except ImportError:
    from nnunetv2.training.nnUNetTrainer.semi.cps.cps_wrapper import CPSDualBranchWrapper
    from nnunetv2.training.nnUNetTrainer.semi.cps.cps_loss import extract_main_logits, compute_cps_loss, CPSLoss

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


class nnUNetTrainerUMambaSDG_CPS(nnUNetTrainerUMambaSDG):
    """
    CPS (Cross Pseudo Supervision) 半监督训练器
    
    继承自 nnUNetTrainerUMambaSDG，确保使用 UMambaSDG 骨干网络
    
    核心特性:
    1. 双分支结构 (branch1 和 branch2)，都是 UMambaSDG
    2. 有标注样本分别计算监督损失
    3. 无标注样本计算 CPS loss (交叉伪标签监督)
    """
    
    DEFAULT_CPS_WEIGHT: float = 1.5
    DEFAULT_LABELED_BS: int = 3
    DEFAULT_WARMUP_EPOCHS: int = 15
    DEFAULT_RAMPUP_EPOCHS: int = 40
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
        self.cps_weight = self.DEFAULT_CPS_WEIGHT
        self.labeled_bs = self.DEFAULT_LABELED_BS
        self.warmup_epochs = self.DEFAULT_WARMUP_EPOCHS
        self.rampup_epochs = self.DEFAULT_RAMPUP_EPOCHS
        self.gradient_accumulation_steps = self.DEFAULT_GRADIENT_ACCUMULATION_STEPS
        
        self.initial_lr = 0.005
        
        self.cps_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        print("🔥🔥🔥 成功加载 CPS 半监督训练器! 🔥🔥🔥")
        print(f"📊 Per-step batch size: {self.batch_size} (labeled={self.labeled_bs}, unlabeled={self.batch_size - self.labeled_bs})")
        print(f"📊 Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"📊 Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"📊 CPS weight: {self.cps_weight}")
        print(f"📊 Warmup epochs: {self.warmup_epochs}")
        print(f"📊 Rampup epochs: {self.rampup_epochs}")
        print(f"📊 Initial LR: {self.initial_lr}")
    
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
        构建双分支 CPS 网络
        
        两个分支都使用 UMambaSDG
        """
        skip_modes = nnUNetTrainerUMambaSDG.DEFAULT_SKIP_MODES
        enable_aux_head = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_AUX_HEAD
        aux_head_stage = nnUNetTrainerUMambaSDG.DEFAULT_AUX_HEAD_STAGE
        enable_adaptive_upsample = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_ADAPTIVE_UPSAMPLE
        
        if len(configuration_manager.patch_size) == 2:
            branch1 = get_umamba_bot_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
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
                deep_supervision=enable_deep_supervision,
                enable_sdg=True,
                skip_modes=skip_modes,
                enable_aux_head=enable_aux_head,
                aux_head_stage=aux_head_stage,
                enable_adaptive_upsample=enable_adaptive_upsample
            )
            
            model = CPSDualBranchWrapper(
                branch1=branch1,
                branch2=branch2,
                enable_aux_head=enable_aux_head,
                enable_deep_supervision=enable_deep_supervision
            )
        else:
            raise NotImplementedError("CPS only supports 2D models currently")
        
        print("🚀🚀🚀 [CPS Mode] Dual UMambaSDG Networks ENABLED! 🚀🚀🚀")
        print(f"📋 Branch1 parameters: {sum(p.numel() for p in branch1.parameters()):,}")
        print(f"📋 Branch2 parameters: {sum(p.numel() for p in branch2.parameters()):,}")
        print(f"📋 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def get_dataloaders(self):
        """
        构建数据加载器
        
        使用两个 dataloader:
        - labeled dataloader: 有标注样本
        - unlabeled dataloader: 无标注样本
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
        
        labeled_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        tr_keys, val_keys = self.do_split()
        
        labeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'labeled.txt')
        unlabeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'unlabeled.txt')
        
        if not os.path.exists(labeled_txt_path):
            raise FileNotFoundError(f"找不到 labeled.txt: {labeled_txt_path}")
        if not os.path.exists(unlabeled_txt_path):
            raise FileNotFoundError(f"找不到 unlabeled.txt: {unlabeled_txt_path}")
        
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
        if allowed_num_processes == 0:
            mt_gen_train_labeled = SingleThreadedAugmenter(dl_tr_labeled, labeled_transforms)
            mt_gen_train_unlabeled = SingleThreadedAugmenter(dl_tr_unlabeled, _identity_transform)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            num_proc = max(1, allowed_num_processes // 2)
            mt_gen_train_labeled = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr_labeled,
                transform=labeled_transforms,
                num_processes=num_proc,
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == 'cuda'
            )
            mt_gen_train_unlabeled = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr_unlabeled,
                transform=_identity_transform,
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
        
        self.logger.my_fantastic_logging['loss_sup_1'] = list()
        self.logger.my_fantastic_logging['loss_sup_2'] = list()
        self.logger.my_fantastic_logging['loss_cps'] = list()
        
        print(f"📊 Labeled dataloader iterations: {len(self.dataloader_train)}")
        print(f"📊 Unlabeled dataloader iterations: {len(self.dataloader_unlabeled)}")
    
    def get_current_cps_weight(self) -> float:
        """
        获取当前 CPS 权重 (rampup 策略)
        
        在 rampup 阶段逐渐增加 CPS 权重
        """
        if self.current_epoch < self.rampup_epochs:
            return self.cps_weight * (self.current_epoch + 1) / self.rampup_epochs
        return self.cps_weight
    
    def train_step(self, batch: dict, accumulation_step: int = 0) -> dict:
        """
        CPS 训练步骤 (支持梯度累加)
        
        1. 获取 labeled 和 unlabeled batch
        2. branch1 和 branch2 分别前向
        3. 计算监督损失 (labeled)
        4. 计算 CPS 损失 (unlabeled)
        5. 总损失反向传播
        """
        labeled_batch = batch['labeled']
        unlabeled_batch = batch['unlabeled']
        
        data_labeled = labeled_batch['data']
        target_labeled = labeled_batch['target']
        
        data_labeled = data_labeled.to(self.device, non_blocking=True)
        if isinstance(target_labeled, list):
            target_labeled = [t.to(self.device, non_blocking=True) for t in target_labeled]
        else:
            target_labeled = target_labeled.to(self.device, non_blocking=True)
        
        data_unlabeled = unlabeled_batch['data']
        
        if isinstance(data_unlabeled, np.ndarray):
            data_unlabeled = torch.from_numpy(data_unlabeled)
        data_unlabeled = data_unlabeled.to(self.device, non_blocking=True)
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            output1_labeled = self.network(data_labeled, branch='branch1')
            output2_labeled = self.network(data_labeled, branch='branch2')
            
            output1_unlabeled = self.network(data_unlabeled, branch='branch1')
            output2_unlabeled = self.network(data_unlabeled, branch='branch2')
            
            if self.enable_aux_head and isinstance(output1_labeled, tuple):
                main_output1_labeled, aux_output1_labeled = output1_labeled
                main_output2_labeled, aux_output2_labeled = output2_labeled
            else:
                main_output1_labeled = output1_labeled
                main_output2_labeled = output2_labeled
                aux_output1_labeled = None
                aux_output2_labeled = None
            
            loss_sup_1 = self.loss(main_output1_labeled, target_labeled)
            loss_sup_2 = self.loss(main_output2_labeled, target_labeled)
            
            if aux_output1_labeled is not None and self.aux_loss_weight > 0:
                if isinstance(target_labeled, list):
                    aux_target = target_labeled[0]
                else:
                    aux_target = target_labeled
                if aux_target.ndim == 4 and aux_target.shape[1] == 1:
                    aux_target = aux_target[:, 0]
                aux_target = aux_target.long()
                
                aux_loss_1 = self.aux_ce_loss(aux_output1_labeled, aux_target)
                aux_loss_2 = self.aux_ce_loss(aux_output2_labeled, aux_target)
                loss_sup_1 = loss_sup_1 + self.aux_loss_weight * aux_loss_1
                loss_sup_2 = loss_sup_2 + self.aux_loss_weight * aux_loss_2
            
            pred1_unlabeled = extract_main_logits(output1_unlabeled)
            pred2_unlabeled = extract_main_logits(output2_unlabeled)
            
            cps_loss = compute_cps_loss(pred1_unlabeled, pred2_unlabeled, self.cps_loss_fn)
            
            current_cps_weight = self.get_current_cps_weight()
            total_loss = loss_sup_1 + loss_sup_2 + current_cps_weight * cps_loss
            
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
        
        log_dict = {
            'loss_sup_1': loss_sup_1.detach().cpu().numpy(),
            'loss_sup_2': loss_sup_2.detach().cpu().numpy(),
            'loss_cps': cps_loss.detach().cpu().numpy(),
            'cps_weight': current_cps_weight,
            'total_loss': total_loss.detach().cpu().numpy() * self.gradient_accumulation_steps
        }
        
        return log_dict
    
    def on_epoch_start(self):
        """
        Epoch 开始时的操作
        """
        super().on_epoch_start()
        
        print(f"📊 Current CPS weight: {self.get_current_cps_weight():.4f}")
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        处理训练 epoch 结束时的输出
        """
        from nnunetv2.utilities.collate_outputs import collate_outputs
        
        outputs = collate_outputs(train_outputs)
        
        loss_sup_1 = np.mean(outputs['loss_sup_1'])
        loss_sup_2 = np.mean(outputs['loss_sup_2'])
        loss_cps = np.mean(outputs['loss_cps'])
        total_loss = np.mean(outputs['total_loss'])
        
        self.logger.log('train_losses', total_loss, self.current_epoch)
        self.logger.log('loss_sup_1', loss_sup_1, self.current_epoch)
        self.logger.log('loss_sup_2', loss_sup_2, self.current_epoch)
        self.logger.log('loss_cps', loss_cps, self.current_epoch)
        
        print(f"Epoch {self.current_epoch}: total_loss={total_loss:.4f}, "
              f"sup1={loss_sup_1:.4f}, sup2={loss_sup_2:.4f}, cps={loss_cps:.4f}")
    
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
            cps_state = {
                'branch1_state_dict': self.network.branch1.state_dict(),
                'branch2_state_dict': self.network.branch2.state_dict(),
                'epoch': self.current_epoch,
                'cps_weight': self.cps_weight
            }
            torch.save(cps_state, checkpoint_path.replace('.pth', '_cps.pth'))
    
    def load_checkpoint(self, filename: str) -> None:
        """
        加载检查点
        """
        super().load_checkpoint(filename)
        
        checkpoint_path = os.path.join(self.output_folder, filename)
        cps_checkpoint_path = checkpoint_path.replace('.pth', '_cps.pth')
        
        if os.path.exists(cps_checkpoint_path):
            cps_state = torch.load(cps_checkpoint_path, map_location=self.device)
            if hasattr(self.network, 'branch1') and hasattr(self.network, 'branch2'):
                self.network.branch1.load_state_dict(cps_state['branch1_state_dict'])
                self.network.branch2.load_state_dict(cps_state['branch2_state_dict'])
                print(f"✅ Loaded CPS checkpoint from {cps_checkpoint_path}")
