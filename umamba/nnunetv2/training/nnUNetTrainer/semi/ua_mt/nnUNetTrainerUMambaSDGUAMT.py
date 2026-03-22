"""
UA-MT (Uncertainty-Aware Mean Teacher) 半监督训练器

基于 UA-MT 论文实现的半监督医学图像分割训练器

核心机制:
1. EMA Teacher 更新
2. Consistency weight ramp-up
3. Teacher 多次随机扰动采样
4. 基于熵的不确定性估计
5. Uncertainty mask 筛选一致性损失

骨干网络: UMambaSDG (通过 enable_sdg=True 启用)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np
import copy
from typing import Optional, Dict

try:
    from ....nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG
except ModuleNotFoundError:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG

try:
    from .ua_mt_modules import (
        update_ema_variables,
        sigmoid_rampup,
        get_current_consistency_weight,
        compute_entropy_uncertainty,
        normalize_entropy,
        generate_uncertainty_mask,
        softmax_mse_loss,
        EMAModel,
        UncertaintyEstimator
    )
except ImportError:
    from nnunetv2.training.nnUNetTrainer.semi.ua_mt.ua_mt_modules import (
        update_ema_variables,
        sigmoid_rampup,
        get_current_consistency_weight,
        compute_entropy_uncertainty,
        normalize_entropy,
        generate_uncertainty_mask,
        softmax_mse_loss,
        EMAModel,
        UncertaintyEstimator
    )

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


def _identity_transform(**data_dict):
    return data_dict


class nnUNetTrainerUMambaSDGUAMT(nnUNetTrainerUMambaSDG):
    """
    UA-MT (Uncertainty-Aware Mean Teacher) 半监督训练器
    
    继承自 nnUNetTrainerUMambaSDG，确保使用 UMambaSDG 骨干网络
    
    核心特性:
    1. Student 网络使用 UMambaSDG (enable_sdg=True)
    2. Teacher 网络是 Student 的 EMA 副本，同样使用 UMambaSDG 结构
    3. 有标注样本计算监督损失
    4. 无标注样本计算 uncertainty-masked consistency loss
    5. Teacher 通过多次随机扰动前向得到平均预测与不确定性估计
    """
    
    DEFAULT_EMA_DECAY: float = 0.99
    DEFAULT_CONSISTENCY: float = 0.1
    DEFAULT_CONSISTENCY_WARMUP: int = 15
    DEFAULT_CONSISTENCY_RAMPUP: int = 40
    DEFAULT_LABELED_BS: int = 6
    DEFAULT_TEACHER_SAMPLES: int = 8
    DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.6
    DEFAULT_DROPOUT_RATE: float = 0.1
    
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
        
        self.batch_size = 12
        self.initial_lr = 0.005
        
        self.unlabeled_batch_iter = None
        
        self.ema_decay = self.DEFAULT_EMA_DECAY
        self.consistency = self.DEFAULT_CONSISTENCY
        self.consistency_warmup = self.DEFAULT_CONSISTENCY_WARMUP
        self.consistency_rampup = self.DEFAULT_CONSISTENCY_RAMPUP
        self.labeled_bs = self.DEFAULT_LABELED_BS
        self.teacher_samples = self.DEFAULT_TEACHER_SAMPLES
        self.uncertainty_threshold = self.DEFAULT_UNCERTAINTY_THRESHOLD
        self.dropout_rate = self.DEFAULT_DROPOUT_RATE
        
        self.ema_model = None
        self.uncertainty_estimator = None
        
        self.global_step = 0
        
        print("=" * 60)
        print("🔥🔥🔥 UA-MT 半监督训练器已加载 🔥🔥🔥")
        print(f"   骨干网络: UMambaSDG (enable_sdg=True)")
        print(f"   initial_lr: {self.initial_lr}")
        print(f"   batch_size: {self.batch_size}")
        print(f"   labeled_bs: {self.labeled_bs}")
        print(f"   EMA decay: {self.ema_decay}")
        print(f"   Consistency weight: {self.consistency}")
        print(f"   Consistency warmup: {self.consistency_warmup} epochs")
        print(f"   Consistency rampup: {self.consistency_rampup} epochs")
        print(f"   Teacher samples (T): {self.teacher_samples}")
        print(f"   Uncertainty threshold: {self.uncertainty_threshold}")
        print(f"   Dropout rate: {self.dropout_rate}")
        print("=" * 60)
    
    def _init_ema_model(self):
        if self.ema_model is not None:
            return
        
        self.ema_model = EMAModel(
            model=self.network,
            decay=self.ema_decay
        )
        
        self.uncertainty_estimator = UncertaintyEstimator(
            num_samples=self.teacher_samples,
            uncertainty_threshold=self.uncertainty_threshold
        )
        
        print(f"🔧 EMA Teacher model initialized with decay={self.ema_decay}")
        print(f"🔧 UncertaintyEstimator initialized with T={self.teacher_samples}, threshold={self.uncertainty_threshold}")
    
    def get_dataloaders(self):
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

        dataset_tr_labeled = nnUNetDataset(self.preprocessed_dataset_folder, tr_labeled_keys,
                                           folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                           num_images_properties_loading_threshold=0)
        dataset_tr_unlabeled = nnUNetDataset(self.preprocessed_dataset_folder, tr_unlabeled_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                             num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)

        labeled_bs = min(self.labeled_bs, self.batch_size)
        unlabeled_bs = self.batch_size - labeled_bs
        
        dl_cls = nnUNetDataLoader2D if dim == 2 else nnUNetDataLoader3D
        
        dl_tr_labeled = dl_cls(dataset_tr_labeled, labeled_bs, initial_patch_size, 
                               self.configuration_manager.patch_size, self.label_manager, 
                               oversample_foreground_percent=self.oversample_foreground_percent)
        
        dl_tr_unlabeled = dl_cls(dataset_tr_unlabeled, unlabeled_bs, initial_patch_size, 
                                 self.configuration_manager.patch_size, self.label_manager, 
                                 oversample_foreground_percent=0.0)
        
        dl_val = dl_cls(dataset_val, self.batch_size, self.configuration_manager.patch_size, 
                        self.configuration_manager.patch_size, self.label_manager, 
                        oversample_foreground_percent=self.oversample_foreground_percent)

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
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == 'cuda'
            )

        self.dataloader_unlabeled = mt_gen_train_unlabeled
        self.dataset_tr_unlabeled = dataset_tr_unlabeled
        self.dl_tr_unlabeled = dl_tr_unlabeled
        
        return mt_gen_train_labeled, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        self._init_ema_model()
        
        print(f"✅ 网络类型确认: {type(self.network).__name__}")
        if hasattr(self.network, 'enable_sdg'):
            print(f"✅ enable_sdg = {self.network.enable_sdg}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.unlabeled_batch_iter = iter(self.dataloader_unlabeled)

    def _get_teacher_prediction_with_uncertainty(
        self, 
        teacher_model: nn.Module, 
        input_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher 多次随机扰动前向传播，计算平均预测和不确定性
        
        实现方式:
        - 在 eval 模式下强制启用 dropout 来制造随机性
        - BatchNorm/InstanceNorm 保持 eval 模式
        
        Args:
            teacher_model: EMA teacher 模型
            input_tensor: 无标签输入 [B, C, H, W]
        
        Returns:
            Dict containing:
                - mean_pred: 平均预测概率 [B, C, H, W]
                - uncertainty: 不确定性图 [B, 1, H, W]
                - uncertainty_mask: 确定性 mask [B, 1, H, W]
                - teacher_logits: 平均 logits [B, C, H, W]
        """
        teacher_model.train()
        
        for m in teacher_model.modules():
            if isinstance(m, nn.Dropout):
                m.p = self.dropout_rate
                m.train()
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.SyncBatchNorm)):
                m.eval()
        
        preds = []
        logits_list = []
        
        with torch.no_grad():
            for _ in range(self.teacher_samples):
                output = teacher_model(input_tensor)
                
                if self.enable_aux_head and isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                logits_list.append(output)
                pred = F.softmax(output, dim=1)
                preds.append(pred)
        
        teacher_model.eval()
        
        preds = torch.stack(preds, dim=0)
        logits_stack = torch.stack(logits_list, dim=0)
        
        mean_pred = preds.mean(dim=0)
        mean_logits = logits_stack.mean(dim=0)
        
        num_classes = mean_pred.shape[1]
        entropy = compute_entropy_uncertainty(mean_pred, dim=1)
        normalized_entropy = normalize_entropy(entropy, num_classes)
        
        uncertainty_mask = (normalized_entropy < self.uncertainty_threshold).float()
        
        return {
            'mean_pred': mean_pred,
            'uncertainty': normalized_entropy,
            'uncertainty_mask': uncertainty_mask,
            'teacher_logits': mean_logits
        }

    def _compute_consistency_loss(
        self,
        student_output: torch.Tensor,
        teacher_pred: torch.Tensor,
        uncertainty_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 uncertainty-masked consistency loss
        
        使用 softmax MSE loss，仅在低不确定性区域计算
        
        Args:
            student_output: student logits [B, C, H, W]
            teacher_pred: teacher 平均预测概率 [B, C, H, W]
            uncertainty_mask: 确定性 mask [B, 1, H, W]
        
        Returns:
            consistency_loss: 标量损失
        """
        student_softmax = F.softmax(student_output, dim=1)
        
        loss = (student_softmax - teacher_pred) ** 2
        
        loss = loss * uncertainty_mask
        
        mask_sum = uncertainty_mask.sum() + 1e-6
        return loss.sum() / mask_sum

    def train_step(self, batch: dict) -> dict:
        """
        UA-MT 训练步骤
        
        流程:
        1. 从 batch 获取 labeled 和 unlabeled 数据
        2. Student 对 labeled 数据前向，计算监督损失
        3. Teacher 对 unlabeled 数据执行多次扰动前向，得到平均预测和不确定性
        4. Student 对 unlabeled 数据前向
        5. 计算 uncertainty-masked consistency loss
        6. 总损失 = supervised_loss + consistency_weight * consistency_loss
        7. 反向传播、优化器更新、EMA 更新
        """
        data_labeled = batch['data']
        target_labeled = batch['target']
        
        try:
            batch_unlabeled = next(self.unlabeled_batch_iter)
        except StopIteration:
            self.unlabeled_batch_iter = iter(self.dataloader_unlabeled)
            batch_unlabeled = next(self.unlabeled_batch_iter)
        
        data_unlabeled = batch_unlabeled['data']

        data_labeled = data_labeled.to(self.device, non_blocking=True)
        if isinstance(target_labeled, list):
            target_labeled = [i.to(self.device, non_blocking=True) for i in target_labeled]
        else:
            target_labeled = target_labeled.to(self.device, non_blocking=True)

        if isinstance(data_unlabeled, np.ndarray):
            data_unlabeled = torch.from_numpy(data_unlabeled)
        data_unlabeled = data_unlabeled.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        consistency_weight = get_current_consistency_weight(
            epoch=self.current_epoch,
            consistency=self.consistency,
            rampup_length=self.consistency_rampup,
            warmup_epochs=self.consistency_warmup
        )
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            student_output_labeled = self.network(data_labeled)
            
            if self.enable_aux_head and isinstance(student_output_labeled, tuple):
                main_output_labeled, aux_output_labeled = student_output_labeled
            else:
                main_output_labeled = student_output_labeled
                aux_output_labeled = None
            
            supervised_loss = self.loss(main_output_labeled, target_labeled)
            
            if aux_output_labeled is not None:
                aux_target = target_labeled[0] if isinstance(target_labeled, list) else target_labeled
                if aux_target.ndim == 4 and aux_target.shape[1] == 1:
                    aux_target = aux_target[:, 0].long()
                else:
                    aux_target = aux_target.long()
                aux_loss = self.aux_ce_loss(aux_output_labeled, aux_target).mean()
                supervised_loss = supervised_loss + self.aux_loss_weight * aux_loss
            
            student_main_labeled = main_output_labeled
            if isinstance(student_main_labeled, (list, tuple)):
                student_main_labeled = student_main_labeled[0]
            
            if data_unlabeled.shape[0] > 0 and consistency_weight > 0:
                teacher_model = self.ema_model.model
                
                teacher_results = self._get_teacher_prediction_with_uncertainty(
                    teacher_model, data_unlabeled
                )
                
                teacher_pred = teacher_results['mean_pred']
                uncertainty_mask = teacher_results['uncertainty_mask']
                uncertainty = teacher_results['uncertainty']
                
                student_output_unlabeled = self.network(data_unlabeled)
                
                if self.enable_aux_head and isinstance(student_output_unlabeled, tuple):
                    main_output_unlabeled, _ = student_output_unlabeled
                else:
                    main_output_unlabeled = student_output_unlabeled
                
                student_main_unlabeled = main_output_unlabeled
                if isinstance(student_main_unlabeled, (list, tuple)):
                    student_main_unlabeled = student_main_unlabeled[0]
                
                consistency_loss = self._compute_consistency_loss(
                    student_main_unlabeled, teacher_pred, uncertainty_mask
                )
                
                total_loss = supervised_loss + consistency_weight * consistency_loss
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)
                uncertainty_mask = torch.zeros(1, device=self.device)
                uncertainty = torch.zeros(1, device=self.device)
                total_loss = supervised_loss
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        self.global_step += 1
        self.ema_model.update(self.network, self.global_step)
        
        mask_coverage = uncertainty_mask.mean().item() if uncertainty_mask.numel() > 1 else 0.0
        
        log_dict = {
            'loss': total_loss.detach().cpu().numpy(),
            'supervised_loss': supervised_loss.detach().cpu().numpy(),
            'consistency_loss': consistency_loss.detach().cpu().numpy() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'consistency_weight': consistency_weight,
            'uncertainty_mask_coverage': mask_coverage
        }
        
        return log_dict

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
                output = self.network(data)
                
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
        super().save_checkpoint(filename)
        
        checkpoint_path = os.path.join(self.output_folder, filename)
        
        if self.ema_model is not None:
            ema_state = {
                'ema_model_state_dict': self.ema_model.state_dict(),
                'global_step': self.global_step
            }
            torch.save(ema_state, checkpoint_path.replace('.pth', '_ema.pth'))

    def load_checkpoint(self, filename: str) -> None:
        super().load_checkpoint(filename)
        
        checkpoint_path = os.path.join(self.output_folder, filename)
        ema_checkpoint_path = checkpoint_path.replace('.pth', '_ema.pth')
        
        if os.path.exists(ema_checkpoint_path):
            ema_state = torch.load(ema_checkpoint_path, map_location=self.device)
            
            if self.ema_model is not None:
                self.ema_model.load_state_dict(ema_state['ema_model_state_dict'])
            
            self.global_step = ema_state.get('global_step', 0)
            print(f"✅ Loaded EMA model from {ema_checkpoint_path}")
