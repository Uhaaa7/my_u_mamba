"""
Mean Teacher 半监督训练器

基于 EMA Mean Teacher 框架的半监督医学图像分割训练器

核心创新点:
1. 伪标签提纯模块: 基于主输出、辅助输出和边界信息协同评估
2. Skip 特征蒸馏: 可靠性引导的增强 skip 特征蒸馏
3. 边界一致性: 专门处理边界区域的约束
4. Prototype-based Semi-supervised Learning:
   - PGPC: Prototype-Guided Pixel Classification
   - APC: Boundary-aware Adaptive Prototype Contrast
   - PAC: Prototype Assignment Consistency

修复:
1. weak/strong augmentation 分离 (同一病例配对)
2. 区域分级监督 (高可靠 hard 学、边界过渡区 soft 学、低可靠区域少学)
3. 保留 nnUNet 原有的 Dice 损失和 deep supervision
4. 验证阶段的 shape 问题
5. aux_loss 标量化
6. ce_loss 和 mask 维度对齐
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from torch.nn.functional import interpolate

try:
    from ....nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG
except ModuleNotFoundError:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG

try:
    from .semi_supervised_modules import (
        SemiSupervisedLoss, 
        EMAModel, 
        get_current_consistency_weight,
        BoundaryHead,
        PseudoLabelRefiner
    )
    from .mean_teacher_wrapper import MeanTeacherWrapper, InferenceOnlyWrapper
    from .prototype_modules import PrototypeBank, PrototypeLoss
except ImportError:
    from nnunetv2.training.nnUNetTrainer.semi.mean_teacher.semi_supervised_modules import (
        SemiSupervisedLoss, 
        EMAModel, 
        get_current_consistency_weight,
        BoundaryHead,
        PseudoLabelRefiner
    )
    from nnunetv2.training.nnUNetTrainer.semi.mean_teacher.mean_teacher_wrapper import (
        MeanTeacherWrapper, 
        InferenceOnlyWrapper
    )
    from nnunetv2.training.nnUNetTrainer.semi.mean_teacher.prototype_modules import (
        PrototypeBank, 
        PrototypeLoss
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

class nnUNetTrainerUMambaSDG_MeanTeacher(nnUNetTrainerUMambaSDG):
    """
    Mean Teacher 半监督训练器
    
    继承自 nnUNetTrainerUMambaSDG，保持第一创新点 (SDG-Block) 不变
    
    新增功能:
    1. EMA Teacher 模型
    2. 边界分支
    3. Skip 特征蒸馏
    4. 伪标签提纯
    5. 边界一致性
    6. Weak/Strong augmentation 分离 (同一病例配对)
    7. Prototype-based Semi-supervised Learning
    """
    
    DEFAULT_EMA_DECAY: float = 0.999
    DEFAULT_EMA_WARMUP_STEPS: int = 2000
    DEFAULT_CONSISTENCY_WARMUP: int = 15
    DEFAULT_CONSISTENCY_RAMPUP: int = 40
    DEFAULT_MAX_CONSISTENCY_WEIGHT: float = 0.1
    
    DEFAULT_SKIP_DISTILL_WEIGHT: float = 0.5
    DEFAULT_BOUNDARY_CONSISTENCY_WEIGHT: float = 0.3
    DEFAULT_PSEUDO_LABEL_WEIGHT: float = 1.0
    DEFAULT_BOUNDARY_WEIGHT: float = 0.4
    DEFAULT_AUX_WEIGHT: float = 0.4
    
    DEFAULT_ENABLE_PROTOTYPE: bool = True
    DEFAULT_PROJECTION_DIM: int = 128
    DEFAULT_NUM_PROTOTYPES_PER_CLASS: int = 5
    DEFAULT_PROTOTYPE_MOMENTUM: float = 0.9
    DEFAULT_PGPC_WEIGHT: float = 1.0
    DEFAULT_APC_WEIGHT: float = 0.5
    DEFAULT_PAC_WEIGHT: float = 0.3
    
    DEFAULT_INCONSISTENCY_PENALTY: float = 0.4
    DEFAULT_BOUNDARY_DECAY_FACTOR: float = 0.3
    DEFAULT_HIGH_RELIABILITY_THRESHOLD: float = 0.75
    DEFAULT_LOW_RELIABILITY_THRESHOLD: float = 0.45
    
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
        self.weak_transforms = None
        self.strong_transforms = None
        
        self.ema_decay = self.DEFAULT_EMA_DECAY
        self.ema_warmup_steps = self.DEFAULT_EMA_WARMUP_STEPS
        self.consistency_warmup = self.DEFAULT_CONSISTENCY_WARMUP
        self.consistency_rampup = self.DEFAULT_CONSISTENCY_RAMPUP
        self.max_consistency_weight = self.DEFAULT_MAX_CONSISTENCY_WEIGHT
        
        self.skip_distill_weight = self.DEFAULT_SKIP_DISTILL_WEIGHT
        self.boundary_consistency_weight = self.DEFAULT_BOUNDARY_CONSISTENCY_WEIGHT
        self.pseudo_label_weight = self.DEFAULT_PSEUDO_LABEL_WEIGHT
        self.boundary_weight = self.DEFAULT_BOUNDARY_WEIGHT
        self.aux_weight = self.DEFAULT_AUX_WEIGHT
        
        self.enable_prototype = self.DEFAULT_ENABLE_PROTOTYPE
        self.projection_dim = self.DEFAULT_PROJECTION_DIM
        self.num_prototypes_per_class = self.DEFAULT_NUM_PROTOTYPES_PER_CLASS
        self.prototype_momentum = self.DEFAULT_PROTOTYPE_MOMENTUM
        self.pGPC_weight = self.DEFAULT_PGPC_WEIGHT
        self.aPC_weight = self.DEFAULT_APC_WEIGHT
        self.pAC_weight = self.DEFAULT_PAC_WEIGHT
        
        self.inconsistency_penalty = self.DEFAULT_INCONSISTENCY_PENALTY
        self.boundary_decay_factor = self.DEFAULT_BOUNDARY_DECAY_FACTOR
        self.high_reliability_threshold = self.DEFAULT_HIGH_RELIABILITY_THRESHOLD
        self.low_reliability_threshold = self.DEFAULT_LOW_RELIABILITY_THRESHOLD
        
        self.prototype_bank = None
        self.prototype_loss = None
        self.prototype_initialized = False
        
        print("=" * 60)
        print("🔥🔥🔥 Mean Teacher 半监督训练器已加载 🔥🔥🔥")
        print(f"   initial_lr: {self.initial_lr}")
        print(f"   batch_size: {self.batch_size}")
        print(f"   EMA decay: {self.ema_decay}")
        print(f"   Consistency warmup: {self.consistency_warmup} epochs")
        print(f"   Consistency rampup: {self.consistency_rampup} epochs")
        print(f"   Skip distill weight: {self.skip_distill_weight}")
        print(f"   Boundary consistency weight: {self.boundary_consistency_weight}")
        print(f"   Inconsistency penalty: {self.inconsistency_penalty}")
        print(f"   Boundary decay factor: {self.boundary_decay_factor}")
        print(f"   High reliability threshold: {self.high_reliability_threshold}")
        print(f"   Low reliability threshold: {self.low_reliability_threshold}")
        print(f"   Enable prototype: {self.enable_prototype}")
        if self.enable_prototype:
            print(f"   Projection dim: {self.projection_dim}")
            print(f"   Num prototypes per class: {self.num_prototypes_per_class}")
            print(f"   Prototype momentum: {self.prototype_momentum}")
            print(f"   PGPC weight: {self.pGPC_weight}")
            print(f"   APC weight: {self.aPC_weight}")
            print(f"   PAC weight: {self.pAC_weight}")
        print("=" * 60)
    
    @staticmethod
    def build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision: bool = True,
        **kwargs
    ) -> nn.Module:
        from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
        
        skip_modes = nnUNetTrainerUMambaSDG.DEFAULT_SKIP_MODES
        enable_aux_head = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_AUX_HEAD
        aux_head_stage = nnUNetTrainerUMambaSDG.DEFAULT_AUX_HEAD_STAGE
        enable_adaptive_upsample = nnUNetTrainerUMambaSDG.DEFAULT_ENABLE_ADAPTIVE_UPSAMPLE
        
        enable_prototype = kwargs.get('enable_prototype', nnUNetTrainerUMambaSDG_MeanTeacher.DEFAULT_ENABLE_PROTOTYPE)
        projection_dim = kwargs.get('projection_dim', nnUNetTrainerUMambaSDG_MeanTeacher.DEFAULT_PROJECTION_DIM)
        
        if len(configuration_manager.patch_size) == 2:
            base_model = get_umamba_bot_2d_from_plans(
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
        else:
            raise NotImplementedError("Only 2D models are supported for Mean Teacher currently")
        
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        
        wrapped_model = MeanTeacherWrapper(
            base_model=base_model,
            num_classes=num_classes,
            boundary_stage=-1,
            enable_prototype=enable_prototype,
            projection_dim=projection_dim
        )
        
        print("🚀🚀🚀 [Mean Teacher Mode] UMambaBot with SDG-Block + Boundary Branch + Prototype! 🚀🚀🚀")
        
        return wrapped_model
    
    def _init_ema_model(self):
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            return
        
        self.ema_model = EMAModel(
            model=self.network,
            decay=self.ema_decay,
            warmup_steps=self.ema_warmup_steps
        )
        
        print(f"🔧 EMA Teacher model initialized with decay={self.ema_decay}")
    
    def _init_loss_functions(self):
        num_classes = self.label_manager.num_segmentation_heads
        
        self.semi_supervised_loss = SemiSupervisedLoss(
            num_classes=num_classes,
            skip_distill_weight=self.skip_distill_weight,
            boundary_consistency_weight=self.boundary_consistency_weight,
            pseudo_label_weight=self.pseudo_label_weight,
            boundary_weight=self.boundary_weight
        )
        
        self.pseudo_label_refiner = PseudoLabelRefiner(
            num_classes=num_classes,
            inconsistency_penalty=self.inconsistency_penalty,
            boundary_decay_factor=self.boundary_decay_factor,
            high_reliability_threshold=self.high_reliability_threshold,
            low_reliability_threshold=self.low_reliability_threshold
        )
        
        self.boundary_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        self.aux_ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        if self.enable_prototype:
            num_stages = len(self.network.skip_modes) if hasattr(self.network, 'skip_modes') else 3
            
            self.prototype_bank = PrototypeBank(
                num_stages=num_stages,
                num_classes=num_classes,
                num_prototypes_per_class=self.num_prototypes_per_class,
                projection_dim=self.projection_dim,
                momentum=self.prototype_momentum,
                device=self.device
            )
            
            self.prototype_loss = PrototypeLoss(
                prototype_bank=self.prototype_bank,
                num_classes=num_classes,
                temperature=0.1
            )
            
            print(f"🔧 Prototype bank initialized with {num_stages} stages, {num_classes} classes")
    
    def get_dataloaders(self):
        """
        重写数据加载逻辑
        
        关键改进：无标签数据使用单个 dataloader（不经过 transform），
        在 train_step 中分别应用 weak/strong transform
        这样确保 teacher 和 student 看到同一张无标签图像的不同增强版本
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
            ignore_label=self.label_manager.ignore_label)

        # 无标签数据不在 dataloader 里做训练增强，直接返回原始 patch。
        # weak / strong 视图在 train_step 中从同一个 raw batch 构造。
        self.strong_transforms = None
        self.weak_transforms = None


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
            raise FileNotFoundError(f"找不到半监督列表 {labeled_txt_path} 或 {unlabeled_txt_path}。")
            
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

        half_batch_size = max(1, self.batch_size // 2)
        dl_cls = nnUNetDataLoader2D if dim == 2 else nnUNetDataLoader3D
        
        dl_tr_labeled = dl_cls(dataset_tr_labeled, half_batch_size, initial_patch_size, 
                               self.configuration_manager.patch_size, self.label_manager, 
                               oversample_foreground_percent=self.oversample_foreground_percent)
        
        dl_tr_unlabeled = dl_cls(dataset_tr_unlabeled, half_batch_size, initial_patch_size, 
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
        self._init_loss_functions()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.unlabeled_batch_iter = iter(self.dataloader_unlabeled)

    def _get_boundary_target(self, target: torch.Tensor, num_classes: int = None) -> torch.Tensor:
        if isinstance(target, (list, tuple)):
            target = target[0]
        
        if num_classes is None:
            num_classes = self.label_manager.num_segmentation_heads
        
        return BoundaryHead.extract_boundary_from_mask(
            target, 
            num_classes=num_classes,
            kernel_size=3,
            ignore_background=True
        )

    def _compute_labeled_loss(self,
                              outputs: Dict[str, torch.Tensor],
                              target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        
        main_outputs = self.network.get_all_main_outputs(outputs)
        
        if isinstance(target, (list, tuple)):
            target_list = target
        else:
            target_list = [target]
        
        main_loss = self.loss(main_outputs, target_list)
        losses['main_loss'] = main_loss
        
        aux_output = self.network.get_aux_output_tensor(outputs)
        if aux_output is not None:
            target_for_aux = target_list[0]
            if target_for_aux.ndim == 4 and target_for_aux.shape[1] == 1:
                target_for_aux = target_for_aux[:, 0]
            target_for_aux = target_for_aux.long()
            aux_loss = self.aux_ce_loss(aux_output, target_for_aux).mean()
            losses['aux_loss'] = aux_loss * self.aux_weight

        
        boundary_output = self.network.get_boundary_output_tensor(outputs)
        boundary_target = None
        if boundary_output is not None:
            boundary_target = self._get_boundary_target(target)
            boundary_loss = self.boundary_loss(boundary_output, boundary_target).mean()
            losses['boundary_loss'] = boundary_loss * self.boundary_weight
        
        if self.enable_prototype and self.prototype_bank is not None:
            projections = self.network.get_projections(outputs)
            if projections is not None and boundary_target is not None:
                if not self.prototype_initialized:
                    target_for_proto = target_list[0]
                    if target_for_proto.ndim == 4 and target_for_proto.shape[1] == 1:
                        target_for_proto = target_for_proto[:, 0]
                    
                    self.prototype_bank.initialize_from_labeled_data(
                        projections, target_for_proto, boundary_target
                    )
                    self.prototype_initialized = True
                
                target_for_proto = target_list[0]
                if target_for_proto.ndim == 4 and target_for_proto.shape[1] == 1:
                    target_for_proto = target_for_proto[:, 0]
                
                reliability_map = torch.ones_like(boundary_target)
                proto_losses = self.prototype_loss(
                    student_projections=projections,
                    teacher_projections=projections,
                    pseudo_labels=target_for_proto,
                    reliability_map=reliability_map,
                    boundary_map=boundary_target,
                    high_reliability_threshold=self.high_reliability_threshold,
                    low_reliability_threshold=self.low_reliability_threshold,
                    pgpc_weight=self.pGPC_weight,
                    apc_weight=self.aPC_weight,
                    pac_weight=0.0
                )
                
                losses['pgpc_loss'] = proto_losses['pgpc_loss']
                losses['apc_loss'] = proto_losses['apc_loss']
        
        losses['total'] = sum(losses.values())
        return losses

    def _compute_unlabeled_loss(self,
                                student_outputs: Dict[str, torch.Tensor],
                                teacher_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        teacher_main = self.network.get_main_output_tensor(teacher_outputs)
        teacher_aux = self.network.get_aux_output_tensor(teacher_outputs)
        teacher_boundary = self.network.get_boundary_output_tensor(teacher_outputs)
        
        with torch.no_grad():
            pseudo_label, reliability_map = self.pseudo_label_refiner(
                teacher_main, teacher_aux, teacher_boundary
            )
        
        high_reliable_mask = self.pseudo_label_refiner.get_reliability_mask(reliability_map, 'high')
        medium_reliable_mask = self.pseudo_label_refiner.get_reliability_mask(reliability_map, 'medium')
        low_reliable_mask = self.pseudo_label_refiner.get_reliability_mask(reliability_map, 'low')
        
        student_main = self.network.get_main_output_tensor(student_outputs)
        
        ce_loss = F.cross_entropy(student_main, pseudo_label, reduction='none')
        ce_loss = ce_loss.unsqueeze(1)
        
        high_loss = (ce_loss * high_reliable_mask.float()).sum() / (high_reliable_mask.float().sum() + 1e-6)
        
        teacher_prob = F.softmax(teacher_main, dim=1)
        kl_loss = F.kl_div(
            F.log_softmax(student_main, dim=1),
            teacher_prob,
            reduction='none'
        ).sum(dim=1, keepdim=True)
        medium_loss = (kl_loss * medium_reliable_mask.float()).sum() / (medium_reliable_mask.float().sum() + 1e-6)
        
        low_loss = (ce_loss * low_reliable_mask.float() * 0.1).sum() / (low_reliable_mask.float().sum() + 1e-6)
        
        pseudo_label_loss = high_loss + medium_loss * 0.5 + low_loss
        losses['pseudo_label_loss'] = pseudo_label_loss * self.pseudo_label_weight
        
        student_skip = self.network.get_skip_features(student_outputs)
        teacher_skip = self.network.get_skip_features(teacher_outputs)
        
        if student_skip and teacher_skip:
            skip_loss = self.semi_supervised_loss.skip_distill_loss(
                student_skip, teacher_skip, reliability_map
            )
            losses['skip_distill_loss'] = skip_loss * self.skip_distill_weight
        
        student_boundary = self.network.get_boundary_output_tensor(student_outputs)
        if student_boundary is not None and teacher_boundary is not None:
            boundary_loss = self.semi_supervised_loss.boundary_consistency_loss(
                student_boundary, teacher_boundary, reliability_map
            )
            losses['boundary_consistency_loss'] = boundary_loss * self.boundary_consistency_weight
        
        if self.enable_prototype and self.prototype_bank is not None and self.prototype_initialized:
            teacher_projections = self.network.get_projections(teacher_outputs)
            student_projections = self.network.get_projections(student_outputs)
            
            if teacher_projections is not None and student_projections is not None:
                teacher_boundary_for_proto = torch.sigmoid(teacher_boundary) if teacher_boundary is not None else reliability_map
                
                self.prototype_bank.update(
                    projections=teacher_projections,
                    labels=pseudo_label,
                    reliability_map=reliability_map,
                    boundary_map=teacher_boundary_for_proto,
                    high_reliability_threshold=self.high_reliability_threshold
                )
                
                proto_losses = self.prototype_loss(
                    student_projections=student_projections,
                    teacher_projections=teacher_projections,
                    pseudo_labels=pseudo_label,
                    reliability_map=reliability_map,
                    boundary_map=teacher_boundary_for_proto,
                    high_reliability_threshold=self.high_reliability_threshold,
                    low_reliability_threshold=self.low_reliability_threshold,
                    pgpc_weight=self.pGPC_weight,
                    apc_weight=self.aPC_weight,
                    pac_weight=self.pAC_weight
                )
                
                losses['pgpc_loss'] = proto_losses['pgpc_loss']
                losses['apc_loss'] = proto_losses['apc_loss']
                losses['pac_loss'] = proto_losses['pac_loss']
        
        losses['total'] = sum(losses.values())
        return losses

    def train_step(self, batch: dict) -> dict:
        """
        训练步骤
        
        关键改进：无标签数据从同一个 batch 获取，然后分别应用 weak/strong transform
        确保 teacher 和 student 看到同一张图像的不同增强版本
        
        数据流:
        1. 无标签 dataloader 使用 _identity_transform，返回原始数据
        2. 在 train_step 中对原始数据分别应用 weak/strong augmentation
        3. Teacher 使用 weak augmentation，Student 使用 strong augmentation
        """
        data_labeled = batch['data']
        target_labeled = batch['target']
        
        try:
            batch_unlabeled = next(self.unlabeled_batch_iter)
        except StopIteration:
            self.unlabeled_batch_iter = iter(self.dataloader_unlabeled)
            batch_unlabeled = next(self.unlabeled_batch_iter)
        
        data_unlabeled_raw = batch_unlabeled['data']

        data_labeled = data_labeled.to(self.device, non_blocking=True)
        if isinstance(target_labeled, list):
            target_labeled = [i.to(self.device, non_blocking=True) for i in target_labeled]
        else:
            target_labeled = target_labeled.to(self.device, non_blocking=True)

        if isinstance(data_unlabeled_raw, np.ndarray):
            data_unlabeled_raw = torch.from_numpy(data_unlabeled_raw)
        data_unlabeled_raw = data_unlabeled_raw.to(self.device, non_blocking=True)

        # 从同一个 raw unlabeled batch 构造两种视图
        data_unlabeled_weak = self._apply_weak_augmentation(data_unlabeled_raw)
        data_unlabeled_strong = self._apply_strong_augmentation(data_unlabeled_raw)

        data_unlabeled_weak = data_unlabeled_weak.to(self.device, non_blocking=True)
        data_unlabeled_strong = data_unlabeled_strong.to(self.device, non_blocking=True)


        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            student_outputs_labeled = self.network(data_labeled)
            labeled_losses = self._compute_labeled_loss(student_outputs_labeled, target_labeled)
            total_loss = labeled_losses['total']
            
            consistency_weight = get_current_consistency_weight(
                self.current_epoch,
                warmup_epochs=self.consistency_warmup,
                rampup_end=self.consistency_rampup,
                max_weight=self.max_consistency_weight
            )
            
            if consistency_weight > 0:
                student_outputs_unlabeled = self.network(data_unlabeled_strong)
                
                teacher_model = self.ema_model.get_model()
                was_training = teacher_model.training
                teacher_model.train()
                with torch.no_grad():
                    teacher_outputs_unlabeled = teacher_model(data_unlabeled_weak)
                teacher_model.train(was_training)
                
                unlabeled_losses = self._compute_unlabeled_loss(student_outputs_unlabeled, teacher_outputs_unlabeled)
                
                total_loss = total_loss + unlabeled_losses['total'] * consistency_weight
            else:
                unlabeled_losses = {}
        
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
        
        self.ema_model.update(self.network)
        
        log_dict = {
            'loss': total_loss.detach().cpu().numpy(),
            'labeled_loss': labeled_losses['total'].detach().cpu().numpy(),
            'consistency_weight': consistency_weight
        }
        
        if unlabeled_losses:
            log_dict['unlabeled_loss'] = unlabeled_losses['total'].detach().cpu().numpy()
        
        return log_dict
    
    def _apply_weak_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        从 raw unlabeled batch 构造 teacher 的 weak view。
        这里只做轻量强度扰动，不做几何变化，保证与 student 输出空间对齐。
        """
        x = data.clone()

        # 轻微高斯噪声
        x = x + torch.randn_like(x) * 0.01

        # 轻微亮度/对比度扰动（逐样本）
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        scale = 0.95 + 0.10 * torch.rand(shape, device=x.device, dtype=x.dtype)
        shift = -0.02 + 0.04 * torch.rand(shape, device=x.device, dtype=x.dtype)
        x = x * scale + shift

        return x


    def _apply_strong_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        从同一个 raw unlabeled batch 构造 student 的 strong view。
        仍然只做强度类扰动，不做几何增强，避免 teacher/student 像素错位。
        """
        x = data.clone()

        shape = [x.shape[0]] + [1] * (x.ndim - 1)

        # 更强的亮度/对比度扰动
        scale = 0.80 + 0.40 * torch.rand(shape, device=x.device, dtype=x.dtype)
        shift = -0.10 + 0.20 * torch.rand(shape, device=x.device, dtype=x.dtype)
        x = x * scale + shift

        # 更强的高斯噪声
        x = x + torch.randn_like(x) * 0.03

        # 随机轻度模糊（保持空间尺寸不变）
        if x.ndim == 4 and torch.rand(1, device=x.device) < 0.5:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        elif x.ndim == 5 and torch.rand(1, device=x.device) < 0.5:
            x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)

        # 随机 dropout / cutout 风格的强度遮挡（不改坐标）
        if torch.rand(1, device=x.device) < 0.5:
            keep_mask = (torch.rand_like(x[:, :1]) > 0.10).to(x.dtype)
            x = x * keep_mask

        return x


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
                main_output = self.network(data)
                
                if isinstance(main_output, (list, tuple)):
                    main_outputs = list(main_output)
                else:
                    main_outputs = [main_output]
                
                if isinstance(target, (list, tuple)):
                    target_list = target
                else:
                    target_list = [target]
                
                l = self.loss(main_outputs, target_list)

            target_for_metrics = target_list[0]
            
            if target_for_metrics.ndim == 4 and target_for_metrics.shape[1] == 1:
                target_for_metrics = target_for_metrics[:, 0]
            
            main_output = main_outputs[0]
            output_seg = main_output.argmax(1)[:, None]
            
            predicted_segmentation_onehot = torch.zeros(
                main_output.shape, device=main_output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

            target_onehot = torch.zeros(
                main_output.shape, device=main_output.device, dtype=torch.float32
            )
            target_onehot.scatter_(1, target_for_metrics.long()[:, None], 1)

            axes = [0] + list(range(2, main_output.ndim))
            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot,
                target_onehot,
                axes=axes
            )

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
        original_forward = self.network.forward
        self.network.forward = self.network.forward_inference
        
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            self.network.forward = original_forward

    def on_epoch_end(self):
        super().on_epoch_end()
        
        if self.ema_model is not None:
            current_decay = self.ema_model.get_decay()
            print(f"📊 EMA decay: {current_decay:.6f}")


if __name__ == '__main__':
    print("Mean Teacher Trainer module loaded successfully!")
    print("Usage: nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaSDG_MeanTeacher")
