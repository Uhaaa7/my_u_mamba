import os
import torch
import contextlib
from torch.nn.functional import interpolate

# 异常捕获与相对路径引用
try:
    from ...nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG
except ModuleNotFoundError as e:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaSDG import nnUNetTrainerUMambaSDG

from .urpc_loss import compute_urpc_loss, get_current_consistency_weight

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper

class nnUNetTrainerUMambaSDG_URPC(nnUNetTrainerUMambaSDG):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.unlabeled_batch_iter = None
        
    def get_dataloaders(self):
        """
        重写数据加载逻辑：实例化有标签和无标签的 Dataset，并完美接入 nnU-Net 原生的数据增强与深层监督下采样管道。
        """
        # === 1. 构建原生的 Data Augmentation Transforms ===
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

        # === 2. 安全读取并划分 keys，绝不污染验证集 ===
        tr_keys, val_keys = self.do_split()
        
        labeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'labeled.txt')
        unlabeled_txt_path = os.path.join(self.preprocessed_dataset_folder, 'unlabeled.txt')
        
        if not os.path.exists(labeled_txt_path) or not os.path.exists(unlabeled_txt_path):
            raise FileNotFoundError(f"找不到半监督列表 {labeled_txt_path} 或 {unlabeled_txt_path}。")
            
        with open(labeled_txt_path, 'r') as f:
            labeled_cases = [line.strip() for line in f.readlines() if line.strip()]
        with open(unlabeled_txt_path, 'r') as f:
            unlabeled_cases = [line.strip() for line in f.readlines() if line.strip()]

        # 【关键保护】：只从当前 Fold 的训练集(tr_keys)中提取对应的样本
        tr_labeled_keys = [k for k in labeled_cases if k in tr_keys]
        tr_unlabeled_keys = [k for k in unlabeled_cases if k in tr_keys]

        # 实例化真正的 Dataset 对象
        dataset_tr_labeled = nnUNetDataset(self.preprocessed_dataset_folder, tr_labeled_keys,
                                           folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                           num_images_properties_loading_threshold=0)
        dataset_tr_unlabeled = nnUNetDataset(self.preprocessed_dataset_folder, tr_unlabeled_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                             num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)

        # === 3. 构建基础 DataLoader (12GB 显存切分策略) ===
        half_batch_size = max(1, self.batch_size // 2)
        dl_cls = nnUNetDataLoader2D if dim == 2 else nnUNetDataLoader3D
        
        dl_tr_labeled = dl_cls(dataset_tr_labeled, half_batch_size, initial_patch_size, self.configuration_manager.patch_size, self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent, sampling_probabilities=None, pad_sides=None)
        dl_tr_unlabeled = dl_cls(dataset_tr_unlabeled, half_batch_size, initial_patch_size, self.configuration_manager.patch_size, self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent, sampling_probabilities=None, pad_sides=None)
        dl_val = dl_cls(dataset_val, self.batch_size, self.configuration_manager.patch_size, self.configuration_manager.patch_size, self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent, sampling_probabilities=None, pad_sides=None)

        # === 4. 包裹多线程数据增强管线 (极其重要) ===
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_labeled = SingleThreadedAugmenter(dl_tr_labeled, tr_transforms)
            mt_gen_train_unlabeled = SingleThreadedAugmenter(dl_tr_unlabeled, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            # 将进程数平分给两路 DataLoader 防止僵死
            num_proc = max(1, allowed_num_processes // 2)
            mt_gen_train_labeled = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_labeled, transform=tr_transforms, num_processes=num_proc, num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_train_unlabeled = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_unlabeled, transform=tr_transforms, num_processes=num_proc, num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val, transform=val_transforms, num_processes=max(1, allowed_num_processes // 2), num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.02)

        self.dataloader_unlabeled = mt_gen_train_unlabeled
        return mt_gen_train_labeled, mt_gen_val

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.unlabeled_batch_iter = iter(self.dataloader_unlabeled)

    def train_step(self, batch: dict) -> dict:
        data_labeled = batch['data']
        target_labeled = batch['target']
        
        # 提取无标签批次
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
            
        data_unlabeled = data_unlabeled.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            # === 1. 有标签数据的前向传播与监督损失计算 ===
            output_labeled = self.network(data_labeled)
            
            # 解析网络输出 (完美适配你的 SDG aux head)
            if self.enable_aux_head and isinstance(output_labeled, tuple):
                main_output_labeled, aux_output_labeled = output_labeled
            else:
                main_output_labeled = output_labeled
                aux_output_labeled = None

            # 1.1 主损失：继续使用 nnUNet 原生 deep supervision loss
            l_sup = self.loss(main_output_labeled, target_labeled)
            
            # 1.2 辅助损失：使用你的单输出 CE loss
            if self.enable_aux_head and aux_output_labeled is not None:
                aux_target = target_labeled[0] if isinstance(target_labeled, list) else target_labeled
                if aux_target.ndim == 4 and aux_target.shape[1] == 1:
                    aux_target = aux_target[:, 0].long()
                else:
                    aux_target = aux_target.long()
                aux_loss = self.aux_ce_loss(aux_output_labeled, aux_target)
                l_sup = l_sup + self.aux_loss_weight * aux_loss

            # === 2. 无标签数据的前向传播与 URPC 损失计算 ===
            output_unlabeled = self.network(data_unlabeled)
            
            # 同样解包无标签的输出，URPC 只计算主网络深层监督的金字塔差异
            if self.enable_aux_head and isinstance(output_unlabeled, tuple):
                main_output_unlabeled, _ = output_unlabeled
            else:
                main_output_unlabeled = output_unlabeled
            
            if isinstance(main_output_unlabeled, (list, tuple)):
                target_shape = main_output_unlabeled[0].shape[2:] 
                is_3d = len(target_shape) == 3
                
                upsampled_outputs = []
                for p in main_output_unlabeled:
                    if p.shape[2:] != target_shape:
                        p_up = interpolate(
                            p, 
                            size=target_shape, 
                            mode='trilinear' if is_3d else 'bilinear', 
                            align_corners=False
                        )
                        upsampled_outputs.append(p_up)
                    else:
                        upsampled_outputs.append(p)
                
                # 计算多尺度一致性损失
                l_cons = compute_urpc_loss(upsampled_outputs)
            else:
                raise ValueError("未检测到多尺度金字塔输出，请确保 deep_supervision=True。")
            
            # === 3. 合并总损失 ===
            weight = get_current_consistency_weight(self.current_epoch)
            l = l_sup + weight * l_cons
            
        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {'loss': l.detach().cpu().numpy()}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        重写验证方法：通过 PyTorch 的 forward hook 拦截网络输出。
        在最终的滑动窗口推理（包含 TTA 翻转）时，自动过滤掉 auxiliary head 产生的 tuple，
        从而防止 nnUNetPredictor 内部的 torch.flip() 报错。
        """
        # 1. 定义一个钩子函数，拦截并修改前向传播的输出
        def unpack_tuple_hook(module, input, output):
            if isinstance(output, tuple):
                return output[0]  # 只保留主输出供推理使用，丢弃 aux 辅助输出
            return output

        # 2. 临时将钩子挂载到网络上
        hook_handle = self.network.register_forward_hook(unpack_tuple_hook)

        try:
            # 3. 调用父类的原生验证流程 (此时 Predictor 拿到的将是纯净的 Tensor)
            super().perform_actual_validation(save_probabilities)
        finally:
            # 4. 无论验证是否成功，最后务必卸载钩子，保持网络纯净
            hook_handle.remove()