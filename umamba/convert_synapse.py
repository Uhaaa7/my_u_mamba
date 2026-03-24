import numpy as np
import os
import h5py
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

# 路径配置
NPZ_TRAIN = "/home/dministrator/TransUNet_Project/project_TransUNet/data/Synapse/train_npz"
H5_TEST = "/home/dministrator/TransUNet_Project/project_TransUNet/data/Synapse/test_vol_h5"
RAW_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset100_Synapse"

maybe_mkdir_p(join(RAW_DIR, "imagesTr"))
maybe_mkdir_p(join(RAW_DIR, "labelsTr"))
maybe_mkdir_p(join(RAW_DIR, "imagesTs"))
maybe_mkdir_p(join(RAW_DIR, "labelsTs"))

# 1. 转换训练集 (2211个 .npz 切片)
for f in os.listdir(NPZ_TRAIN):
    if f.endswith('.npz'):
        data = np.load(join(NPZ_TRAIN, f))
        # 增加伪 3D 维度以符合 nnU-Net 规范
        img = nib.Nifti1Image(data['image'][None, ...], np.eye(4))
        lbl = nib.Nifti1Image(data['label'][None, ...].astype(np.uint8), np.eye(4))
        case_id = f.replace('.npz', '')
        nib.save(img, join(RAW_DIR, "imagesTr", f"{case_id}_0000.nii.gz"))
        nib.save(lbl, join(RAW_DIR, "labelsTr", f"{case_id}.nii.gz"))

# 2. 转换测试集 (12个 .h5 卷)
for f in os.listdir(H5_TEST):
    if f.endswith('.h5'):
        with h5py.File(join(H5_TEST, f), 'r') as hf:
            img = nib.Nifti1Image(hf['image'][:], np.eye(4))
            lbl = nib.Nifti1Image(hf['label'][:].astype(np.uint8), np.eye(4))
            case_id = f.replace('.npz.h5', '') # 处理文件名差异
            nib.save(img, join(RAW_DIR, "imagesTs", f"{case_id}_0000.nii.gz"))
            nib.save(lbl, join(RAW_DIR, "labelsTs", f"{case_id}.nii.gz"))