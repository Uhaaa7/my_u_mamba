import os
import json
import glob

# 1. 设置正确的绝对路径 (基于你刚才 ls 的输出)
preprocessed_dir = "/home/dministrator/U-Mamba/data/nnUNet_preprocessed/Dataset100_Synapse/nnUNetPlans_2d"
split_file_path = "/home/dministrator/U-Mamba/data/nnUNet_preprocessed/Dataset100_Synapse/splits_final.json"

# 2. 读取你原有的、只有基础 case 名字的 split 文件
with open(split_file_path, 'r') as f:
    splits = json.load(f)

# 3. 扫描硬盘上实际存在的切片文件
all_pkl_files = glob.glob(os.path.join(preprocessed_dir, "*.pkl"))
# 提取出实际的切片名字 (例如从 '/.../case0005_slice000.pkl' 提取 'case0005_slice000')
actual_slice_names = [os.path.basename(p).replace(".pkl", "") for p in all_pkl_files]

new_splits = []
for fold_split in splits:
    new_fold = {"train": [], "val": []}
    
    # 展开训练集
    for base_case in fold_split["train"]:
        # 找到所有以该 case 开头的 slice
        matching_slices = [s for s in actual_slice_names if s.startswith(base_case)]
        new_fold["train"].extend(matching_slices)
        
    # 展开验证集
    for base_case in fold_split["val"]:
        matching_slices = [s for s in actual_slice_names if s.startswith(base_case)]
        new_fold["val"].extend(matching_slices)
        
    new_splits.append(new_fold)

# 4. 覆盖保存为符合实际文件名的 splits_final.json (建议先备份原文件)
backup_path = split_file_path.replace(".json", "_backup.json")
os.rename(split_file_path, backup_path)
print(f"原 split 文件已备份至: {backup_path}")

with open(split_file_path, 'w') as f:
    json.dump(new_splits, f, indent=4)
print(f"成功生成新的展开版 splits_final.json！现在可以直接启动训练了。")