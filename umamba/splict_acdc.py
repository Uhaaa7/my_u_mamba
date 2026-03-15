import os
import shutil
import json

# ==========================================
# 1. 路径配置区 (请修改为你真实的绝对路径)
# ==========================================
# 你的列表文件存放的文件夹
LIST_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC" 

# nnUNet 原始数据集的根目录
RAW_DATA_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC"

# nnUNet 预处理后的数据集目录 (用于存放 splits_final.json)
PREPROCESSED_DIR = "/home/dministrator/U-Mamba/data/nnUNet_preprocessed/Dataset027_ACDC"

# ==========================================
# 2. 读取名单函数
# ==========================================
def read_list(file_name):
    file_path = os.path.join(LIST_DIR, file_name)
    with open(file_path, 'r') as f:
        # 读取每一行并去除换行符，忽略空行
        return [line.strip() for line in f.readlines() if line.strip()]

train_cases = read_list('train.list')
val_cases = read_list('val.list')
test_cases = read_list('test.list')

print(f"✅ 成功读取名单: 训练集 {len(train_cases)} 例, 验证集 {len(val_cases)} 例, 测试集 {len(test_cases)} 例")

# ==========================================
# 3. 物理隔离测试集 (移至 imagesTs / labelsTs)
# ==========================================
images_tr_dir = os.path.join(RAW_DATA_DIR, "imagesTr")
labels_tr_dir = os.path.join(RAW_DATA_DIR, "labelsTr")
images_ts_dir = os.path.join(RAW_DATA_DIR, "imagesTs")
labels_ts_dir = os.path.join(RAW_DATA_DIR, "labelsTs")

# 确保目标文件夹存在
os.makedirs(images_ts_dir, exist_ok=True)
os.makedirs(labels_ts_dir, exist_ok=True)

moved_count = 0
for case in test_cases:
    # 图像文件必须带 _0000.nii.gz
    img_src = os.path.join(images_tr_dir, f"{case}_0000.nii.gz")
    img_dst = os.path.join(images_ts_dir, f"{case}_0000.nii.gz")
    
    # 标签文件不带 _0000
    lbl_src = os.path.join(labels_tr_dir, f"{case}.nii.gz")
    lbl_dst = os.path.join(labels_ts_dir, f"{case}.nii.gz")
    
    # 移动图像
    if os.path.exists(img_src):
        shutil.move(img_src, img_dst)
        moved_count += 1
    
    # 移动标签
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)

print(f"✅ 成功将 {moved_count} 个测试集图像及其标签移动至 Ts 文件夹！")

# ==========================================
# 4. 生成 splits_final.json
# ==========================================
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
splits_path = os.path.join(PREPROCESSED_DIR, "splits_final.json")

# 严格按照 nnU-Net 要求的格式构建列表包裹字典的结构
splits_data = [
    {
        "train": train_cases,
        "val": val_cases
    }
]

with open(splits_path, 'w') as f:
    json.dump(splits_data, f, indent=4)

print(f"✅ 成功生成固定划分文件: {splits_path}")
print("🎉 数据集划分全部完成，你可以开始执行 nnUNetv2_train 了！")