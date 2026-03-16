import os
import numpy as np
import SimpleITK as sitk

# 异常捕获：处理可能不存在的依赖库
try:
    from medpy import metric
except ModuleNotFoundError:
    print("\n❌ ModuleNotFoundError: 缺少计算 HD95 的关键依赖库 'medpy'。")
    print("👉 解决方案: 请在你的 umamba 虚拟环境中执行以下命令进行安装：")
    print("   pip install medpy")
    print("   安装完成后重新运行本脚本即可。\n")
    exit()

# ==========================================
# 1. 路径配置区 (运行前请务必检查这里！)
# ==========================================
# 预测结果文件夹 (先填 SDG 的，跑完再换成 Bot 的)
PRED_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/predictions_SDG"

# 测试集真实标签文件夹 (标准答案)
GT_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/labelsTs"

# ==========================================
# 2. 核心算分逻辑
# ==========================================
CLASSES = {1: "RV (右心室)", 2: "Myo (心肌)", 3: "LV (左心室)"}

def calculate_metrics(pred, gt, class_id, voxel_spacing):
    """同时计算单个类别的 Dice 和 HD95"""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    
    # 极端情况一：预测和真值都没有这个器官 (完美预测为背景)
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return np.nan, np.nan
        
    # 极端情况二：真值有，但没预测出来；或真值没有，但误判出了器官
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0, np.nan # HD95 在空集下数学上无定义，记为 NaN 不参与均值计算
    
    # 计算 Dice
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())
    
    # 计算 HD95 (必须传入 spacing 物理间距，否则算出来的是体素个数而不是毫米)
    try:
        hd95 = metric.binary.hd95(pred_mask, gt_mask, voxelspacing=voxel_spacing)
    except Exception as e:
        hd95 = np.nan
        
    return dice, hd95

def main():
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')])
    
    if not pred_files:
        print(f"❌ 在 {PRED_DIR} 中没有找到 .nii.gz 文件！请检查路径。")
        return

    all_dices = {1: [], 2: [], 3: []}
    all_hd95s = {1: [], 2: [], 3: []}

    print(f"🔍 正在对比文件夹: \n预测: {PRED_DIR}\n真值: {GT_DIR}\n")

    for pred_filename in pred_files:
        pred_path = os.path.join(PRED_DIR, pred_filename)
        gt_path = os.path.join(GT_DIR, pred_filename)

        if not os.path.exists(gt_path):
            print(f"⚠️ 警告: 找不到对应的真实标签 {pred_filename}，跳过。")
            continue

        # 读取图像信息
        pred_sitk = sitk.ReadImage(pred_path)
        gt_sitk = sitk.ReadImage(gt_path)
        
        # 提取 Numpy 矩阵
        pred_img = sitk.GetArrayFromImage(pred_sitk)
        gt_img = sitk.GetArrayFromImage(gt_sitk)
        
        # 💡 核心细节：SimpleITK 的间距是 (X, Y, Z)，而 Numpy 数组是 (Z, Y, X)
        # 所以必须把 spacing 倒序传递给 medpy，才能保证 HD95 的物理距离计算正确！
        spacing = gt_sitk.GetSpacing()[::-1] 

        for class_id in CLASSES.keys():
            dice, hd95 = calculate_metrics(pred_img, gt_img, class_id, spacing)
            
            if not np.isnan(dice):
                all_dices[class_id].append(dice)
            if not np.isnan(hd95):
                all_hd95s[class_id].append(hd95)
                
    # ==========================================
    # 3. 打印可以直接抄进论文的表格数据
    # ==========================================
    print("="*65)
    print(f"🏆 ACDC 测试集最终成绩单")
    print("="*65)
    print(f"{'器官类别':<15} | {'Dice (%) ↑':<20} | {'HD95 (mm) ↓':<20}")
    print("-" * 65)
    
    mean_dices, mean_hd95s = [], []
    
    for class_id, name in CLASSES.items():
        # 处理 Dice
        if all_dices[class_id]:
            d_np = np.array(all_dices[class_id]) * 100
            d_mean, d_std = np.mean(d_np), np.std(d_np)
            mean_dices.append(d_mean)
            dice_str = f"{d_mean:.2f} ± {d_std:.2f}"
        else:
            dice_str = "N/A"
            
        # 处理 HD95
        if all_hd95s[class_id]:
            h_np = np.array(all_hd95s[class_id])
            h_mean, h_std = np.mean(h_np), np.std(h_np)
            mean_hd95s.append(h_mean)
            hd_str = f"{h_mean:.2f} ± {h_std:.2f}"
        else:
            hd_str = "N/A"
            
        print(f"{name:<15} | {dice_str:<20} | {hd_str:<20}")
            
    print("-" * 65)
    if mean_dices and mean_hd95s:
        print(f"⭐ Average        | {np.mean(mean_dices):.2f}%               | {np.mean(mean_hd95s):.2f} mm")
    print("="*65)

if __name__ == "__main__":
    main()