import os
import numpy as np
import SimpleITK as sitk

try:
    from medpy import metric
except ModuleNotFoundError:
    print("\n❌ ModuleNotFoundError: 缺少计算 HD95 的关键依赖库 'medpy'。")
    print("👉 解决方案: 请在你的 umamba 虚拟环境中执行以下命令进行安装：")
    print("   pip install medpy")
    print("   安装完成后重新运行本脚本即可。\n")
    exit()

PRED_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/predictions_SwinUNETR"
GT_DIR = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/labelsTs"

CLASSES = {1: "RV (右心室)", 2: "Myo (心肌)", 3: "LV (左心室)"}

def calculate_metrics(pred, gt, class_id, voxel_spacing):
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return np.nan, np.nan
        
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0, np.nan
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())
    
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

        pred_sitk = sitk.ReadImage(pred_path)
        gt_sitk = sitk.ReadImage(gt_path)
        
        pred_img = sitk.GetArrayFromImage(pred_sitk)
        gt_img = sitk.GetArrayFromImage(gt_sitk)
        
        spacing = gt_sitk.GetSpacing()[::-1] 

        for class_id in CLASSES.keys():
            dice, hd95 = calculate_metrics(pred_img, gt_img, class_id, spacing)
            
            if not np.isnan(dice):
                all_dices[class_id].append(dice)
            if not np.isnan(hd95):
                all_hd95s[class_id].append(hd95)
    
    output_lines = []
    
    output_lines.append("="*65)
    output_lines.append("🏆 ACDC 测试集最终成绩单")
    output_lines.append("="*65)
    output_lines.append("器官类别            | Dice (%) ↑           | HD95 (mm) ↓         ")
    output_lines.append("-" * 65)
    
    mean_dices, mean_hd95s = [], []
    
    for class_id, name in CLASSES.items():
        if all_dices[class_id]:
            d_np = np.array(all_dices[class_id]) * 100
            d_mean = np.mean(d_np)
            d_std = np.std(d_np, ddof=1)
            mean_dices.append(d_mean)
            dice_str = f"{d_mean:.2f} ± {d_std:.2f}"
        else:
            dice_str = "N/A"
            
        if all_hd95s[class_id]:
            h_np = np.array(all_hd95s[class_id])
            h_mean = np.mean(h_np)
            h_std = np.std(h_np, ddof=1)
            mean_hd95s.append(h_mean)
            hd_str = f"{h_mean:.2f} ± {h_std:.2f}"
        else:
            hd_str = "N/A"
            
        output_lines.append(f"{name:<15} | {dice_str:<20} | {hd_str:<20}")
            
    output_lines.append("-" * 65)
    if mean_dices and mean_hd95s:
        output_lines.append(f"⭐ Average        | {np.mean(mean_dices):.2f}%               | {np.mean(mean_hd95s):.2f} mm")
    output_lines.append("="*65)
    
    output_lines.append("")
    output_lines.append("="*75)
    output_lines.append("📊 详细统计信息 (用于论文写作)")
    output_lines.append("="*75)
    output_lines.append("📌 说明:")
    output_lines.append("   - SD (标准差): 反映病例间的变异性，数值大说明不同病例分割效果差异大")
    output_lines.append("   - SEM (标准误 = SD/√n): 反映均值估计的精度，数值小说明均值估计可靠")
    output_lines.append("   - 论文中常见做法: 报告 Mean±SD 或 Mean±SEM，请根据目标期刊要求选择")
    output_lines.append("-" * 75)
    output_lines.append("器官类别       | n    | Mean    | SD      | SEM     | Min     | Max     ")
    output_lines.append("-" * 75)
    
    for class_id, name in CLASSES.items():
        if all_dices[class_id]:
            d_np = np.array(all_dices[class_id]) * 100
            n = len(d_np)
            mean = np.mean(d_np)
            std = np.std(d_np, ddof=1)
            sem = std / np.sqrt(n)
            output_lines.append(f"{name:<12} | {n:<4} | {mean:<8.2f} | {std:<8.2f} | {sem:<8.2f} | {np.min(d_np):<8.2f} | {np.max(d_np):<8.2f}")
    
    output_lines.append("="*75)
    
    output_lines.append("")
    output_lines.append("📝 论文写作建议:")
    output_lines.append("   1. 如果期刊要求报告 Mean±SD:")
    output_lines.append("      - 这反映的是病例间的变异性")
    output_lines.append("      - SD 较大说明不同病例的分割难度差异大，这是医学图像的正常现象")
    output_lines.append("   2. 如果期刊要求报告 Mean±SEM:")
    output_lines.append("      - 这反映的是均值估计的精度")
    output_lines.append("      - SEM 通常比 SD 小很多，看起来更'好看'")
    output_lines.append("   3. ACDC 挑战赛官方排行榜通常报告 Mean±SD")
    output_lines.append("   4. 如果 SD 确实较大，可以在论文中解释:")
    output_lines.append("      '较大的标准差反映了 ACDC 数据集中不同病例间解剖结构和图像质量的显著差异'")
    
    output_lines.append("")
    output_lines.append("="*75)
    output_lines.append("📋 可直接复制到论文的表格 (Mean±SD 格式):")
    output_lines.append("="*75)
    output_lines.append("Class      | Dice (%)       ")
    output_lines.append("-" * 30)
    for class_id, name in CLASSES.items():
        if all_dices[class_id]:
            d_np = np.array(all_dices[class_id]) * 100
            mean = np.mean(d_np)
            std = np.std(d_np, ddof=1)
            short_name = name.split('(')[0].strip()
            output_lines.append(f"{short_name:<10} | {mean:.2f}±{std:.2f}")
    if mean_dices:
        avg_std = np.mean([np.std(np.array(all_dices[c])*100, ddof=1) for c in CLASSES.keys() if all_dices[c]])
        output_lines.append(f"{'Avg':<10} | {np.mean(mean_dices):.2f}±{avg_std:.2f}")
    
    output_lines.append("")
    output_lines.append("="*75)
    output_lines.append("📋 可直接复制到论文的表格 (Mean±SEM 格式):")
    output_lines.append("="*75)
    output_lines.append("Class      | Dice (%)       ")
    output_lines.append("-" * 30)
    for class_id, name in CLASSES.items():
        if all_dices[class_id]:
            d_np = np.array(all_dices[class_id]) * 100
            n = len(d_np)
            mean = np.mean(d_np)
            sem = np.std(d_np, ddof=1) / np.sqrt(n)
            short_name = name.split('(')[0].strip()
            output_lines.append(f"{short_name:<10} | {mean:.2f}±{sem:.2f}")
    if mean_dices:
        avg_sem = np.mean([np.std(np.array(all_dices[c])*100, ddof=1)/np.sqrt(len(all_dices[c])) for c in CLASSES.keys() if all_dices[c]])
        output_lines.append(f"{'Avg':<10} | {np.mean(mean_dices):.2f}±{avg_sem:.2f}")
    
    output_content = "\n".join(output_lines)
    
    print(output_content)
    
    output_file = os.path.join(PRED_DIR, "eval_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
