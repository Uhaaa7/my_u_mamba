import os
import numpy as np
import nibabel as nib

# ==========================================
# 🚨 请把下面这两个路径改成你电脑上的实际路径
# ==========================================
# 你的 U-Mamba 预测结果文件夹
pred_dir = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/predictions_URPC" 
# ACDC 测试集的真实标签文件夹 (Ground Truth)
gt_dir = "/home/dministrator/U-Mamba/data/nnUNet_raw/Dataset027_ACDC/labelsTs"  

def calculate_dice(pred, gt, class_label=1):
    """计算单个类别的 Dice 分数"""
    pred_mask = (pred == class_label)
    gt_mask = (gt == class_label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    volume_sum = pred_mask.sum() + gt_mask.sum()
    
    if volume_sum == 0:
        return np.nan # 如果预测和真实标签里都没这个器官，返回 NaN
    return 2. * intersection / volume_sum

def main():
    results = []
    
    # 遍历预测文件夹里的所有 nii.gz 文件
    for filename in os.listdir(pred_dir):
        if not filename.endswith(".nii.gz"):
            continue
            
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename) # 假设预测和真实标签文件名一样
        
        # 如果文件名不一样（比如真实标签带有 _gt.nii.gz 后缀），这里需要稍微处理一下字符串
        # gt_path = os.path.join(gt_dir, filename.replace(".nii.gz", "_gt.nii.gz"))
        
        if not os.path.exists(gt_path):
            print(f"找不到对应的金标准文件: {gt_path}")
            continue
            
        # 读取图像数据
        pred_data = nib.load(pred_path).get_fdata()
        gt_data = nib.load(gt_path).get_fdata()
        
        # 计算 RV (类别 1) 的 Dice
        rv_dice = calculate_dice(pred_data, gt_data, class_label=3)
        
        results.append({
            "patient": filename,
            "rv_dice": rv_dice
        })
        
    # 按 RV 的 Dice 分数从低到高排序
    results.sort(key=lambda x: x["rv_dice"] if not np.isnan(x["rv_dice"]) else 999)
    
    print("\n" + "="*50)
    print("🏆 RV (右心室) Dice 成绩单 (从低到高排名)")
    print("="*50)
    for res in results:
        dice_score = res['rv_dice']
        score_str = f"{dice_score * 100:.2f}%" if not np.isnan(dice_score) else "N/A"
        
        # 高亮显示低于 10% 的极差结果
        if not np.isnan(dice_score) and dice_score < 0.1:
            print(f"🚨 翻车警告 -> 病历: {res['patient']} | RV Dice: {score_str}")
        else:
            print(f"✅ 正常发挥 -> 病历: {res['patient']} | RV Dice: {score_str}")

if __name__ == "__main__":
    main()