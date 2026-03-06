#!/bin/bash
# ============================================================================
# ACDC 数据集训练脚本 - RTX 4070 12GB
# ============================================================================

export nnUNet_raw="/home/dministrator/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/home/dministrator/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/home/dministrator/U-Mamba/data/nnUNet_results"

source /home/dministrator/miniconda3/etc/profile.d/conda.sh
conda activate umamba

DATASET_ID=27

echo "=========================================="
echo "ACDC 数据集训练"
echo "=========================================="
echo ""
echo "训练命令:"
echo ""
echo "【Baseline (原版 UMamba)】"
echo "nnUNetv2_train $DATASET_ID 2d 0 --trainer nnUNetTrainerUMambaBaseline --npz"
echo ""
echo "【改进版 (UMamba + SDG_Block)】"
echo "nnUNetv2_train $DATASET_ID 2d 0 --trainer nnUNetTrainerUMambaSDG --npz"
echo ""
echo "=========================================="

if [ "$1" = "baseline" ]; then
    echo "开始训练 Baseline..."
    nnUNetv2_train $DATASET_ID 2d 0 --trainer nnUNetTrainerUMambaBaseline --npz
elif [ "$1" = "improved" ]; then
    echo "开始训练改进版..."
    nnUNetv2_train $DATASET_ID 2d 0 --trainer nnUNetTrainerUMambaSDG --npz
else
    echo ""
    echo "使用方法:"
    echo "  ./train_acdc.sh baseline   # 训练 Baseline"
    echo "  ./train_acdc.sh improved  # 训练改进版"
fi
