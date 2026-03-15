import torch
import torch.nn as nn
import numpy as np

def compute_urpc_loss(preds_unlabeled):
    """
    完全复刻 SSL4MIS 的 URPC 无标签一致性损失计算
    :param preds_unlabeled: list of tensors, 包含 [主输出, 辅助输出1, 辅助输出2...]，需已统一为相同分辨率
    """
    # 1. 转换为 Softmax 概率分布 (对应源码 outputs_soft, outputs_aux1_soft...)
    preds_soft = [torch.softmax(p, dim=1) for p in preds_unlabeled]
    
    # 2. 计算平均预测作为伪标签 Target (对应源码 preds = (...)/4)
    mean_pred = sum(preds_soft) / len(preds_soft)
    
    kl_distance = nn.KLDivLoss(reduction='none')
    consistency_loss_total = 0.0
    
    # 遍历主输出和所有的辅助输出
    for p_soft in preds_soft:
        # 3. 计算 KL 散度作为 Variance (对应源码 variance_main = sum(kl_distance(...)))
        # 注意: KLDivLoss 的 input 要求是 log_softmax, target 是 softmax
        variance = torch.sum(kl_distance(torch.log(p_soft + 1e-8), mean_pred), dim=1, keepdim=True)
        exp_variance = torch.exp(-variance)
        
        # 4. 计算一致性距离 (对应源码 consistency_dist_main = (preds - outputs_soft) ** 2)
        consistency_dist = (mean_pred - p_soft) ** 2
        
        # 5. 计算带不确定性整流的 Loss (对应源码 consistency_loss_main = ...)
        # 公式: mean(dist * exp_var) / (mean(exp_var) + 1e-8) + mean(var)
        loss_i = torch.mean(consistency_dist * exp_variance) / (torch.mean(exp_variance) + 1e-8) + torch.mean(variance)
        
        consistency_loss_total += loss_i
        
    # 求平均 (对应源码 consistency_loss = (...)/4)
    consistency_loss_final = consistency_loss_total / len(preds_soft)
    
    return consistency_loss_final

def get_current_consistency_weight(epoch, warmup_epochs=30, rampup_end=80, consistency=0.1):
    """
    前 warmup_epochs 个 epoch 完全关闭 URPC loss
    之后到 rampup_end 逐步升到 consistency
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch >= rampup_end:
        return consistency
    else:
        progress = (epoch - warmup_epochs) / (rampup_end - warmup_epochs)
        return consistency * np.exp(-5.0 * (1.0 - progress) ** 2)