"""
ABD (Attention-Based Dual-branch) 损失函数

核心损失:
1. 监督损失: CE + Dice
2. 伪标签损失: Cross Teaching
3. ABD-I 损失: 位移后的监督损失
4. ABD-R 损失: 位移后的伪标签损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation.
    inputs: [B, C, H, W]
    target: [B, H, W] or [B, 1, H, W]
    """

    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim == 4 and input_tensor.shape[1] == 1:
            input_tensor = input_tensor[:, 0]

        if input_tensor.ndim != 3:
            raise ValueError(
                f"target must be [B,H,W] or [B,1,H,W], got {tuple(input_tensor.shape)}"
            )

        return F.one_hot(
            input_tensor.long(), num_classes=self.n_classes
        ).permute(0, 3, 1, 2).float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        return 1.0 - loss

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        softmax: bool = True
    ) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)

        if inputs.shape != target.shape:
            raise AssertionError(
                f"predict & target shape do not match: {inputs.shape} vs {target.shape}"
            )

        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


def compute_supervised_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    ce_loss: nn.Module,
    dice_loss: DiceLoss
) -> torch.Tensor:
    """
    output: [B, C, H, W]
    target: [B, H, W]
    """
    target = target.long()
    loss_ce = ce_loss(output, target)
    loss_dice = dice_loss(output, target, softmax=True)
    return 0.5 * (loss_ce + loss_dice)


def compute_pseudo_supervision_loss(
    output: torch.Tensor,
    pseudo_label: torch.Tensor,
    dice_loss: DiceLoss
) -> torch.Tensor:
    """
    output: [B, C, H, W]
    pseudo_label: [B, H, W]
    """
    pseudo_label = pseudo_label.long()
    return dice_loss(output, pseudo_label, softmax=True)


def compute_abd_loss(
    outputs1_labeled: torch.Tensor,
    outputs2_labeled: torch.Tensor,
    outputs1_unlabeled: torch.Tensor,
    outputs2_unlabeled: torch.Tensor,
    target_labeled: torch.Tensor,
    image_output_supervised_1: torch.Tensor,
    image_output_supervised_2: torch.Tensor,
    label_patch_supervised: torch.Tensor,
    image_output_1: torch.Tensor,
    image_output_2: torch.Tensor,
    pseudo_image_output_1: torch.Tensor,
    pseudo_image_output_2: torch.Tensor,
    labeled_bs: int,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
    consistency_weight: float,
    iteration: int = 0,
    disable_abd_i_after: int = 20000
) -> dict:
    """
    保留接口，内部逻辑修正为真正基于 unlabeled logits 的 cross teaching。
    如果外部没有传 ABD-I / ABD-R 有效结果，相应项可为零张量。
    """
    loss1 = compute_supervised_loss(
        outputs1_labeled[:labeled_bs], target_labeled[:labeled_bs], ce_loss, dice_loss
    )
    loss2 = compute_supervised_loss(
        outputs2_labeled[:labeled_bs], target_labeled[:labeled_bs], ce_loss, dice_loss
    )

    if outputs1_unlabeled.shape[0] > 0:
        pseudo_outputs1_u = torch.argmax(torch.softmax(outputs1_unlabeled.detach(), dim=1), dim=1)
        pseudo_outputs2_u = torch.argmax(torch.softmax(outputs2_unlabeled.detach(), dim=1), dim=1)

        pseudo_supervision1 = compute_pseudo_supervision_loss(
            outputs1_unlabeled, pseudo_outputs2_u, dice_loss
        )
        pseudo_supervision2 = compute_pseudo_supervision_loss(
            outputs2_unlabeled, pseudo_outputs1_u, dice_loss
        )
    else:
        zero = torch.tensor(0.0, device=outputs1_labeled.device)
        pseudo_supervision1 = zero
        pseudo_supervision2 = zero

    if iteration > disable_abd_i_after:
        zero = torch.tensor(0.0, device=outputs1_labeled.device)
        loss3 = zero
        loss4 = zero
    else:
        loss3 = compute_supervised_loss(
            image_output_supervised_1, label_patch_supervised, ce_loss, dice_loss
        )
        loss4 = compute_supervised_loss(
            image_output_supervised_2, label_patch_supervised, ce_loss, dice_loss
        )

    if image_output_1.shape[0] > 0:
        pseudo_supervision3 = compute_pseudo_supervision_loss(
            image_output_1, pseudo_image_output_2, dice_loss
        )
        pseudo_supervision4 = compute_pseudo_supervision_loss(
            image_output_2, pseudo_image_output_1, dice_loss
        )
    else:
        zero = torch.tensor(0.0, device=outputs1_labeled.device)
        pseudo_supervision3 = zero
        pseudo_supervision4 = zero

    model1_loss = loss1 + 2.0 * loss3 + consistency_weight * (pseudo_supervision1 + pseudo_supervision3)
    model2_loss = loss2 + 2.0 * loss4 + consistency_weight * (pseudo_supervision2 + pseudo_supervision4)
    total_loss = model1_loss + model2_loss

    return {
        'total_loss': total_loss,
        'loss1': loss1,
        'loss2': loss2,
        'loss3': loss3,
        'loss4': loss4,
        'pseudo_supervision1': pseudo_supervision1,
        'pseudo_supervision2': pseudo_supervision2,
        'pseudo_supervision3': pseudo_supervision3,
        'pseudo_supervision4': pseudo_supervision4,
        'model1_loss': model1_loss,
        'model2_loss': model2_loss,
        'consistency_weight': consistency_weight,
    }