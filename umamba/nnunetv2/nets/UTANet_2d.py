import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He


class Expert(nn.Module):
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(),
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MoE(nn.Module):
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 64):
        super().__init__()
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(num_experts)])
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.top = top
        
    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> torch.Tensor:
        batch_size, emb_size, H, W = x.shape
        
        x0 = self.gap(x).view(batch_size, emb_size)
        gate_out = F.softmax(x0 @ gate_weights, dim=1)
        
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)
        used_experts = torch.unique(top_index)
        unused_experts = set(range(len(self.experts))) - set(used_experts.tolist())
        
        top_weights = F.softmax(top_weights, dim=1)
        
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        y = torch.zeros_like(x_expanded)
        
        for expert_i, expert_model in enumerate(self.experts):
            expert_mask = (top_index == expert_i).view(-1)
            expert_indices = expert_mask.nonzero().flatten()
            
            if expert_indices.numel() > 0:
                x_expert = x_expanded[expert_indices]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                random_sample = torch.randint(0, x.size(0), (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)
        
        top_weights = top_weights.view(-1, 1, 1, 1).expand_as(y)
        y = y * top_weights
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)
        
        return y
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y1 = self._process_gate(x, self.gate1)
        y2 = self._process_gate(x, self.gate2)
        y3 = self._process_gate(x, self.gate3)
        y4 = self._process_gate(x, self.gate4)
        
        return y1, y2, y3, y4


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UTANet2D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        base_channels: int = 64,
        use_moe: bool = True,
        deep_supervision: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_moe = use_moe
        self.deep_supervision = deep_supervision

        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        if self.use_moe:
            self.fuse = nn.Sequential(
                nn.Conv2d(base_channels * 15, base_channels, 1, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.moe = MoE(num_experts=4, top=2, emb_size=base_channels)
            self.docker1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 1, 1, bias=True),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.docker2 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 1, 1, bias=True),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
            self.docker3 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 4, 1, 1, bias=True),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
            self.docker4 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 8, 1, 1, bias=True),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self.decoder = nn.Module()
        self.decoder.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_moe:
            x1_resized = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
            x3_resized = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4_resized = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=False)
            
            fused = torch.cat([x1_resized, x2, x3_resized, x4_resized], dim=1)
            fused = self.fuse(fused)
            
            o1, o2, o3, o4 = self.moe(fused)
            
            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)
            
            x5_h, x5_w = x5.shape[2:]
            o4 = F.interpolate(o4, size=(x5_h, x5_w), mode='bilinear', align_corners=False)
            
            x4_h, x4_w = x4.shape[2:]
            o3 = F.interpolate(o3, size=(x4_h, x4_w), mode='bilinear', align_corners=False)
            
            x2_h, x2_w = x2.shape[2:]
            o2 = F.interpolate(o2, size=(x2_h, x2_w), mode='bilinear', align_corners=False)
            
            o1 = F.interpolate(o1, size=input_size, mode='bilinear', align_corners=False)
        else:
            o1, o2, o3, o4 = x1, x2, x3, x4

        x = self.up1(x5, o4)
        x = self.up2(x, o3)
        x = self.up3(x, o2)
        x = self.up4(x, o1)
        
        logits = self.outc(x)
        
        return logits


def get_utanet_2d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = False
):
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads

    model = UTANet2D(
        input_channels=num_input_channels,
        num_classes=num_classes,
        base_channels=64,
        use_moe=True,
        deep_supervision=deep_supervision
    )
    model.apply(InitWeights_He(1e-2))

    return model
