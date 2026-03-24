from torch import nn
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UTANet_2d import get_utanet_2d_from_plans


class nnUNetTrainerUTANet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True
    ) -> nn.Module:

        if len(configuration_manager.patch_size) != 2:
            raise NotImplementedError("Only 2D UTANet is supported")

        model = get_utanet_2d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=False
        )

        print("UTANet2D: {}".format(model))
        print("[Baseline Mode] UTANet initialized successfully.")
        print("[UTANet Fix] Deep supervision forced OFF.")

        return model
