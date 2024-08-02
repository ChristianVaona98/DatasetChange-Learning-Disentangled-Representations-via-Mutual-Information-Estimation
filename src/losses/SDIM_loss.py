import torch
import torch.nn as nn
from losses.loss_functions import DJSLoss, ClassifLoss
from utils.custom_typing import SDIMSmallNORBLosses, SDIMSmallNORBOutputs


class SDIMLoss(nn.Module):
    """Loss function to extract shared information from the image, see paper equation (5)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        shared_loss_coeff (float): Coefficient of L1 loss, see paper equation (6)
    """

    def __init__(
        self,
        local_mutual_loss_coeff: float,
        global_mutual_loss_coeff: float,
        shared_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.l1_loss = nn.L1Loss()  # see equation (6)

    def __call__(
        self,
        sdim_outputs: SDIMSmallNORBOutputs,
        cat_labels: torch.Tensor,
        elevation_labels: torch.Tensor,
        lightning_labels: torch.Tensor,
    ) -> SDIMSmallNORBLosses:
        """Compute all the loss functions needed to extract the shared part

        Args:
            sdim_outputs (SDIMOutputs): Output of the forward pass of the shared information model
            digit_labels (torch.Tensor): Label of the digit
            color_bg_labels (torch.Tensor): Background color of the images
            color_fg_labels (torch.Tensor): Foreground color of the images

        Returns:
            SDIMLosses: Shared information losses
        """

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_x,
            T_prime=sdim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_y,
            T_prime=sdim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_x,
            T_prime=sdim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_y,
            T_prime=sdim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        # Compute L1 on shared features
        shared_loss = self.l1_loss(sdim_outputs.shared_x, sdim_outputs.shared_y)
        shared_loss = shared_loss * self.shared_loss_coeff

        # Get classification error
        cat_classif_loss, cat_accuracy = self.classif_loss(
            y_pred=sdim_outputs.cat_logits, target=cat_labels
        )
        elevation_classif_loss, elevation_accuracy = self.classif_loss(
            y_pred=sdim_outputs.elevation_logits, target=elevation_labels
        )
        lightning_classif_loss, lightning_accuracy = self.classif_loss(
            y_pred=sdim_outputs.lightning_logits, target=lightning_labels
        )

        encoder_loss = global_mutual_loss + local_mutual_loss + shared_loss

        total_loss = (
            global_mutual_loss
            + local_mutual_loss
            + shared_loss
            + cat_classif_loss
            + elevation_classif_loss
            + lightning_classif_loss
        )

        return SDIMSmallNORBLosses(
            total_loss=total_loss,
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            shared_loss=shared_loss,
            cat_classif_loss=cat_classif_loss,
            elevation_classif_loss=elevation_classif_loss,
            lightning_classif_loss=lightning_classif_loss,
            cat_accuracy=cat_accuracy,
            elevation_accuracy=elevation_accuracy,
            lightning_accuracy=lightning_accuracy,
        )
