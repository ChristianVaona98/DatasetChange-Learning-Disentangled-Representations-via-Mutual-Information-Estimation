import torch.nn as nn
import torch
from losses.loss_functions import (
    DJSLoss,
    ClassifLoss,
    DiscriminatorLoss,
    GeneratorLoss,
)
from utils.custom_typing import (
    DiscriminatorOutputs,
    EDIMClassifierOutputs,
    EDIMSmallNORBOutputs,
    GenLosses,
    DiscrLosses,
    EDIMClassifLossesSlim,
)


class EDIMLoss(nn.Module):
    """Loss function to extract exclusive information from the image, see paper equation (8)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        disentangling_loss_coeff (float): Coefficient of the Gan loss
    """

    def __init__(
        self,
        local_mutual_loss_coeff: float,
        global_mutual_loss_coeff: float,
        disentangling_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.disentangling_loss_coeff = disentangling_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

    def compute_generator_loss(self, edim_outputs: EDIMSmallNORBOutputs) -> GenLosses:
        """Generator loss function

        Args:
            edim_outputs (EDIMOutputs): Output of the forward pass of the exclusive information model

        Returns:
            GenLosses: Generator losses
        """

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_x,
            T_prime=edim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_y,
            T_prime=edim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_x,
            T_prime=edim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_y,
            T_prime=edim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        gan_loss_x_g = self.generator_loss(fake_logits=edim_outputs.fake_x)
        gan_loss_y_g = self.generator_loss(fake_logits=edim_outputs.fake_y)

        gan_loss_g = (gan_loss_x_g + gan_loss_y_g) * self.disentangling_loss_coeff

        # Get classification error

        # For each network, we assign a loss objective
        encoder_loss = global_mutual_loss + local_mutual_loss + gan_loss_g

        return GenLosses(
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            gan_loss_g=gan_loss_g,
        )

    def compute_discriminator_loss(
        self, discr_outputs: DiscriminatorOutputs
    ) -> DiscrLosses:
        """Discriminator loss see paper equation (9)

        Args:
            discr_outputs (DiscriminatorOutputs): Output of the forward pass of the discriminators model

        Returns:
            DiscrLosses: Discriminator losses
        """
        gan_loss_x_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_x_prime,
            fake_logits=discr_outputs.disentangling_information_x,
        )
        gan_loss_y_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_y_prime,
            fake_logits=discr_outputs.disentangling_information_y,
        )

        gan_loss_d = (gan_loss_x_d + gan_loss_y_d) * self.disentangling_loss_coeff

        return DiscrLosses(gan_loss_d=gan_loss_d)

    def compute_classif_loss(
        self,
        edim_classif_outputs: EDIMClassifierOutputs,
        left_cat_labels: torch.Tensor,
        right_cat_labels: torch.Tensor,
        elevation_labels: torch.Tensor,
        lightning_labels: torch.Tensor,
    ) -> EDIMClassifLossesSlim:
        """Compute classifiers losses. The accuracy of the classifiers allow to quantify the representations level of disentanglement.

        Args:
            classif_outputs (ClassifierOutputs): Classifiers Outputs
            digit_labels (torch.Tensor): Label of the digit
            color_bg_labels (torch.Tensor): Background color of the images
            color_fg_labels (torch.Tensor): Foreground color of the images

        Returns:
            EDIMClassifLosses: Classifiers losses
        """

        left_cat_classif_loss, left_cat_accuracy = self.classif_loss(
            y_pred=edim_classif_outputs.left_cat_logits, target=left_cat_labels
        )
        elevation_classif_loss, elevation_accuracy = self.classif_loss(
            y_pred=edim_classif_outputs.elevation_logits, target=elevation_labels
        )
        lightning_classif_loss, lightning_accuracy = self.classif_loss(
            y_pred=edim_classif_outputs.lightning_logits, target=lightning_labels
        )

        right_cat_classif_loss, right_cat_accuracy = self.classif_loss(
            y_pred=edim_classif_outputs.right_cat_logits, target=right_cat_labels
        )

        classif_loss = (
            left_cat_classif_loss
            + elevation_classif_loss
            + lightning_classif_loss
            + right_cat_classif_loss
        )
        
        return EDIMClassifLossesSlim(
            classif_loss=classif_loss,

            cat_classif_loss = (left_cat_classif_loss + right_cat_classif_loss) / 2,
            elevation_classif_loss = elevation_classif_loss,
            lightning_classif_loss = lightning_classif_loss,
            cat_accuracy = (left_cat_accuracy + right_cat_accuracy) / 2,
            elevation_accuracy = elevation_accuracy,
            lightning_accuracy = lightning_accuracy
        )
