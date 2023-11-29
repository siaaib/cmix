from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from torchvision.transforms.functional import resize
from src.models.base import BaseModel


class CustomNormalization(nn.Module):
    def __init__(self, mean, std):
        super(CustomNormalization, self).__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).cuda()
        self.std = torch.tensor(std).view(1, -1, 1, 1).cuda()

    def forward(self, x):
        # Apply normalization during the forward pass
        return (x - self.mean) / self.std

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean", weight=torch.tensor([1.0, 5.0, 5.0])):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is not None:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, weight=self.weight.cuda()
            )
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss
        
class SpecTr(BaseModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.channels_fc = nn.Linear(feature_extractor.out_chans, 1)
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0, 5.0, 5.0]).cuda())
        self.normalize = CustomNormalization(mean, std)

    def _forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)
        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)
        x = x.squeeze(1)
        logits = self.decoder(x)  # (batch_size, n_classes, n_timesteps)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)[:, :, [1, 2]]

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [1, 2]]
