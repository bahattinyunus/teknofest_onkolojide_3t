"""
Kayıp Fonksiyonları — BraTS Segmentasyon Eğitimi için Kayıp Fonksiyonları.

Desteklenen Kayıplar:
    - Dice Loss
    - Binary Cross-Entropy (BCE)
    - Focal Loss
    - Dice + Focal (birleşik)
    - Deep Supervision Loss (ağırlıklı toplam)

Referans:
    - Milletari et al. (2016) — V-Net, Dice Loss
    - Lin et al. (2017) — Focal Loss
    - Isensee et al. (2021) — nnU-Net compound loss
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss — çakışma tabanlı segmentasyon kaybı.

    Sınıf dengesizliğine karşı BCE'den daha güçlüdür.
    BraTS gibi küçük tümör bölgelerinde kritik öneme sahiptir.

    Args:
        smooth    : Düzleştirme sabiti (sıfıra bölme önleme)
        reduction : "mean" | "sum" | "none"
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D) — ham model çıkışı (sigmoid öncesi)
            targets : (B, C, H, W, D) — one-hot kodlanmış hedef maske

        Returns:
            Dice kaybı skaleri
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims
        B, C = probs.shape[:2]
        probs_flat = probs.contiguous().view(B, C, -1)
        targets_flat = targets.contiguous().view(B, C, -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score  # (B, C)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss — sınıf dengesizliği için BCE tabanlı ağırlıklı kayıp.

    Zor örneklere (false negatives) daha fazla ağırlık verir.

    Args:
        alpha : Pozitif sınıf ağırlığı (0-1)
        gamma : Odak parametresi (γ=0 → BCE, γ=2 önerilir)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D) ham logitler
            targets : (B, C, H, W, D) binary hedef

        Returns:
            Focal kayıp skaleri
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Bileşik Dice + Focal kayıp (nnU-Net önerisi).

    Loss = λ_dice * DiceLoss + λ_focal * FocalLoss

    Args:
        dice_weight  : Dice kayıp ağırlığı
        focal_weight : Focal kayıp ağırlığı
        smooth       : Dice düzleştirme sabiti
        alpha        : Focal alpha
        gamma        : Focal gamma
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        smooth: float = 1e-5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D)
            targets : (B, C, H, W, D) one-hot maske

        Returns:
            Bileşik kayıp skaleri
        """
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        total = self.dice_weight * d_loss + self.focal_weight * f_loss
        return total


class DeepSupervisionLoss(nn.Module):
    """
    Derin Denetleme Kaybı — çoklu ölçek çıkışlarını birleştirir.

    Her ölçek için ayrı kayıp hesaplanır, ağırlıklı toplamı alınır.
    Düşük ölçek çıkışları (küçük boyutlu) daha az ağırlık alır.

    Args:
        base_loss : Temel kayıp fonksiyonu (DiceFocalLoss önerilir)
        weights   : Her ölçek için ağırlık listesi (büyükten küçüğe)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25]

    def forward(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs : List[Tensor] — büyükten küçüğe ölçek çıkışları
            targets : Ana ölçek hedef maskesi

        Returns:
            Ağırlıklı toplam kayıp
        """
        total_loss = torch.tensor(0.0, device=targets.device)

        for i, (out, w) in enumerate(zip(outputs, self.weights)):
            if out.shape != targets.shape:
                scaled_target = F.interpolate(
                    targets.float(),
                    size=out.shape[2:],
                    mode="nearest",
                )
            else:
                scaled_target = targets.float()

            loss = self.base_loss(out, scaled_target)
            total_loss = total_loss + w * loss

        return total_loss


def build_loss(config: dict) -> nn.Module:
    """
    Konfigürasyondan kayıp fonksiyonu oluştur.

    Args:
        config: {
            "type": "dice_focal",  # "dice" | "focal" | "dice_focal"
            "dice_weight": 1.0,
            "focal_weight": 1.0,
            "deep_supervision": false,
            "ds_weights": [1.0, 0.5, 0.25]
        }

    Returns:
        nn.Module kayıp fonksiyonu
    """
    loss_type = config.get("type", "dice_focal").lower()

    if loss_type == "dice":
        base_loss = DiceLoss()
    elif loss_type == "focal":
        base_loss = FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
        )
    elif loss_type == "dice_focal":
        base_loss = DiceFocalLoss(
            dice_weight=config.get("dice_weight", 1.0),
            focal_weight=config.get("focal_weight", 1.0),
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
        )
    else:
        raise ValueError(f"Bilinmeyen kayıp tipi: {loss_type}")

    if config.get("deep_supervision", False):
        return DeepSupervisionLoss(
            base_loss=base_loss,
            weights=config.get("ds_weights", [1.0, 0.5, 0.25]),
        )

    return base_loss
