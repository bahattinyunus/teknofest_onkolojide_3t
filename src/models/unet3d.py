"""
3D U-Net — BraTS Beyin Tümörü Segmentasyonu için 3D U-Net Mimarisi.

Mimari Özellikleri:
    - Encoder-Decoder yapısı (skip connections ile)
    - 4 aşamalı kodlayıcı (3D convolution + InstanceNorm + LeakyReLU)
    - 4 aşamalı çözücü (bilinear upsample + concatenate + conv)
    - Son katman: 3-sınıf segmentasyon çıkışı (WT/TC/ET)
    - Kanal sayısı: 4 giriş modalite → [32, 64, 128, 256, 320]

Referans Mimariler:
    - Çınar & Yıldırım (2022) — 3D U-Net for BraTS
    - nnU-Net (Isensee et al., 2021) — Self-configuring baseline
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Temel 3D Konvolüsyon Bloğu.

    Yapı: [Conv3D → InstanceNorm → LeakyReLU] × 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Kodlayıcı Bloğu: ConvBlock + MaxPool3D (stride ile downsample).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip: Skip connection tensörü (conv çıkışı)
            down: Pooling sonrası tensör
        """
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    Çözücü Bloğu: ConvTranspose3D (upsample) + Skip Concatenation + ConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(
            out_channels + skip_channels, out_channels, dropout_p=dropout_p
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)

        # Boyut uyuşmazlığı düzeltme (padding)
        if x.shape != skip.shape:
            x = F.pad(
                x,
                [
                    0, skip.shape[4] - x.shape[4],
                    0, skip.shape[3] - x.shape[3],
                    0, skip.shape[2] - x.shape[2],
                ],
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net — BraTS beyin tümörü segmentasyonu.

    Args:
        in_channels     : Giriş kanal sayısı (BraTS: 4 — T1, T1ce, T2, FLAIR)
        out_channels    : Çıkış sınıf sayısı (BraTS binary: 3 — WT, TC, ET)
        base_features   : İlk katmandaki filtre sayısı
        dropout_p       : Dropout olasılığı
        deep_supervision: Derin denetleme katmanları ekle
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_features: int = 32,
        dropout_p: float = 0.1,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = base_features
        features = [f, f*2, f*4, f*8, f*16]  # [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = EncoderBlock(in_channels, features[0], dropout_p=0.0)
        self.enc2 = EncoderBlock(features[0], features[1], dropout_p=0.0)
        self.enc3 = EncoderBlock(features[1], features[2], dropout_p=dropout_p)
        self.enc4 = EncoderBlock(features[2], features[3], dropout_p=dropout_p)

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[4], dropout_p=dropout_p)

        # Decoder
        self.dec4 = DecoderBlock(features[4], features[3], features[3], dropout_p=dropout_p)
        self.dec3 = DecoderBlock(features[3], features[2], features[2], dropout_p=dropout_p)
        self.dec2 = DecoderBlock(features[2], features[1], features[1], dropout_p=dropout_p)
        self.dec1 = DecoderBlock(features[1], features[0], features[0], dropout_p=0.0)

        # Çıkış katmanı
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # Derin denetleme çıkışları
        if deep_supervision:
            self.ds3 = nn.Conv3d(features[2], out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(features[1], out_channels, kernel_size=1)

        self._init_weights()
        logger.info(
            f"UNet3D başlatıldı | in={in_channels}, out={out_channels}, "
            f"base_features={f}, params={self._count_params():,}"
        )

    def _init_weights(self):
        """Kaiming normal başlatma."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Args:
            x : (B, C, H, W, D) — B: batch, C: modalite sayısı

        Returns:
            Segmentasyon logit tensörü (B, num_classes, H, W, D)
            deep_supervision=True ise: List[Tensor] (büyükten küçüğe)
        """
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)

        if self.deep_supervision:
            ds3_out = self.ds3(x)

        x = self.dec2(x, skip2)

        if self.deep_supervision:
            ds2_out = self.ds2(x)

        x = self.dec1(x, skip1)
        out = self.out_conv(x)

        if self.deep_supervision:
            return out, ds3_out, ds2_out

        return out


def build_unet3d(config: dict) -> UNet3D:
    """
    Konfigürasyon dict'inden UNet3D oluştur.

    Args:
        config: {
            "in_channels": 4,
            "out_channels": 3,
            "base_features": 32,
            "dropout_p": 0.1,
            "deep_supervision": false
        }

    Returns:
        UNet3D modeli
    """
    return UNet3D(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 3),
        base_features=config.get("base_features", 32),
        dropout_p=config.get("dropout_p", 0.1),
        deep_supervision=config.get("deep_supervision", False),
    )
