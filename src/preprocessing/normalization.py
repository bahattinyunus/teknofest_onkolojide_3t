"""
Normalizasyon Modülü — MRI görüntüleri için normalizasyon yöntemleri.

Desteklenen yöntemler:
    - Z-score normalizasyon (beyin maskesi içi)
    - Percentile kırpma + min-max ölçekleme
    - WhiteStripe (FLAIR/T2 için)
    - Min-Max normalizasyon
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def z_score_normalize(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalizasyon — MRI görüntüsünü (μ=0, σ=1) olacak şekilde normalize et.

    Beyin maskesi verilirse yalnızca beyin bölgesi içindeki istatistikler kullanılır.
    Bu yaklaşım BraTS veri setlerinde standart kabul gören yöntemdir.

    Args:
        volume : (H, W, D) MRI hacmi
        mask   : İsteğe bağlı (H, W, D) boolean beyin maskesi
        eps    : Sıfıra bölme önleme sabiti

    Returns:
        Normalize edilmiş (H, W, D) ndarray
    """
    volume = volume.astype(np.float32)

    if mask is not None:
        brain_voxels = volume[mask > 0]
    else:
        # Sıfır olmayan vokseller üzerinde hesapla
        brain_voxels = volume[volume > 0]

    if len(brain_voxels) == 0:
        logger.warning("Beyin maskesi boş — global normalize ediliyor")
        brain_voxels = volume.flatten()

    mu = brain_voxels.mean()
    sigma = brain_voxels.std()

    normalized = (volume - mu) / (sigma + eps)
    logger.debug(f"Z-score: μ={mu:.4f}, σ={sigma:.4f}")
    return normalized


def percentile_normalize(
    volume: np.ndarray,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Percentile kırpma ve [0, 1] ölçekleme.

    Aşırı değerlerin (artifact) etkisini azaltmak için
    alt ve üst percentile değerleri ile kırpma uygulanır.

    Args:
        volume    : (H, W, D) MRI hacmi
        lower_pct : Alt percentile (0-100)
        upper_pct : Üst percentile (0-100)
        mask      : İsteğe bağlı beyin maskesi

    Returns:
        [0, 1] aralığında normalize edilmiş ndarray
    """
    volume = volume.astype(np.float32)

    if mask is not None:
        voxels = volume[mask > 0]
    else:
        voxels = volume[volume > 0]

    p_low = np.percentile(voxels, lower_pct)
    p_high = np.percentile(voxels, upper_pct)

    clipped = np.clip(volume, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
    logger.debug(
        f"Percentile normalize: [{p_low:.2f}, {p_high:.2f}] → [0, 1]"
    )
    return normalized.astype(np.float32)


def min_max_normalize(
    volume: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Min-Max normalizasyon.

    Args:
        volume        : (H, W, D) MRI hacmi
        feature_range : Hedef değer aralığı
        eps           : Sıfıra bölme önleme sabiti

    Returns:
        Normalize edilmiş ndarray
    """
    volume = volume.astype(np.float32)
    v_min, v_max = volume.min(), volume.max()
    scaled = (volume - v_min) / (v_max - v_min + eps)

    a, b = feature_range
    normalized = scaled * (b - a) + a
    return normalized.astype(np.float32)


def compute_brain_mask(
    volume: np.ndarray,
    threshold: float = 0.0,
    fill_holes: bool = True,
    dilation_iters: int = 2,
) -> np.ndarray:
    """
    Eşikleme tabanlı basit beyin maskesi hesaplama.

    Args:
        volume        : (H, W, D) MRI hacmi (tercihen FLAIR veya T1ce)
        threshold     : Eşik değeri (default: 0 — sıfır olmayan vokseller)
        fill_holes    : Delik doldurma uygula
        dilation_iters: Dilation iterasyon sayısı

    Returns:
        (H, W, D) binary beyin maskesi
    """
    mask = (volume > threshold).astype(np.uint8)

    if fill_holes:
        # Her eksen boyunca doldur
        for axis in range(3):
            mask = ndimage.binary_fill_holes(mask, structure=None).astype(np.uint8)

    if dilation_iters > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_dilation(
            mask, structure=struct, iterations=dilation_iters
        ).astype(np.uint8)

    return mask


def normalize_multimodal(
    modalities: dict,
    method: str = "zscore",
    mask_key: Optional[str] = "flair",
) -> dict:
    """
    Çok modaliteli MRI verisi için toplu normalizasyon.

    Args:
        modalities : {"t1": arr, "t1ce": arr, "t2": arr, "flair": arr}
        method     : "zscore" | "percentile" | "minmax"
        mask_key   : Beyin maskesi için kullanılacak modalite

    Returns:
        Normalize edilmiş modalite dict'i
    """
    # Beyin maskesini hesapla
    brain_mask = None
    if mask_key and mask_key in modalities:
        brain_mask = compute_brain_mask(modalities[mask_key])

    normalized = {}
    for key, vol in modalities.items():
        if not isinstance(vol, np.ndarray):
            normalized[key] = vol
            continue

        if method == "zscore":
            normalized[key] = z_score_normalize(vol, mask=brain_mask)
        elif method == "percentile":
            normalized[key] = percentile_normalize(vol, mask=brain_mask)
        elif method == "minmax":
            normalized[key] = min_max_normalize(vol)
        else:
            raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")

    return normalized
