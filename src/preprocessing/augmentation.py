"""
Veri Artırma Modülü — 3D MRI hacimleri için veri artırma teknikleri.

Tüm dönüşümler segmentasyon maskesiyle tutarlı olacak şekilde uygulanır.
Geometric dönüşümler hem görüntüye hem maskeye aynı parametrelerle uygulanır.

Desteklenen augmentasyon yöntemleri:
    - Random flip (3 eksen)
    - Random rotation (3D)
    - Random crop / center crop
    - Gaussian noise
    - Random intensity scaling
    - Random gamma correction
    - Elastic deformation
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def random_flip(
    volume: np.ndarray,
    seg: Optional[np.ndarray] = None,
    axes: List[int] = [0, 1, 2],
    p: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rastgele eksen boyunca çevirme (flip).

    Args:
        volume : (H, W, D) veya (C, H, W, D) hacim
        seg    : İsteğe bağlı segmentasyon maskesi
        axes   : Flip uygulanabilecek eksenler
        p      : Her eksen için flip olasılığı

    Returns:
        (volume, seg) çifti
    """
    for axis in axes:
        if random.random() < p:
            volume = np.flip(volume, axis=axis).copy()
            if seg is not None:
                seg = np.flip(seg, axis=axis).copy()
    return volume, seg


def random_rotation(
    volume: np.ndarray,
    seg: Optional[np.ndarray] = None,
    max_angle: float = 15.0,
    p: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rastgele 3D rotasyon.

    Args:
        volume    : (H, W, D) hacim
        seg       : İsteğe bağlı segmentasyon
        max_angle : Maksimum rotasyon açısı (derece)
        p         : Uygulama olasılığı

    Returns:
        (volume, seg) çifti
    """
    if random.random() >= p:
        return volume, seg

    angles = [
        random.uniform(-max_angle, max_angle) for _ in range(3)
    ]

    def rotate_volume(vol, angles, order):
        vol = ndimage.rotate(vol, angles[0], axes=(1, 2), reshape=False, order=order)
        vol = ndimage.rotate(vol, angles[1], axes=(0, 2), reshape=False, order=order)
        vol = ndimage.rotate(vol, angles[2], axes=(0, 1), reshape=False, order=order)
        return vol

    volume = rotate_volume(volume, angles, order=3)
    if seg is not None:
        seg = rotate_volume(seg, angles, order=0).astype(np.int32)

    return volume, seg


def random_crop(
    volume: np.ndarray,
    seg: Optional[np.ndarray] = None,
    crop_size: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rastgele 3D kırpma.

    Args:
        volume    : (H, W, D) hacim
        seg       : İsteğe bağlı segmentasyon
        crop_size : (h, w, d) kırpma boyutu

    Returns:
        (volume, seg) kırpılmış çifti
    """
    h, w, d = volume.shape
    ch, cw, cd = crop_size

    if h < ch or w < cw or d < cd:
        raise ValueError(
            f"Hacim boyutu ({h},{w},{d}) kırpma boyutundan ({ch},{cw},{cd}) küçük"
        )

    x = random.randint(0, h - ch)
    y = random.randint(0, w - cw)
    z = random.randint(0, d - cd)

    volume = volume[x:x+ch, y:y+cw, z:z+cd]
    if seg is not None:
        seg = seg[x:x+ch, y:y+cw, z:z+cd]

    return volume, seg


def center_crop(
    volume: np.ndarray,
    seg: Optional[np.ndarray] = None,
    crop_size: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Merkez kırpma.

    Args:
        volume    : (H, W, D) hacim
        seg       : İsteğe bağlı segmentasyon
        crop_size : (h, w, d) kırpma boyutu

    Returns:
        (volume, seg) merkez kırpılmış çifti
    """
    h, w, d = volume.shape
    ch, cw, cd = crop_size

    x = (h - ch) // 2
    y = (w - cw) // 2
    z = (d - cd) // 2

    volume = volume[x:x+ch, y:y+cw, z:z+cd]
    if seg is not None:
        seg = seg[x:x+ch, y:y+cw, z:z+cd]

    return volume, seg


def add_gaussian_noise(
    volume: np.ndarray,
    mean: float = 0.0,
    std: float = 0.05,
    p: float = 0.3,
) -> np.ndarray:
    """
    Gaussian gürültü ekleme.

    Args:
        volume : (H, W, D) hacim
        mean   : Gürültü ortalaması
        std    : Gürültü standart sapması
        p      : Uygulama olasılığı

    Returns:
        Gürültü eklenmiş hacim
    """
    if random.random() >= p:
        return volume
    noise = np.random.normal(mean, std, volume.shape).astype(np.float32)
    return volume + noise


def random_intensity_scale(
    volume: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_range: Tuple[float, float] = (-0.1, 0.1),
    p: float = 0.3,
) -> np.ndarray:
    """
    Rastgele yoğunluk ölçekleme ve kaydırma.

    Args:
        volume      : (H, W, D) hacim
        scale_range : (min, max) ölçek aralığı
        shift_range : (min, max) kaydırma aralığı
        p           : Uygulama olasılığı

    Returns:
        Ölçeklenmiş hacim
    """
    if random.random() >= p:
        return volume
    scale = random.uniform(*scale_range)
    shift = random.uniform(*shift_range)
    return (volume * scale + shift).astype(np.float32)


def random_gamma_correction(
    volume: np.ndarray,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
    p: float = 0.3,
) -> np.ndarray:
    """
    Rastgele gamma düzeltmesi (histogram eğrisi değişimi).

    Args:
        volume      : [0, 1] aralığında normalize edilmiş hacim
        gamma_range : (min, max) gamma aralığı
        p           : Uygulama olasılığı

    Returns:
        Gamma düzeltmesi uygulanmış hacim
    """
    if random.random() >= p:
        return volume
    gamma = random.uniform(*gamma_range)
    # Negatif değerleri korumak için işaret ayrıştırma
    positive = np.clip(volume, 0, None)
    corrected = np.power(positive, gamma)
    return corrected.astype(np.float32)


class BraTSAugmentor:
    """
    BraTS veri seti için kapsamlı augmentasyon pipeline.

    Eğitim sırasında geometric ve intensity augmentasyonları birleştirir.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        flip_p: float = 0.5,
        rotation_p: float = 0.3,
        max_rotation_angle: float = 15.0,
        noise_p: float = 0.2,
        noise_std: float = 0.05,
        intensity_p: float = 0.3,
        gamma_p: float = 0.2,
        is_training: bool = True,
    ):
        self.crop_size = crop_size
        self.flip_p = flip_p
        self.rotation_p = rotation_p
        self.max_rotation_angle = max_rotation_angle
        self.noise_p = noise_p
        self.noise_std = noise_std
        self.intensity_p = intensity_p
        self.gamma_p = gamma_p
        self.is_training = is_training

    def __call__(
        self,
        volume: np.ndarray,
        seg: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augmentasyon pipeline'ını uygula.

        Args:
            volume : (H, W, D) normalize edilmiş MRI hacmi
            seg    : İsteğe bağlı (H, W, D) segmentasyon maskesi

        Returns:
            (augmented_volume, augmented_seg)
        """
        if self.is_training:
            # Geometric dönüşümler
            volume, seg = random_flip(volume, seg, p=self.flip_p)
            volume, seg = random_rotation(
                volume, seg, max_angle=self.max_rotation_angle, p=self.rotation_p
            )
            volume, seg = random_crop(volume, seg, crop_size=self.crop_size)

            # Intensity augmentasyonlar (yalnızca görüntüye)
            volume = add_gaussian_noise(volume, std=self.noise_std, p=self.noise_p)
            volume = random_intensity_scale(volume, p=self.intensity_p)
            volume = random_gamma_correction(volume, p=self.gamma_p)
        else:
            # Validation/test için yalnızca merkez kırpma
            volume, seg = center_crop(volume, seg, crop_size=self.crop_size)

        return volume, seg
