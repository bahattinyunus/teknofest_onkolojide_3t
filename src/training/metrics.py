"""
Değerlendirme Metrikleri — BraTS Segmentasyon ve Sürviyal Metrikleri.

Segmentasyon Metrikleri:
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance 95th percentile (HD95)
    - Intersection over Union (IoU / Jaccard)
    - Sensitivity (Recall) ve Specificity
    - Positive Predictive Value (PPV / Precision)

Sürviyal Metrikleri:
    - Concordance Index (C-index / Harrell's C)
    - Integrated Brier Score (IBS)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Segmentasyon Metrikleri ─────────────────────────────────────────────────

def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Dice Benzerlik Katsayısı (DSC).

    DSC = 2 * |P ∩ T| / (|P| + |T|)

    Args:
        pred   : (H, W, D) binary tahmin maskesi
        target : (H, W, D) binary hedef maske
        smooth : Sıfıra bölme önleme

    Returns:
        Dice skoru [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0  # Her iki maske de boş → mükemmel uyum

    return float(2.0 * intersection / (union + smooth))


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Intersection over Union (Jaccard Index).

    IoU = |P ∩ T| / |P ∪ T|

    Args:
        pred   : Binary tahmin maskesi
        target : Binary hedef maske

    Returns:
        IoU skoru [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = (pred & target).sum()
    union = (pred | target).sum()

    if union == 0:
        return 1.0

    return float(intersection / (union + smooth))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    95. Yüzdelik Hausdorff Mesafesi (HD95).

    BraTS resmi metriği. Her iki yönden (pred→target, target→pred)
    yüzey mesafeleri hesaplanır, 95. yüzdelik alınır.

    Args:
        pred          : Binary tahmin maskesi
        target        : Binary hedef maske
        voxel_spacing : (x, y, z) voksel boyutu (mm)

    Returns:
        HD95 (mm). Boş maske durumunda inf döner.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        logger.error("scipy gerekli — pip install scipy")
        return float("inf")

    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if not pred_bool.any() or not target_bool.any():
        return float("inf")

    # Kenar vokselleri bul
    def surface_voxels(mask):
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        return mask & ~eroded

    pred_surf = surface_voxels(pred_bool)
    target_surf = surface_voxels(target_bool)

    # Mesafe dönüşümleri
    dt_pred = distance_transform_edt(~pred_bool, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target_bool, sampling=voxel_spacing)

    # Yüzey noktalarından mesafeler
    dist_pred_to_target = dt_target[pred_surf]
    dist_target_to_pred = dt_pred[target_surf]

    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return float(np.percentile(all_distances, 95))


def sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Duyarlılık (Recall): TP / (TP + FN)."""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = (pred & target).sum()
    fn = (~pred & target).sum()
    return float(tp / (tp + fn + smooth))


def specificity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Özgüllük: TN / (TN + FP)."""
    pred, target = pred.astype(bool), target.astype(bool)
    tn = (~pred & ~target).sum()
    fp = (pred & ~target).sum()
    return float(tn / (tn + fp + smooth))


def precision(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Kesinlik (PPV): TP / (TP + FP)."""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    return float(tp / (tp + fp + smooth))


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    regions: Optional[List[str]] = None,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    compute_hd95: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    BraTS tümör bölgesi metrikleri hesapla.

    Args:
        pred          : (H, W, D) segmentasyon tahmini (BraTS etiketleri: 0,1,2,4)
        target        : (H, W, D) ground truth segmentasyon
        regions       : Hesaplanacak bölgeler (["WT", "TC", "ET"])
        voxel_spacing : (x, y, z) voksel boyutu (mm)
        compute_hd95  : HD95 hesaplansın mı (yavaş olabilir)

    Returns:
        {
            "WT": {"dice": 0.91, "iou": 0.84, "hd95": 3.2, ...},
            "TC": {...},
            "ET": {...},
        }
    """
    if regions is None:
        regions = ["WT", "TC", "ET"]

    # BraTS bölge maskelerini hesapla
    region_masks_pred = {
        "WT": np.isin(pred, [1, 2, 4]),
        "TC": np.isin(pred, [1, 4]),
        "ET": pred == 4,
    }
    region_masks_target = {
        "WT": np.isin(target, [1, 2, 4]),
        "TC": np.isin(target, [1, 4]),
        "ET": target == 4,
    }

    results = {}
    for region in regions:
        p = region_masks_pred[region]
        t = region_masks_target[region]

        metrics = {
            "dice": dice_coefficient(p, t),
            "iou": iou_score(p, t),
            "sensitivity": sensitivity(p, t),
            "specificity": specificity(p, t),
            "precision": precision(p, t),
        }

        if compute_hd95:
            metrics["hd95"] = hausdorff_distance_95(p, t, voxel_spacing)

        results[region] = metrics

    return results


class MetricTracker:
    """Eğitim süreci boyunca metrikleri takip et ve özetle."""

    def __init__(self):
        self._history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._history.setdefault(k, []).append(v)

    def mean(self, key: str) -> float:
        vals = self._history.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self._history}

    def reset(self) -> None:
        self._history.clear()

    def __str__(self) -> str:
        s = self.summary()
        return " | ".join(f"{k}: {v:.4f}" for k, v in s.items())
