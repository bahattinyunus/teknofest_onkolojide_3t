"""
Radyasyon Onkolojisi Modülü — Otomatik Hedef Hacim (CTV/PTV) Planlama.

TEKNOFEST 2026 Kategori 7 (Radyasyon Onkolojisi) kapsamında;
tümör maskesinden (GTV) başlayarak klinik (CTV) ve planlama (PTV) 
hedef hacimlerini otomatik olarak oluşturur.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


def generate_target_volumes(
    gtv_mask: np.ndarray,
    ctv_margin_mm: float = 20.0,
    ptv_margin_mm: float = 3.0,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    GTV'den CTV ve PTV hacimlerini oluştur.
    
    Args:
        gtv_mask      : (H, W, D) Brüt Tümör Hacmi (Gross Tumor Volume)
        ctv_margin_mm : Klinik marjin (Genişleme - mm)
        ptv_margin_mm : Planlama marjini (Setup - mm)
        voxel_spacing : Voksel boyutları
        
    Returns:
        CTV/PTV maskeleri ve hacim istatistikleri
    """
    if not np.any(gtv_mask):
        return {"error": "GTV (Gross Tumor Volume) bulunamadı."}

    avg_spacing = np.mean(voxel_spacing)
    vx, vy, vz = voxel_spacing
    voxel_vol_ml = vx * vy * vz / 1000.0

    # 1. CTV Oluşturma (Genellikle GBM için 20mm)
    ctv_iterations = int(np.ceil(ctv_margin_mm / avg_spacing))
    ctv_mask = binary_dilation(gtv_mask > 0, iterations=ctv_iterations)
    
    # 2. PTV Oluşturma (Setup marjini, genellikle 3-5mm)
    ptv_iterations = int(np.ceil(ptv_margin_mm / avg_spacing))
    ptv_mask = binary_dilation(ctv_mask, iterations=ptv_iterations)

    return {
        "gtv_stats": {"volume_ml": float((gtv_mask > 0).sum() * voxel_vol_ml)},
        "ctv_stats": {
            "margin_mm": ctv_margin_mm,
            "volume_ml": float(ctv_mask.sum() * voxel_vol_ml)
        },
        "ptv_stats": {
            "margin_mm": ptv_margin_mm,
            "volume_ml": float(ptv_mask.sum() * voxel_vol_ml)
        },
        "ctv_mask": ctv_mask,
        "ptv_mask": ptv_mask,
        "radiation_planning_status": "CTV/PTV Margins Generated (ESTRO Standard)"
    }
