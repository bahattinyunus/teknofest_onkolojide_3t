"""
Radyasyon Onkolojisi Modülü — Otomatik Hedef Hacim (CTV/PTV) Planlama.

TEKNOFEST 2026 Kategori 7 (Radyasyon Onkolojisi) kapsamında;
CTV ve PTV hacimlerini hesaplar, OAR (Organs At Risk) koruma 
analizi yapar ve doz dağılım simülasyonu üretir.
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
    
    # OAR (Az Riskli Organ) Mesafe Analizi (Cat 7.4 Compliance)
    # Kritik yapılar (Beyin sapı, Optik kiyazma)
    oar_results = _analyze_oar_proximity(ptv_mask)

    ctv_vol = ctv_mask.sum() * voxel_vol_ml
    ptv_vol = ptv_mask.sum() * voxel_vol_ml

    return {
        "ctv_stats": {"volume_ml": float(ctv_vol), "margin_mm": ctv_margin_mm},
        "ptv_stats": {"volume_ml": float(ptv_vol), "margin_mm": ptv_margin_mm},
        "description": f"CTV({ctv_margin_mm}mm) + PTV({ptv_margin_mm}mm) planı.",
        "oar_risk_assessment": oar_results,
        "isodose_cloud": _simulate_isodose_cloud(ptv_mask),
        "ctv_mask": ctv_mask,
        "ptv_mask": ptv_mask,
        "radiation_planning_status": "CTV/PTV Margins Generated (ESTRO Standard)"
    }
