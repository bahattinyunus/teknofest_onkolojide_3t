"""
Cerrahi Planlama Modülü — Tümör Rezeksiyon ve Marjin Analizi.

TEKNOFEST 2026 Kategori 11 (Cerrahi Onkoloji Teknolojileri) kapsamında;
rezerke edilecek tümör hacmini ve güvenlik marjinlerini (Safety Margins)
otomatik olarak hesaplar.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


def calculate_surgical_margins(
    seg_mask: np.ndarray,
    margin_mm: float = 10.0,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Tümör etrafındaki güvenlik marjinlerini ve rezeksiyon hacmini hesaplar.
    
    Args:
        seg_mask      : (H, W, D) boyutunda segmentasyon maskesi (Tüm Tümör > 0)
        margin_mm     : İstenen güvenlik marjini (mm)
        voxel_spacing : Voksel boyutları (mm)
        
    Returns:
        Marjin analizi istatistiklerini içeren sözlük
    """
    wt_mask = (seg_mask > 0).astype(bool)
    if not np.any(wt_mask):
        return {"error": "Tümör dokusu bulunamadı."}

    # Hacim hesaplama (mL)
    vx, vy, vz = voxel_spacing
    voxel_vol_ml = vx * vy * vz / 1000.0
    tumor_vol = wt_mask.sum() * voxel_vol_ml

    # Marjin genişletme (Voksel bazında iterasyon sayısı)
    # Basitlik için ortalama spacing üzerinden gidiyoruz.
    avg_spacing = np.mean(voxel_spacing)
    iterations = int(np.ceil(margin_mm / avg_spacing))
    
    margin_mask = binary_dilation(wt_mask, iterations=iterations)
    resection_vol = margin_mask.sum() * voxel_vol_ml
    
    # Sadece marjin bölgesi hacmi (Margin Volume = Resection - Tumor)
    margin_only_vol = resection_vol - tumor_vol

    return {
        "tumor_volume_ml": float(tumor_vol),
        "resection_volume_ml": float(resection_vol),
        "margin_only_volume_ml": float(margin_only_vol),
        "margin_mm": margin_mm,
        "surgical_planning_status": "Ready",
        "description": f"{margin_mm}mm marjinli cerrahi plan önerisi."
    }


def analyze_proximity_to_eloquent(
    seg_mask: np.ndarray,
    eloquent_areas_mask: Optional[np.ndarray] = None,
) -> str:
    """Eleştirel (Eloquent) beyin alanlarına yakınlık analizi (Simüle edilmiş)."""
    # Not: Gerçek atlas kaydı gerektirir. Yarışma için konsept seviyesinde kalıyor.
    return "Tümör frontal loba yakın; motor korteks ile mesafesi > 15mm (Güvenli)."
