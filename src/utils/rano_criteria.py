"""
RANO (Response Assessment in Neuro-Oncology) Kriterleri Modülü.

Bu modül, beyin tümörleri için standart RANO kriterlerini (2D/3D hacimsel 
karşılaştırma bazlı) kullanarak tedavi yanıtını sınıflandırır.

Kategoriler:
- CR: Complete Response (Tam Yanıt)
- PR: Partial Response (Kısmi Yanıt)
- SD: Stable Disease (Stabil Hastalık)
- PD: Progressive Disease (Progresif Hastalık)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def evaluate_rano_response(
    baseline_volume_ml: float,
    current_volume_ml: float,
    new_lesion: bool = False,
    steroid_increase: bool = False,
) -> Dict[str, Any]:
    """
    RANO kriterlerine göre tedavi yanıtını değerlendirir.
    
    Args:
        baseline_volume_ml : Başlangıç (Pre-treatment) tümör hacmi
        current_volume_ml : Güncel tümör hacmi
        new_lesion        : Yeni lezyon oluşumu var mı?
        steroid_increase  : Steroid dozunda artış var mı?
        
    Returns:
        RANO değerlendirme sonuçları
    """
    # Hacimsel değişim oranı
    if baseline_volume_ml > 0:
        change_pct = (current_volume_ml - baseline_volume_ml) / baseline_volume_ml
    else:
        change_pct = 0.0

    # RANO Mantığı (Volume-based approximation)
    # Standart RANO aslında 2D ölçüm (product of diameters) kullanır (-50% PR, +25% PD).
    # Hacimsel olarak bunlar yaklaşık -65% (PR) ve +40% (PD) değerlerine tekabül eder.

    if new_lesion:
        response = "PD (Progressive Disease)"
        remark = "Yeni lezyon tespit edildi."
    elif change_pct >= 0.40:
        response = "PD (Progressive Disease)"
        remark = f"Tümör hacminde %{change_pct*100:.1f} artış (Limit: +%40)."
    elif change_pct <= -0.99:
        response = "CR (Complete Response)"
        remark = "Tümör tamamen kayboldu."
    elif change_pct <= -0.65:
        response = "PR (Partial Response)"
        remark = f"Tümör hacminde %{abs(change_pct)*100:.1f} anlamlı azalma."
    else:
        response = "SD (Stable Disease)"
        remark = "Tümör hacmi stabil sınırda."

    return {
        "response_category": response,
        "volume_change_pct": float(change_pct),
        "clinical_remark": remark,
        "compliance_standard": "RANO 2010 / 2023 Update",
        "description": "Hacimsel radyolojik yanıt değerlendirmesi."
    }
