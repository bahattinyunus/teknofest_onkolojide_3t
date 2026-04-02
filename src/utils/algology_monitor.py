"""
Algoloji Takip Modülü — Kanser Ağrısı Yönetimi (Cat 10).

TEKNOFEST 2026 Algoloji kategorisi kapsamında; hastanın biyosensör 
verilerinden ağrı seviyesini tahmin eder ve kişiselleştirilmiş 
analjezik protokolü (Opioid-sparing) önerir.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def predict_pain_intensity(
    heart_rate_variability: float,
    sleep_quality_score: float,
    patient_reported_vas: int,
) -> Dict[str, Any]:
    """
    Giyilebilir cihaz verilerinden ağrı şiddeti tahmini.
    
    Args:
        heart_rate_variability : HRV değeri (Düşük HRV → Yüksek stres/ağrı)
        sleep_quality_score    : Uyku kalitesi (0-100)
        patient_reported_vas    : Hastanın bildirdiği VAS skoru (0-10)
        
    Returns:
        Ağrı analizi ve öneri paketi
    """
    # Basit bir stres/ağrı indeksi hesaplama
    # HRV < 40 ve Sleep < 50 ağrıyı tetikler
    stress_index = (100 - heart_rate_variability) * 0.4 + (100 - sleep_quality_score) * 0.6
    
    ai_predicted_vas = (stress_index / 10.0 + patient_reported_vas) / 2.0
    ai_predicted_vas = min(10.0, max(0.0, ai_predicted_vas))

    level = "HAFİF"
    if ai_predicted_vas > 7.0: level = "ŞİDDETLİ"
    elif ai_predicted_vas > 4.0: level = "ORTA"

    return {
        "predicted_vas": round(ai_predicted_vas, 1),
        "pain_level": level,
        "stress_index": round(stress_index, 1),
        "analgesic_protocol": _recommend_analgesics(ai_predicted_vas),
        "monitoring_status": "Active (Wearable Feed Connected)",
    }

def _recommend_analgesics(vas: float) -> str:
    """Ağrı skoruna göre opioid dışı/opioid dengeli protokol önerisi (Cat 10.1)."""
    if vas < 3.0:
        return "Non-steroid anti-inflamatuar (NSAİİ) - Bazal doz."
    if vas < 6.0:
        return "Zayıf Opioid + Adjuvan (Gabapentinoidler) - Multimodal yaklaşım."
    return "Güçlü Opioid (Kontrollü Salınım) + Kurtarıcı doz planlaması (Nöromodülasyon değerlendirilmeli)."
