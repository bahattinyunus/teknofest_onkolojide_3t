"""
GlioSight API — Hastane (PACS) ve Klinik Karar Destek Entegrasyon Servisi.

TEKNOFEST 2026 Onkolojide 3T - Biyomedikal Cihaz ve Uygulanabilirlik 
kategorisi için geliştirilmiş REST API servisidir.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from ..inference import SegmentationPipeline, SurvivalPipeline
from ..inference.radiogenomic_pipeline import RadiogenomicPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GlioSight AI — Onkoloji Karar Destek API",
    description="Multimodal MRI Segmentasyon ve Sağkalım Tahmin Servisi",
    version="1.0.0"
)

# Servis başlatıldığında modelleri yükle (Demo için mock)
# Gerçekte config.yaml üzerinden yüklenir.
try:
    from ..main import GlioSightEngine
    engine = GlioSightEngine()
except Exception as e:
    logger.error(f"Engine başlatılamadı: {e}")
    engine = None


class PatientResponse(BaseModel):
    subject_id: str
    risk_score: float
    risk_level: str
    mgmt_status: str
    mgmt_probability: float
    wt_volume_ml: float


@app.get("/")
async def root():
    return {"message": "GlioSight API Çalışıyor", "status": "online"}


@app.post("/analyze/{subject_id}", response_model=PatientResponse)
async def analyze_patient(subject_id: str, data_path: str):
    """
    Belirtilen hasta klasörünü analiz et ve sonuçları döndür.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="GlioSight Engine devredışı")
        
    path = Path(data_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Klasör bulunamadı")
        
    try:
        # Mevcut motoru çalıştır (main.py'deki engine)
        results = engine.process_patient(data_path)
        
        # Radyogenomik tahmin (MGMT) ekle (Eğer main.py'ye eklenmediyse)
        # Şimdilik ana akışa ekleyeceğiz.
        
        return PatientResponse(
            subject_id=subject_id,
            risk_score=results["survival"]["risk_score"],
            risk_level=results["survival"]["risk_level"],
            mgmt_status=results.get("radiogenomics", {}).get("mgmt_status", "Tahmin Edilemedi"),
            mgmt_probability=results.get("radiogenomics", {}).get("mgmt_probability", 0.5),
            wt_volume_ml=results["survival"]["features"]["wt_volume_ml"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
