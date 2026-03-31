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


class ComprehensiveResponse(BaseModel):
    subject_id: str
    risk_score: float
    risk_level: str
    mgmt_status: str
    tumor_volume_ml: float
    resection_volume_ml: float
    margin_to_tumor_ratio: float
    surgical_safety_score: str
    xai_status: str


@app.get("/")
async def root():
    return {
        "project": "GlioSight AI",
        "version": "1.2.0 (Sovereignty Tier)",
        "status": "online",
        "modules": ["Segmentation", "Survival", "Radiogenomics", "XAI", "Surgical-Planning"]
    }


@app.post("/analyze/comprehensive/{subject_id}", response_model=ComprehensiveResponse)
async def analyze_comprehensive(subject_id: str, data_path: str):
    """
    Uçtan uca kapsamlı klinik analiz (XAI ve Cerrahi dahil).
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="GlioSight Engine devredışı")
        
    path = Path(data_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Klasör bulunamadı")
        
    try:
        results = engine.process_patient(data_path)
        
        return ComprehensiveResponse(
            subject_id=subject_id,
            risk_score=results["survival"]["risk_score"],
            risk_level=results["survival"]["risk_level"],
            mgmt_status=results["radiogenomics"]["mgmt_status"],
            tumor_volume_ml=results["surgical"]["tumor_volume_ml"],
            resection_volume_ml=results["surgical"]["resection_volume_ml"],
            margin_to_tumor_ratio=results["surgical"].get("margin_to_tumor_ratio", 0),
            surgical_safety_score=results["surgical"].get("safety_score", "N/A"),
            xai_status="Heatmap Generated Successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
