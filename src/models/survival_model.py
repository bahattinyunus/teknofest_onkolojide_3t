"""
Hayatta Kalma (Sürviyal) Modeli — Glioblastoma Hastalarında Genel Sağkalım Tahmini.

İki aşamalı yaklaşım:
    1. Radyomik özellik çıkarımı (PyRadiomics) → MRI'dan sayısal özellikler
    2. Cox Proportional Hazards + XGBoost Cox modeli ile sürviyal regresyon

Klinik Hedefler:
    - Genel sağkalım (OS) tahmini (gün cinsinden)
    - Risk katmanlaması (yüksek / orta / düşük risk grubu)
    - Prognostik biyobelirteçlerin belirlenmesi (SHAP)

Referans Çalışmalar:
    - Kickingereder et al., Radiology 2016 — MRI radyomik + OS
    - BraTS-Path 2023 — Segmentasyon + sürviyal entegrasyonu
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kullanılabilir modeller için tip uyarısı
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost yüklü değil — pip install xgboost")

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("lifelines yüklü değil — pip install lifelines")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─── Özellik Mühendisliği ────────────────────────────────────────────────────

def extract_volumetric_features(
    seg: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Segmentasyon maskesinden temel hacimsel özellikler çıkar.

    Args:
        seg           : (H, W, D) segmentasyon maskesi (BraTS etiketleri)
        voxel_spacing : (x, y, z) voksel boyutu (mm)

    Returns:
        {
            "wt_volume_ml"         : Tüm tümör hacmi (mL),
            "tc_volume_ml"         : Tümör çekirdeği hacmi (mL),
            "et_volume_ml"         : Kontrast tutan bölge hacmi (mL),
            "et_tc_ratio"          : ET/TC oran,
            "tc_wt_ratio"          : TC/WT oranı,
            "et_surface_volume_ratio": Yüzey/Hacim oranı (sferisite proxy),
        }
    """
    vx, vy, vz = voxel_spacing
    voxel_vol_ml = vx * vy * vz / 1000.0  # mm³ → mL

    # BraTS etiket bölgeleri
    wt_mask = (seg > 0)          # Whole Tumor: 1+2+4
    tc_mask = np.isin(seg, [1, 4])  # Tumor Core: 1+4
    et_mask = (seg == 4)         # Enhancing Tumor: 4

    wt_vol = float(wt_mask.sum()) * voxel_vol_ml
    tc_vol = float(tc_mask.sum()) * voxel_vol_ml
    et_vol = float(et_mask.sum()) * voxel_vol_ml

    features = {
        "wt_volume_ml": wt_vol,
        "tc_volume_ml": tc_vol,
        "et_volume_ml": et_vol,
        "et_tc_ratio": et_vol / (tc_vol + 1e-6),
        "tc_wt_ratio": tc_vol / (wt_vol + 1e-6),
    }

    logger.debug(f"Hacimsel özellikler: WT={wt_vol:.2f}mL, TC={tc_vol:.2f}mL, ET={et_vol:.2f}mL")
    return features


def extract_intensity_features(
    mri: np.ndarray,
    seg: np.ndarray,
    modality: str = "t1ce",
) -> Dict[str, float]:
    """
    Tümör bölgesinden temel yoğunluk istatistiklerini çıkar.

    Args:
        mri      : (H, W, D) MRI hacmi (normalize edilmiş)
        seg      : (H, W, D) segmentasyon maskesi
        modality : Modalite adı (loglama için)

    Returns:
        Yoğunluk istatistikleri dict'i
    """
    et_voxels = mri[seg == 4]  # Enhancing Tumor vokselleri
    wt_voxels = mri[seg > 0]   # Whole Tumor vokselleri

    def _stats(voxels: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(voxels) == 0:
            return {}
        return {
            f"{prefix}_mean": float(voxels.mean()),
            f"{prefix}_std": float(voxels.std()),
            f"{prefix}_skewness": float(
                np.mean(((voxels - voxels.mean()) / (voxels.std() + 1e-8)) ** 3)
            ),
            f"{prefix}_kurtosis": float(
                np.mean(((voxels - voxels.mean()) / (voxels.std() + 1e-8)) ** 4)
            ),
            f"{prefix}_p25": float(np.percentile(voxels, 25)),
            f"{prefix}_p75": float(np.percentile(voxels, 75)),
            f"{prefix}_iqr": float(
                np.percentile(voxels, 75) - np.percentile(voxels, 25)
            ),
        }

    features = {}
    features.update(_stats(et_voxels, f"{modality}_et"))
    features.update(_stats(wt_voxels, f"{modality}_wt"))
    return features


# ─── Cox PH Modeli ───────────────────────────────────────────────────────────

class CoxSurvivalModel:
    """
    Cox Proportional Hazards sürviyal modeli (lifelines sarıcısı).

    Kullanım:
        model = CoxSurvivalModel()
        model.fit(df, duration_col="os_days", event_col="event")
        risk_scores = model.predict_risk(df_test)
        c_index = model.c_index(df_test, duration_col="os_days", event_col="event")
    """

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        if not LIFELINES_AVAILABLE:
            raise ImportError("pip install lifelines")
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "os_days",
        event_col: str = "event",
        feature_cols: Optional[List[str]] = None,
    ) -> "CoxSurvivalModel":
        """
        Modeli eğit.

        Args:
            df           : Özellik ve sürviyal verilerini içeren DataFrame
            duration_col : Takip süresi sütunu (gün)
            event_col    : Olay göstergesi sütunu (1=ölüm, 0=sansürlü)
            feature_cols : Kullanılacak özellik sütunları (None→tüm diğerleri)
        """
        if feature_cols is None:
            feature_cols = [
                c for c in df.columns if c not in [duration_col, event_col]
            ]

        train_df = df[feature_cols + [duration_col, event_col]].copy()
        train_df = train_df.dropna()

        self.model.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )
        self._is_fitted = True
        logger.info(
            f"Cox PH modeli eğitildi | "
            f"C-index (train): {self.model.concordance_index_:.4f}"
        )
        return self

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Risk skorları tahmin et (yüksek skor = kötü prognoz).

        Returns:
            (N,) risk skor dizisi
        """
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi — önce .fit() çağırın")
        return self.model.predict_partial_hazard(df).values

    def predict_survival_function(
        self, df: pd.DataFrame, times: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Sürviyal fonksiyonunu tahmin et.

        Args:
            df    : Test DataFrame
            times : Değerlendirme zaman noktaları (gün)

        Returns:
            DataFrame: satır=time, sütun=hasta
        """
        return self.model.predict_survival_function(df, times=times)

    def c_index(
        self,
        df: pd.DataFrame,
        duration_col: str = "os_days",
        event_col: str = "event",
    ) -> float:
        """Concordance Index (C-index) hesapla."""
        risk_scores = self.predict_risk(df)
        return concordance_index(
            df[duration_col].values,
            -risk_scores,
            df[event_col].values,
        )

    def summary(self) -> pd.DataFrame:
        """Model özeti (hazard ratio, p-value, CI)."""
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi")
        return self.model.summary

    def stratify_risk(
        self,
        risk_scores: np.ndarray,
        low_pct: float = 33.3,
        high_pct: float = 66.7,
    ) -> np.ndarray:
        """
        Risk gruplarına ayır.

        Returns:
            (N,) array — değerler: 0=düşük, 1=orta, 2=yüksek risk
        """
        low_threshold = np.percentile(risk_scores, low_pct)
        high_threshold = np.percentile(risk_scores, high_pct)
        groups = np.zeros(len(risk_scores), dtype=int)
        groups[risk_scores > low_threshold] = 1
        groups[risk_scores > high_threshold] = 2
        return groups


# ─── XGBoost Sürviyal Modeli ─────────────────────────────────────────────────

class XGBoostSurvivalModel:
    """
    XGBoost tabanlı sürviyal regresyon modeli.

    objective="survival:cox" kullanır — Cox regresyon kaybı.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not XGB_AVAILABLE:
            raise ImportError("pip install xgboost")

        self.params = {
            "objective": "survival:cox",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": "cox-nloglik",
            "random_state": random_state,
            "verbosity": 0,
        }
        self.model = xgb.XGBRegressor(**self.params) if XGB_AVAILABLE else None
        self._is_fitted = False
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event: np.ndarray,
        eval_set: Optional[Tuple] = None,
    ) -> "XGBoostSurvivalModel":
        """
        Modeli eğit.

        Args:
            X     : Özellik DataFrame'i
            y     : Sürviyal süreleri (gün, pozitif → olay, negatif → sansür)
            event : Olay göstergesi (XGBoost için süre işaretine encode edilir)
            eval_set: (X_val, y_val) tuple
        """
        self.feature_names_ = list(X.columns)
        # XGBoost Cox için: negatif süre → sansürlü, pozitif → olay
        y_encoded = np.where(event == 1, y, -y).astype(np.float32)

        fit_params = {}
        if eval_set is not None:
            X_val, y_val, e_val = eval_set
            y_val_enc = np.where(e_val == 1, y_val, -y_val).astype(np.float32)
            fit_params["eval_set"] = [(X_val.values, y_val_enc)]

        self.model.fit(X.values, y_encoded, **fit_params)
        self._is_fitted = True
        logger.info("XGBoost sürviyal modeli eğitildi")
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Risk skor tahmini (yüksek = kötü prognoz)."""
        if not self._is_fitted:
            raise RuntimeError("Model eğitilmedi")
        return self.model.predict(X.values)

    def explain_shap(
        self, X: pd.DataFrame, max_display: int = 15
    ) -> Optional[np.ndarray]:
        """
        SHAP değerleri ile özellik önem analizi.

        Args:
            X           : Açıklanacak örnekler
            max_display : Gösterilecek maksimum özellik sayısı

        Returns:
            SHAP değerleri matrisi (N, F)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP yüklü değil — pip install shap")
            return None
        if not self._is_fitted:
            raise RuntimeError("Model eğitilmedi")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.values)
        logger.info(f"SHAP değerleri hesaplandı | shape: {shap_values.shape}")
        return shap_values
