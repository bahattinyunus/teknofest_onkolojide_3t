"""
Yapılandırma Yönetimi — YAML tabanlı konfigürasyon sistemi.

Tüm eğitim, model ve veri parametreleri YAML üzerinden yönetilir.
OmegaConf ile tip güvenli yapılandırma sağlanır.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Varsayılan konfigürasyon
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "GlioSight",
        "version": "0.1.0",
        "seed": 42,
        "competition": "TEKNOFEST 2026 Onkolojide 3T",
    },
    "data": {
        "root": "data/raw",
        "splits_file": "data/splits/brats2021_splits.json",
        "modalities": ["t1", "t1ce", "t2", "flair"],
        "normalization": "zscore",
        "crop_size": [128, 128, 128],
        "num_workers": 4,
        "pin_memory": True,
    },
    "model": {
        "type": "unet3d",
        "in_channels": 4,
        "out_channels": 3,
        "base_features": 32,
        "dropout_p": 0.1,
        "deep_supervision": True,
    },
    "loss": {
        "type": "dice_focal",
        "dice_weight": 1.0,
        "focal_weight": 1.0,
        "alpha": 0.25,
        "gamma": 2.0,
        "deep_supervision": True,
        "ds_weights": [1.0, 0.5, 0.25],
    },
    "training": {
        "epochs": 300,
        "batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "use_amp": True,
        "patience": 50,
        "gradient_clip": 12.0,
    },
    "augmentation": {
        "flip_p": 0.5,
        "rotation_p": 0.3,
        "max_rotation_angle": 15.0,
        "noise_p": 0.2,
        "noise_std": 0.05,
        "intensity_p": 0.3,
        "gamma_p": 0.2,
    },
    "inference": {
        "roi_size": [128, 128, 128],
        "overlap": 0.5,
        "threshold": 0.5,
        "use_tta": False,
        "sw_batch_size": 2,
    },
    "logging": {
        "output_dir": "experiments/results",
        "tensorboard": True,
        "mlflow": False,
        "log_every_n_steps": 10,
    },
}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML konfigürasyon dosyasını yükle.

    Args:
        config_path : YAML dosya yolu

    Returns:
        Konfigürasyon dict'i (varsayılanlarla birleştirilmiş)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    # Derin birleştirme (varsayılan üzerine kullanıcı ayarları)
    config = _deep_merge(DEFAULT_CONFIG.copy(), user_config)
    logger.info(f"Konfigürasyon yüklendi: {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Konfigürasyonu YAML olarak kaydet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"Konfigürasyon kaydedildi: {output_path}")


def set_seed(seed: int) -> None:
    """Tüm kütüphaneler için seed ayarla (tekrarlanabilirlik)."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Seed ayarlandı: {seed}")


def get_device(prefer_gpu: bool = True) -> "torch.device":
    """
    En uygun eğitim cihazını seç.

    Returns:
        torch.device — CUDA > MPS > CPU sırasıyla
    """
    import torch

    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon MPS kullanılıyor")
        else:
            device = torch.device("cpu")
            logger.warning("GPU bulunamadı — CPU kullanılıyor (YAVAŞ olabilir)")
    else:
        device = torch.device("cpu")

    return device


def _deep_merge(base: dict, override: dict) -> dict:
    """İki dict'i derin olarak birleştir."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    """Uygulama geneli loglama ayarı."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
