"""
MRI Loader — BraTS formatında NIfTI dosya yükleme modülü.

BraTS 2021/2023 veri seti formatını destekler:
    - T1 (T1-weighted)
    - T1ce (T1 with contrast enhancement)
    - T2 (T2-weighted)
    - FLAIR (Fluid Attenuated Inversion Recovery)
    - Seg (segmentasyon maskesi)

Tümör bölgeleri:
    - WT  (Whole Tumor)         = label 1 + 2 + 4
    - TC  (Tumor Core)          = label 1 + 4
    - ET  (Enhancing Tumor)     = label 4
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# BraTS modalite sabitleri
MODALITIES = ["t1", "t1ce", "t2", "flair"]
SEG_LABELS = {
    "background": 0,
    "necrotic_core": 1,    # NCR/NET
    "edema": 2,            # ED
    "enhancing": 4,        # ET
}

# BraTS tümör bölgesi maskesi hesaplama
REGION_COMBINATIONS = {
    "WT": [1, 2, 4],   # Whole Tumor
    "TC": [1, 4],      # Tumor Core
    "ET": [4],         # Enhancing Tumor
}


def load_nifti(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    NIfTI dosyasını yükle ve numpy dizisine dönüştür.

    Args:
        path: NIfTI dosya yolu (.nii veya .nii.gz)

    Returns:
        data  : (H, W, D) şeklinde float32 numpy dizisi
        affine: (4, 4) affin dönüşüm matrisi
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI dosyası bulunamadı: {path}")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    logger.debug(f"Yüklendi: {path.name} | shape: {data.shape}")
    return data, affine


def load_brats_subject(
    subject_dir: Union[str, Path],
    modalities: List[str] = MODALITIES,
    load_seg: bool = True,
) -> Dict[str, np.ndarray]:
    """
    BraTS formatındaki bir hasta dizinini yükle.

    BraTS dizin yapısı:
        BraTS2021_XXXXX/
            BraTS2021_XXXXX_t1.nii.gz
            BraTS2021_XXXXX_t1ce.nii.gz
            BraTS2021_XXXXX_t2.nii.gz
            BraTS2021_XXXXX_flair.nii.gz
            BraTS2021_XXXXX_seg.nii.gz

    Args:
        subject_dir : Hasta dizin yolu
        modalities  : Yüklenecek modalite listesi
        load_seg    : Segmentasyon maskesini de yükle

    Returns:
        {
            "t1"   : (H, W, D) ndarray,
            "t1ce" : (H, W, D) ndarray,
            "t2"   : (H, W, D) ndarray,
            "flair": (H, W, D) ndarray,
            "seg"  : (H, W, D) ndarray  [isteğe bağlı],
            "subject_id": str,
        }
    """
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name
    result: Dict[str, np.ndarray] = {"subject_id": subject_id}

    for mod in modalities:
        # BraTS 2021 isimlendirme formatı
        candidates = [
            subject_dir / f"{subject_id}_{mod}.nii.gz",
            subject_dir / f"{subject_id}_{mod}.nii",
            subject_dir / f"{mod}.nii.gz",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            raise FileNotFoundError(
                f"'{mod}' modalitesi bulunamadı: {subject_dir}"
            )
        data, _ = load_nifti(found)
        result[mod] = data

    if load_seg:
        seg_candidates = [
            subject_dir / f"{subject_id}_seg.nii.gz",
            subject_dir / f"{subject_id}_seg.nii",
            subject_dir / "seg.nii.gz",
        ]
        for c in seg_candidates:
            if c.exists():
                seg_data, _ = load_nifti(c)
                result["seg"] = seg_data.astype(np.int32)
                break

    return result


def compute_region_masks(
    seg: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Segmentasyon etiketlerinden BraTS bölge maskelerini hesapla.

    Args:
        seg: (H, W, D) segmentasyon maskesi (etiketler: 0, 1, 2, 4)

    Returns:
        {
            "WT": (H, W, D) bool array,
            "TC": (H, W, D) bool array,
            "ET": (H, W, D) bool array,
        }
    """
    masks = {}
    for region, labels in REGION_COMBINATIONS.items():
        mask = np.zeros_like(seg, dtype=bool)
        for lbl in labels:
            mask |= (seg == lbl)
        masks[region] = mask
    return masks


def load_dataset_from_splits(
    data_root: Union[str, Path],
    split_file: Union[str, Path],
    split_key: str = "train",
    modalities: List[str] = MODALITIES,
    load_seg: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    JSON split dosyasından veri setini tüm belleğe yükle.

    Args:
        data_root  : Ham veri kök dizini
        split_file : JSON split dosyası ({"train": [...], "val": [...], "test": [...]})
        split_key  : "train" | "val" | "test"
        modalities : Yüklenecek modaliteler
        load_seg   : Segmentasyon maskesi yükle

    Returns:
        Liste[hasta_dict]
    """
    import json

    data_root = Path(data_root)
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    subject_ids = splits.get(split_key, [])
    logger.info(f"Yükleniyor: {split_key} split | {len(subject_ids)} hasta")

    dataset = []
    for sid in subject_ids:
        subject_dir = data_root / sid
        try:
            sample = load_brats_subject(
                subject_dir, modalities=modalities, load_seg=load_seg
            )
            dataset.append(sample)
        except FileNotFoundError as e:
            logger.warning(f"Atlandı ({sid}): {e}")

    logger.info(f"Yükleme tamamlandı: {len(dataset)} / {len(subject_ids)} hasta")
    return dataset


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """
    Numpy dizisini NIfTI formatında kaydet.

    Args:
        data        : Kaydedilecek veri dizisi
        affine      : Affin dönüşüm matrisi
        output_path : Çıkış dosya yolu
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(output_path))
    logger.info(f"Kaydedildi: {output_path}")
