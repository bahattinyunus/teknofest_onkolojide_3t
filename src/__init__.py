"""
GlioSight — TEKNOFEST 2026 Onkolojide 3T
Multimodal AI ile Glioblastoma Tanı ve Tedavi Yanıt Tahmini

Modüller:
    preprocessing : MRI yükleme, normalizasyon, artırma
    models        : 3D U-Net, sürviyal modeli
    training      : Eğitim döngüsü, kayıp fonksiyonları, metrikler
    inference     : Segmentasyon ve sürviyal pipeline
    utils         : Görselleştirme, konfigürasyon
"""

__version__ = "0.1.0"
__author__ = "GlioSight Team"
__license__ = "MIT"
