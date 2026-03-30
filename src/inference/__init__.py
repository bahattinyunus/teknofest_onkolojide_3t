"""Inference modülü."""
from .segmentation_pipeline import SegmentationPipeline, logits_to_segmentation

__all__ = ["SegmentationPipeline", "logits_to_segmentation"]
