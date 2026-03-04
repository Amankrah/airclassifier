"""
Pipeline Orchestration Package
==============================

Manages the multi-stage simulation pipeline:
Pretreatment (GP-15) -> Milling (Hammer Mill) -> Air Classifier

Components:
    - PipelineState: Tracks completed stages and their results
    - PipelineStage: Enum for stage identification
    - StageResult: Stores result from a completed stage
    - map_pretreatment_to_milling: Transform PT outlet -> Mill input
    - map_milling_to_classification: Transform Mill outlet -> Classifier input
"""

from .state import PipelineState, PipelineStage, StageResult
from .mappings import (
    map_pretreatment_to_milling,
    map_milling_to_classification,
    get_pipeline_summary,
)

__all__ = [
    "PipelineState",
    "PipelineStage",
    "StageResult",
    "map_pretreatment_to_milling",
    "map_milling_to_classification",
    "get_pipeline_summary",
]
