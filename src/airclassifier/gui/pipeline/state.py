"""
Pipeline State Management
=========================

Tracks the state of the multi-stage simulation pipeline:
Pretreatment -> Milling -> Air Classification

Stores results from each completed stage for data transfer and persistence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class PipelineStage(Enum):
    """Pipeline stages in processing order."""
    PRETREATMENT = auto()
    MILLING = auto()
    CLASSIFICATION = auto()


@dataclass
class StageResult:
    """Holds result from a completed simulation stage."""

    stage: PipelineStage
    result: Dict[str, Any]          # Full result dict from simulation
    outlet: Any                      # OutletState or MillingOutletState
    completed_at: float              # Unix timestamp
    params_used: Dict[str, Any]      # Configuration used for this run

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for project save."""
        # Note: outlet objects need special handling for serialization
        outlet_dict = {}
        if self.outlet is not None:
            # Convert outlet dataclass to dict if it has to_dict method
            if hasattr(self.outlet, 'to_dict'):
                outlet_dict = self.outlet.to_dict()
            elif hasattr(self.outlet, '__dataclass_fields__'):
                # Fallback for dataclasses without to_dict
                outlet_dict = {
                    k: getattr(self.outlet, k)
                    for k in self.outlet.__dataclass_fields__
                    if not k.startswith('_')
                }
                # Filter out non-serializable types (numpy arrays, etc.)
                outlet_dict = {
                    k: v for k, v in outlet_dict.items()
                    if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None)))
                }

        return {
            "stage": self.stage.name,
            "completed_at": self.completed_at,
            "params_used": self.params_used,
            "outlet_summary": outlet_dict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageResult":
        """Deserialize from project file.

        Note: Full outlet state is not restored; only summary data is available.
        """
        return cls(
            stage=PipelineStage[data["stage"]],
            result={},  # Full result not persisted
            outlet=None,  # Outlet reconstructed from summary if needed
            completed_at=data.get("completed_at", 0.0),
            params_used=data.get("params_used", {}),
        )


@dataclass
class PipelineState:
    """Tracks the state of the multi-stage pipeline.

    Stores completed stage results and enables data transfer between stages.
    Mass balance tracking ensures consistency across the pipeline.
    """

    # Stage results (None = not yet run)
    pretreatment_result: Optional[StageResult] = None
    milling_result: Optional[StageResult] = None
    classification_result: Optional[StageResult] = None

    # Mass balance tracking [kg]
    initial_mass_kg: float = 0.0
    mass_after_pretreatment_kg: float = 0.0
    mass_after_milling_kg: float = 0.0
    mass_after_classification_kg: float = 0.0

    def is_stage_complete(self, stage: PipelineStage) -> bool:
        """Check if a specific stage has been completed."""
        if stage == PipelineStage.PRETREATMENT:
            return self.pretreatment_result is not None
        elif stage == PipelineStage.MILLING:
            return self.milling_result is not None
        elif stage == PipelineStage.CLASSIFICATION:
            return self.classification_result is not None
        return False

    def get_stage_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Get result for a specific stage."""
        if stage == PipelineStage.PRETREATMENT:
            return self.pretreatment_result
        elif stage == PipelineStage.MILLING:
            return self.milling_result
        elif stage == PipelineStage.CLASSIFICATION:
            return self.classification_result
        return None

    def get_upstream_outlet(self, stage: PipelineStage) -> Optional[Any]:
        """Get outlet state from the stage before this one.

        Returns:
            OutletState for milling stage (from pretreatment)
            MillingOutletState for classification stage (from milling)
            None for pretreatment stage or if upstream not complete
        """
        if stage == PipelineStage.MILLING:
            if self.pretreatment_result is not None:
                return self.pretreatment_result.outlet
        elif stage == PipelineStage.CLASSIFICATION:
            if self.milling_result is not None:
                return self.milling_result.outlet
        return None

    def set_stage_result(
        self,
        stage: PipelineStage,
        result: Dict[str, Any],
        outlet: Any,
        params_used: Dict[str, Any],
    ) -> None:
        """Set result for a stage and clear downstream results.

        When a stage is re-run, downstream results become invalid.
        """
        stage_result = StageResult(
            stage=stage,
            result=result,
            outlet=outlet,
            completed_at=time.time(),
            params_used=params_used,
        )

        if stage == PipelineStage.PRETREATMENT:
            self.pretreatment_result = stage_result
            # Clear downstream
            self.milling_result = None
            self.classification_result = None
            self.mass_after_milling_kg = 0.0
            self.mass_after_classification_kg = 0.0
        elif stage == PipelineStage.MILLING:
            self.milling_result = stage_result
            # Clear downstream
            self.classification_result = None
            self.mass_after_classification_kg = 0.0
        elif stage == PipelineStage.CLASSIFICATION:
            self.classification_result = stage_result

    def clear_all(self) -> None:
        """Reset all pipeline state."""
        self.pretreatment_result = None
        self.milling_result = None
        self.classification_result = None
        self.initial_mass_kg = 0.0
        self.mass_after_pretreatment_kg = 0.0
        self.mass_after_milling_kg = 0.0
        self.mass_after_classification_kg = 0.0

    def get_completed_stages(self) -> list[PipelineStage]:
        """Get list of completed stages in order."""
        completed = []
        if self.pretreatment_result is not None:
            completed.append(PipelineStage.PRETREATMENT)
        if self.milling_result is not None:
            completed.append(PipelineStage.MILLING)
        if self.classification_result is not None:
            completed.append(PipelineStage.CLASSIFICATION)
        return completed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline state for project save."""
        return {
            "pretreatment_result": (
                self.pretreatment_result.to_dict()
                if self.pretreatment_result else None
            ),
            "milling_result": (
                self.milling_result.to_dict()
                if self.milling_result else None
            ),
            "classification_result": (
                self.classification_result.to_dict()
                if self.classification_result else None
            ),
            "initial_mass_kg": self.initial_mass_kg,
            "mass_after_pretreatment_kg": self.mass_after_pretreatment_kg,
            "mass_after_milling_kg": self.mass_after_milling_kg,
            "mass_after_classification_kg": self.mass_after_classification_kg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Deserialize from project file."""
        state = cls()

        if data.get("pretreatment_result"):
            state.pretreatment_result = StageResult.from_dict(
                data["pretreatment_result"]
            )
        if data.get("milling_result"):
            state.milling_result = StageResult.from_dict(
                data["milling_result"]
            )
        if data.get("classification_result"):
            state.classification_result = StageResult.from_dict(
                data["classification_result"]
            )

        state.initial_mass_kg = data.get("initial_mass_kg", 0.0)
        state.mass_after_pretreatment_kg = data.get("mass_after_pretreatment_kg", 0.0)
        state.mass_after_milling_kg = data.get("mass_after_milling_kg", 0.0)
        state.mass_after_classification_kg = data.get("mass_after_classification_kg", 0.0)

        return state
