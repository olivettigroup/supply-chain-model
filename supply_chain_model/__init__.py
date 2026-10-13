"""Material-agnostic supply chain model: stages linked to their parent through cost and to their
children through demand.

The classes and functions needed to build and run a chain are importable from the package itself,
e.g. ``from supply_chain_model import SupplyChain``. Everything else stays in its submodule.
"""

from .supply_chain import ModelParams, SupplyChain, StageContext
from .supply_chain_stages import (
    SupplyPlant, Mine, WasteStream,
    SupplyChainStage, MiningStage, ProcessingStage,
    Pipeline, PoolContext, mine_npv, default_grade_fn, flat_decision_price,
)
from .utils import SupplyCurve, SupplyCurve2D, SectorDemand, CompositeDemand, generate_cost_curve

__all__ = [
    "ModelParams", "SupplyChain", "StageContext",
    "SupplyPlant", "Mine", "WasteStream",
    "SupplyChainStage", "MiningStage", "ProcessingStage",
    "Pipeline", "PoolContext", "mine_npv", "default_grade_fn", "flat_decision_price",
    "SupplyCurve", "SupplyCurve2D", "SectorDemand", "CompositeDemand", "generate_cost_curve",
]
