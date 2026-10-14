"""Material-agnostic supply chain model: stages linked to their parent through cost and to their
children through demand.

The classes and functions needed to build and run a chain are importable from the package itself,
e.g. ``from supply_chain_model import SupplyChain``. Everything else stays in its submodule.
"""

from .supply_chain import ModelParams, SupplyChain, StageContext
from .supply_chain_stages import (
    SupplyPlant, Mine, WasteStream, RecycledSupply,
    SupplyChainStage, MiningStage, ProcessingStage,
    Pipeline, PoolContext, mine_npv, plant_npv, default_grade_fn, flat_decision_price,
)
from .utils import SupplyCurve, SupplyCurve2D, Sector, CompositeDemand, RecyclingTechnology, generate_cost_curve
from .data import (
    ModelInput, generate_inputs, fill_from_summaries, validate_inputs, params_from_inputs, chain_from_inputs,
    write_template, write_inputs, load_inputs, write_results,
)

__all__ = [
    "ModelParams", "SupplyChain", "StageContext",
    "SupplyPlant", "Mine", "WasteStream", "RecycledSupply",
    "SupplyChainStage", "MiningStage", "ProcessingStage",
    "Pipeline", "PoolContext", "mine_npv", "plant_npv", "default_grade_fn", "flat_decision_price",
    "SupplyCurve", "SupplyCurve2D", "Sector", "CompositeDemand", "RecyclingTechnology", "generate_cost_curve",
    "ModelInput", "generate_inputs", "fill_from_summaries", "validate_inputs", "params_from_inputs",
    "chain_from_inputs", "write_template", "write_inputs", "load_inputs", "write_results",
]
