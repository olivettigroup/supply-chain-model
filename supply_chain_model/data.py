"""Model inputs: the ModelInput tables, synthetic data generation, filling pools from stage summaries,
validation, building a chain, and Excel input and output.

A chain can be built and run from ModelInput tables (plain dicts of numpy arrays) without any
workbook: pandas and openpyxl are only imported by the Excel functions.
"""
import re
import warnings
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.optimize import brentq

from .supply_chain import ModelParams, SupplyChain
from .supply_chain_stages import (
    Mine, MiningStage, ProcessingStage, SupplyPlant, WasteStream, default_grade_fn, mine_npv, plant_npv,
)
from .utils import (
    CompositeDemand, RecyclingTechnology, Sector, SupplyCurve, SupplyCurve2D, generate_cost_curve, plant_quantities,
)

NAN = np.nan

# Every input table: {column: (value of a missing entry, meaning and units)}. Prices and costs are in
# real 2024 currency per quantity unit of their stage. In a workbook each table is one sheet (TITLES),
# except plants, which get one sheet per stage (PLANT_SHEET_COLUMNS).
SHEETS = {
    "stages": {
        "stage": ("", "Stage name, unique."),
        "type": ("", "'mining' or 'processing'. A mining stage must be the root."),
        "parent": ("", "Stage this one buys its input from; blank for the root."),
        "intensity": (NAN, "Parent material per unit of this stage's output (processing stages)."),
        "intensity_unit": ("", "Unit of intensity, '<parent quantity_unit>/<quantity_unit>'. Optional; checked when given."),
        "eta": (NAN, "Supply elasticity of each plant once the price is above its cost."),
        "quantity_unit": ("", "Unit of the stage's quantities (e.g. t)."),
        "price_unit": ("", "Unit of the stage's prices (e.g. $/t)."),
        "ref_parent_price": (NAN, "Parent price at which the stage's plant costs in `plants` are quoted (processing stages with a parent)."),
        "capex_intensity": (NAN, "Capex of new capacity per unit of annual capacity. Required for processing stages, and for mining stages unless every mine gives its own capex."),
        "base_price": (NAN, "The stage's price at which the V0 of the sectors drawing from it apply. Required when sectors draw from the stage."),
    },
    "plants": {
        "stage": ("", "Stage the plant belongs to."),
        "plant_name": ("", "Optional label."),
        "kind": ("", "'mine' (mining stages), 'plant' or 'waste' (processing stages)."),
        "pool": ("", "Mining: resource, developing, incentive, opening, operating, closed or exhausted. Processing: construction, operating, idle or closed."),
        "capacity": (NAN, "Annual capacity, in the stage's quantity unit."),
        "cost": (NAN, "Total cost per unit; for a processing plant, at the stage's ref_parent_price."),
        "grade": (1.0, "Ore grade (mines)."),
        "recovery": (1.0, "Metallurgical recovery (mines)."),
        "capex": (NAN, "Total capex in currency. Blank: capex_intensity x capacity."),
        "reserves": (NAN, "Remaining reserves in quantity units (mines). Blank: default_reserve_life x capacity."),
        "years_in_pool": (0, "Full years already spent in the current pool."),
        "synthetic": (False, "True for plants made by the generator or the stage-summary fill."),
        "capacity_ratio": (NAN, "Waste streams: share of the source stage's production that becomes scrap."),
        "source": ("", "Waste streams: stage whose production feeds the stream (blank: its own stage)."),
        "mode": ("", "Waste streams: 'flow' or 'stock' (blank: flow)."),
    },
    "stage_summaries": {
        "stage": ("", "Stage whose pool is filled."),
        "pool": ("", "Pool to fill (as in `plants`)."),
        "total_capacity": (NAN, "Total capacity the pool should have, known plants included."),
        "n_plants": (NAN, "Number of plants the pool should have, known plants included."),
        "cost_p10": (NAN, "10th percentile of the pool's plant costs (same basis as `plants`)."),
        "cost_p50": (NAN, "Median plant cost."),
        "cost_p90": (NAN, "90th percentile of the pool's plant costs."),
    },
    "demand_sectors": {
        "sector": ("", "Sector name (e.g. EVs)."),
        "driver_unit": ("", "Unit of the sector's driver (e.g. vehicles sold)."),
        "V0": (NAN, "Driver in the first year."),
        "cagr": (0.0, "Driver growth rate at the start of the run."),
        "phi": (1.0, "Factor by which driver growth is scaled by end_year (1: constant growth)."),
        "chi": (1.0, "Factor by which every intensity of the sector has changed by end_year (1: constant)."),
        "elasticity": (NAN, "Price elasticity of the driver to the sector's input price."),
        "other_cost": (0.0, "Non-material cost per driver unit, part of the sector's input price."),
        "lifetime": (NAN, "Mean product lifetime in years. Blank: no end-of-life flows."),
        "lifetime_sd": (NAN, "Standard deviation of the lifetime. Blank: lifetime_sd_fraction x lifetime."),
        "collection_rate": (1.0, "Share of end-of-life products collected."),
        "disassembly_rate": (1.0, "Share of the collected material that can be physically extracted."),
        "technology": ("", "Recycling technology the scrap goes through (from `recycling_technologies`). Blank: nothing is recovered."),
        "history_growth": (NAN, "Driver growth rate before the run, used by the backcast. Blank: the sector's own equations."),
    },
    "sector_intensities": {
        "sector": ("", "Sector (from `demand_sectors`)."),
        "stage": ("", "Stage the sector draws material from."),
        "I0": (NAN, "Material drawn per driver unit in the first year."),
        "intensity_unit": ("", "Unit of I0, '<stage quantity_unit>/<driver_unit>'. Optional; checked when given."),
        "recycling_target": ("", "Processing stage the recycled material re-enters. Blank: the stage it was drawn from."),
    },
    "recycling_technologies": {
        "technology": ("", "Technology name."),
        "stage": ("", "Stage (element) the rate applies to; elements not listed are not recovered."),
        "recovery_rate": (NAN, "Share of the element in collected, disassembled scrap that the technology recovers."),
    },
    "parameters": {
        "name": ("", "A ModelParams field."),
        "value": ("", "Its value. Parameters left out keep their defaults."),
        "description": ("", "What the parameter does."),
    },
}

# Workbook sheet of each table, and what it holds (the first sheet, "Sheets", lists these)
TITLES = {
    "stages": ("Stages", "One row per stage of the chain: its type, its parent and its stage-level inputs."),
    "plants": ("Plants - <stage>", "One sheet per stage, one row per plant (mine, processing plant or waste stream) "
                                   "in any pool. Rename the sheet after the stage; empty plant sheets are ignored."),
    "stage_summaries": ("Stage summaries", "Totals and cost percentiles of a stage's pool, used to fill in the plants "
                                           "the plant sheets lack."),
    "demand_sectors": ("Demand sectors", "One row per demand sector (e.g. EVs): its driver, growth, elasticity and end of life."),
    "sector_intensities": ("Sector intensities", "How much each sector draws from each stage per driver unit, and "
                                                 "where its recycled material goes."),
    "recycling_technologies": ("Recycling technologies", "Recovery rate of each recycling technology for each element."),
    "parameters": ("Parameters", "Model parameters (ModelParams); left-out parameters keep their defaults."),
}
PLANTS_PREFIX = "Plants - "

# Plant columns on a stage's sheet (the stage is the sheet's name; mining stages hold mines only)
PLANT_SHEET_COLUMNS = {
    "mining": ["plant_name", "pool", "capacity", "cost", "grade", "recovery", "capex", "reserves", "years_in_pool",
               "synthetic"],
    "processing": ["plant_name", "kind", "pool", "capacity", "cost", "capex", "years_in_pool", "synthetic",
                   "capacity_ratio", "source", "mode"],
}

PIPELINES = {"mining": MiningStage.POOLS, "processing": ProcessingStage.POOLS}

# Numerical guard: generated costs never fall below this fraction of their stage's reference price
MIN_GENERATED_COST_FRACTION = 0.01


def _empty(sheet):
    return _table([], sheet)


@dataclass
class ModelInput:
    """Model inputs as one table per workbook sheet.

    Each table is a dict {column: numpy array} with one entry per row, with the columns in SHEETS. In a
    workbook each table is one sheet (TITLES), except plants, which are split into one sheet per stage.
    Costs in `plants` and `stage_summaries` are total costs: for a processing plant, at its stage's
    ref_parent_price.

    Attributes:
        stages (dict): One row per stage.
        plants (dict): One row per plant, in long format (stage and pool are columns).
        demand_sectors (dict): One row per demand sector.
        sector_intensities (dict): One row per sector and stage it draws from.
        recycling_technologies (dict): One row per technology and element it recovers.
        stage_summaries (dict): One row per (stage, pool) to fill from totals and cost percentiles.
        parameters (dict): ModelParams values that differ from the defaults.
    """
    stages: dict
    plants: dict
    demand_sectors: dict
    sector_intensities: dict
    recycling_technologies: dict = field(default_factory=lambda: _empty("recycling_technologies"))
    stage_summaries: dict = field(default_factory=lambda: _empty("stage_summaries"))
    parameters: dict = field(default_factory=lambda: _empty("parameters"))


def _rows(table):
    """The rows of a table as a list of dicts."""
    return [dict(zip(table, values)) for values in zip(*table.values())]


def _table(rows, sheet):
    """A table {column: array} of a sheet from a list of row dicts; missing entries take the column's default."""
    return {name: np.array([row.get(name, default) for row in rows], dtype=object if isinstance(default, str) else None)
            if rows else np.array([], dtype=object if isinstance(default, str) else float)
            for name, (default, _) in SHEETS[sheet].items()}


def _missing(value):
    """True for an empty string or a NaN, the two ways a table marks a missing entry."""
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


def _value(x, default=None):
    """x, or default when x is missing."""
    return default if _missing(x) else x


def _parameter_descriptions():
    """{ModelParams field: description}, read from the ModelParams docstring."""
    doc = ModelParams.__doc__.split("Attributes:")[1]
    descriptions, name = {}, None
    for line in doc.splitlines():
        match = re.match(r"\s{8}(\w+) \(\w+\): (.*)", line)
        if match:
            name = match.group(1)
            descriptions[name] = match.group(2)
        elif name and line.strip():
            descriptions[name] += " " + line.strip()
    return descriptions


def _parameter_rows(params):
    """One parameters row per ModelParams field, with the value from params."""
    descriptions = _parameter_descriptions()
    return [{"name": f.name, "value": getattr(params, f.name), "description": descriptions.get(f.name, "")}
            for f in fields(ModelParams)]


def generate_inputs(topology, targets, params, known_plants=None, seed=None, *, cost_curve_steepness=2.0,
                    p90_margin=0.2, max_cost_margin=0.02, reserve_life_range=(15, 40), incentive_cost_spread=1.5,
                    resource_years=300, resource_cost_max_factor=3, capex_price_ratio=1.4,
                    processing_cost_steepness=1.5, plant_capex_conversion_ratio=3.0, capacity_log_sd=0.5):
    """Synthetic model inputs that start close to a steady state.

    Steps: base quantities bottom-up from the sectors' first-year draws; base prices top-down (the
    mining price p0, then each processing stage at its template plant's break-even); operating plants
    drawn to the cost calibration targets and scaled so that supply at the base prices equals the base
    quantities; then the mine pipeline pools sized by the yearly flow of new capacity the stage needs,
    and the processing construction pools by demand growth. Known plants are placed first, and only
    the residual capacity of each (stage, pool) is generated.

    Args:
        topology (list): One dict per stage, with the `stages` columns: "stage", "type" ("mining" or
            "processing"), "parent" (None or "" for the root), "intensity" (processing), "eta",
            "quantity_unit" and "price_unit". A mining stage must be the root.
        targets (dict): "p0": the mining stage's base price; "conversion_cost": {processing stage:
            typical (median) conversion cost}; "n_plants": {stage: number of operating plants};
            "sectors": a list of sector dicts with the `demand_sectors` columns (at least "sector", "V0",
            "cagr" and "elasticity"), plus "intensities": {stage: I0} and optionally
            "recycling_targets": {stage: target stage}. Every leaf stage needs a sector drawing from it.
        params (ModelParams): Model parameters; the pipeline sizes, NPVs and growth rates use them, and
            they fill the parameters table.
        known_plants (dict, optional): A plants table of plants that exist in the data. They are kept
            as given, and generation fills only what they leave missing. Defaults to None.
        seed (int, optional): Seed of the one random generator used throughout. Defaults to None.
        cost_curve_steepness (float, optional): P90 / P25 ratio of operating and incentive mine costs.
        p90_margin (float, optional): The P90 operating mine cost is p0 * (1 - p90_margin).
        max_cost_margin (float, optional): Every existing plant's total cost is at most
            (1 - max_cost_margin) times its stage's base price.
        reserve_life_range (tuple, optional): Range of the uniform draw of full reserves, in years of
            capacity.
        incentive_cost_spread (float, optional): Incentive (and developing) mine costs are capped at
            incentive_cost_spread * c*, where c* is the break-even mine cost.
        resource_years (float, optional): Total resource-pool reserves, in years of base production.
        resource_cost_max_factor (float, optional): Resource mine costs spread from c* up to
            resource_cost_max_factor * c*.
        capex_price_ratio (float, optional): Mining capex_intensity = capex_price_ratio * p0.
        processing_cost_steepness (float, optional): P90 / P25 ratio of processing conversion costs.
        plant_capex_conversion_ratio (float, optional): Processing capex_intensity =
            plant_capex_conversion_ratio * median conversion cost.
        capacity_log_sd (float, optional): Standard deviation of log plant capacity, for every pool
            (see generate_cost_curve). The copper fit's own is about 2.15; the default 0.5 keeps the
            largest of ~100 plants at a few percent of the stage. Pass None for the copper fit.

    Returns:
        ModelInput: The generated tables.

    Raises:
        ValueError: If the topology isn't a single tree with a mining root (when it has a mining
            stage), if a leaf has no sector drawing from it, or if incentive_ratio is too small to hold
            the development pipeline.
    """
    rng = np.random.default_rng(seed)
    specs = {spec["stage"]: dict(spec) for spec in topology}
    known = _rows(known_plants) if known_plants is not None else []
    sectors = list(targets["sectors"])

    # Tree structure, top-down order
    children = {name: [] for name in specs}
    roots = []
    for name, spec in specs.items():
        parent = spec.get("parent") or ""
        spec["parent"] = parent
        if parent:
            children[parent].append(name)
        else:
            roots.append(name)
    if len(roots) != 1:
        raise ValueError(f"The topology needs exactly one root stage; found {roots}.")
    for name, spec in specs.items():
        if spec["type"] == "mining" and spec["parent"]:
            raise ValueError(f"Mining stage {name!r} has a parent; a mining stage must be the root.")
    order = [roots[0]]
    for name in order:
        order.extend(children[name])

    # 1. Base quantities and demand growth, bottom-up: child stages' draws plus sectors' first-year draws
    q0, growth = {}, {}
    for name in reversed(order):
        flows = [(specs[c]["intensity"]*q0[c], growth[c]) for c in children[name]]
        flows += [(s["V0"]*s["intensities"][name], s.get("cagr", 0.0)) for s in sectors if name in s["intensities"]]
        if not flows:
            raise ValueError(f"Leaf stage {name!r} has no sector drawing from it.")
        q0[name] = float(sum(f for f, _ in flows))
        growth[name] = float(sum(f*g for f, g in flows)/q0[name])

    # 2-8. Prices top-down, then each stage's plants
    price, capex_intensity, plant_rows = {}, {}, []
    knobs = dict(cost_curve_steepness=cost_curve_steepness, p90_margin=p90_margin, max_cost_margin=max_cost_margin,
                 reserve_life_range=reserve_life_range, incentive_cost_spread=incentive_cost_spread,
                 resource_years=resource_years, resource_cost_max_factor=resource_cost_max_factor,
                 processing_cost_steepness=processing_cost_steepness,
                 plant_capex_conversion_ratio=plant_capex_conversion_ratio, capacity_log_sd=capacity_log_sd)
    for name in order:
        spec = specs[name]
        parent_price = price[spec["parent"]] if spec["parent"] else 0.0
        known_here = [row for row in known if row["stage"] == name]
        if spec["type"] == "mining":
            price[name] = targets["p0"]
            capex_intensity[name] = capex_price_ratio*targets["p0"]
            plant_rows += _generate_mines(name, spec, q0[name], growth[name], targets["p0"], capex_intensity[name],
                                          targets["n_plants"][name], known_here, params, rng, **knobs)
        else:
            rows, price[name], capex_intensity[name] = _generate_processing(
                name, spec, q0[name], growth[name], parent_price, targets["conversion_cost"][name],
                targets["n_plants"][name], known_here, params, rng, **knobs)
            plant_rows += rows

    drawn = {stage for s in sectors for stage in s["intensities"]}
    stage_rows = []
    for name in order:
        spec = specs[name]
        parent = spec["parent"]
        stage_rows.append({
            **{column: spec[column] for column in SHEETS["stages"] if column in spec},
            "parent": parent,
            "intensity_unit": (f"{specs[parent].get('quantity_unit', '')}/{spec.get('quantity_unit', '')}"
                               if parent else ""),
            "ref_parent_price": price[parent] if parent else NAN,
            "capex_intensity": capex_intensity[name],
            "base_price": price[name] if name in drawn else NAN,
        })

    sector_rows = [{k: v for k, v in s.items() if k not in ("intensities", "recycling_targets")} for s in sectors]
    intensity_rows = [{"sector": s["sector"], "stage": stage, "I0": I0,
                       "recycling_target": s.get("recycling_targets", {}).get(stage, "")}
                      for s in sectors for stage, I0 in s["intensities"].items()]
    return ModelInput(stages=_table(stage_rows, "stages"), plants=_table(plant_rows, "plants"),
                      demand_sectors=_table(sector_rows, "demand_sectors"),
                      sector_intensities=_table(intensity_rows, "sector_intensities"),
                      parameters=_table(_parameter_rows(params), "parameters"))


def _years_in_pool(n, residence, rng):
    """Desynchronised years_in_pool for plants in a timed pool: uniform on [0, residence - 1]."""
    return rng.integers(0, max(residence, 1), n)


def _known_capacity(known, pool):
    """Total capacity of the known plants in one pool."""
    return sum(row["capacity"] for row in known if row["pool"] == pool)


def _residual(known, pool, target, stage):
    """Capacity left to generate in a pool once its known plants are placed; warns if they exceed it."""
    residual = target - _known_capacity(known, pool)
    if residual < 0:
        warnings.warn(f"Known plants in {stage!r} {pool} exceed the target capacity ({target:.4g}); "
                      "nothing is generated for that pool.")
    return max(residual, 0.0)


def _plant_rows(stage, kind, pool, capacities, costs, **columns):
    """Rows of generated plants; each extra column is a scalar or an array with one value per plant."""
    rows = []
    for i, (capacity, cost) in enumerate(zip(capacities, costs)):
        row = {"stage": stage, "plant_name": f"{stage} {pool} {i}", "kind": kind, "pool": pool,
               "capacity": float(capacity), "cost": float(cost), "synthetic": True}
        row.update({key: value[i] if np.ndim(value) else value for key, value in columns.items()})
        rows.append(row)
    return rows


def _n_plants(capacity, reference_count, reference_capacity):
    """Number of plants for a pool: the same average size as the reference pool, at least one if the
    pool has any capacity."""
    if capacity <= 0:
        return 0
    return max(1, int(round(reference_count*capacity/reference_capacity)))


def _place_below(costs, capacities, boundary, capacity_below):
    """Scale costs so that the cheapest `capacity_below` of capacity costs less than `boundary`.

    The boundary falls between two plants (at the geometric mean of their scaled costs), never on one,
    so no plant sits exactly at it. If capacity_below is less than the cheapest plant's capacity, the
    cheapest plant ends up 5% above the boundary.
    """
    order = np.argsort(costs)
    cum = np.cumsum(capacities[order])
    k = int(np.sum(cum <= capacity_below*(1 + 1e-9)))
    if k == 0:
        reference = costs[order[0]]/1.05
    elif k == len(costs):
        reference = costs[order[-1]]*1.05
    else:
        reference = np.sqrt(costs[order[k - 1]]*costs[order[k]])
    return costs*boundary/reference


def _generate_mines(name, spec, q0, growth, p0, capex_intensity, n_operating, known, params, rng, *,
                    cost_curve_steepness, p90_margin, max_cost_margin, reserve_life_range, incentive_cost_spread,
                    resource_years, resource_cost_max_factor, capacity_log_sd, **_):
    """Rows of every mine of a mining stage, pool by pool (steps 3-7 of generate_inputs)."""
    eta = spec["eta"]
    low_life, high_life = reserve_life_range
    floor = MIN_GENERATED_COST_FRACTION*p0

    def lives(n):
        return rng.uniform(low_life, high_life, n)

    # 3. Operating: costs to the calibration targets, capacities scaled so supply(p0) = q0
    p90 = p0*(1 - p90_margin)
    capacities, costs = generate_cost_curve(n_operating, 1.0, rng=rng, capacity_log_sd=capacity_log_sd,
                                            cost_quantiles=((25, p90/cost_curve_steepness), (90, p90)))
    costs = np.clip(costs, floor, (1 - max_cost_margin)*p0)

    # 4. Reserves: operating mines start partly depleted, spread evenly from nearly empty to full, as in
    # a pool that has been replaced at a steady rate, so exhaustion runs at that rate from the first year.
    # Remaining reserves are in years of capacity, so a mine with under a year left supplies only that.
    operating_lives = lives(n_operating)
    remaining_years = operating_lives*rng.uniform(0.0, 1.0, n_operating)

    known_operating = [row for row in known if row["pool"] == "operating"]
    known_supply = sum(float(plant_quantities(
        p0, np.array([min(row["capacity"], row["reserves"]) if not _missing(row["reserves"]) else row["capacity"]]),
        np.array([row["cost"]]), eta).sum()) for row in known_operating)
    unit_supply = float(plant_quantities(p0, capacities*np.minimum(remaining_years, 1.0), costs, eta).sum())
    scale = (q0 - known_supply)/unit_supply
    if scale < 0:
        warnings.warn(f"Known operating mines of {name!r} already supply more than the base quantity; "
                      "no operating mines are generated.")
        scale = 0.0
    capacities = capacities*scale
    remaining = remaining_years*capacities
    rows = [{**row, "synthetic": bool(row.get("synthetic", False))} for row in known]
    if scale > 0:
        rows += _plant_rows(name, "mine", "operating", capacities, costs, reserves=remaining)

    # 5. Pipeline pools sized by the yearly flow of new capacity
    K_operating = float(np.sum(capacities)) + _known_capacity(known, "operating")
    exhaustion = float(np.sum(capacities/operating_lives)/np.sum(capacities)) if scale > 0 else 1.0/np.mean(reserve_life_range)
    flow = q0*(params.grade_decline_rate + growth + exhaustion)
    K_opening = flow*params.construction_time
    K_developing = flow*params.feasibility_time
    K_incentive = params.incentive_ratio*q0 - K_developing
    if K_incentive <= 0:
        raise ValueError(
            f"incentive_ratio is too small to hold the development pipeline of {name!r}: it must exceed "
            f"F * feasibility_time / q0 = {flow:.4g} * {params.feasibility_time} / {q0:.4g} "
            f"= {flow*params.feasibility_time/q0:.4g}, but it is {params.incentive_ratio}.")

    # 6. Break-even cost c* of a new mine at a flat p0
    def npv_at(cost):
        mine = Mine(capacity=1.0, cost=cost, reserves=float(np.mean(reserve_life_range)), capex=capex_intensity)
        return mine_npv(mine, np.full(params.mine_life, p0), params, default_grade_fn)
    c_star = brentq(npv_at, 1e-6*p0, p0)

    def incentive_curve(capacity):
        """Mines with the operating curve's steepness, one year's flow of capacity below c*, capped."""
        n = _n_plants(capacity, n_operating, K_operating)
        if n == 0:
            return [], []
        caps, unit_costs = generate_cost_curve(n, capacity, rng=rng, capacity_log_sd=capacity_log_sd,
                                               cost_quantiles=((25, 1.0), (90, cost_curve_steepness)))
        unit_costs = np.maximum(unit_costs, 1e-3)
        share_below = flow/K_incentive
        placed = _place_below(unit_costs, caps, c_star, share_below*capacity)
        return caps, np.minimum(placed, incentive_cost_spread*c_star)

    for pool, target, residence in [("incentive", K_incentive, 0), ("developing", K_developing, params.feasibility_time)]:
        caps, pool_costs = incentive_curve(_residual(known, pool, target, name))
        rows += _plant_rows(name, "mine", pool, caps, pool_costs, reserves=lives(len(caps))*caps,
                            years_in_pool=_years_in_pool(len(caps), residence, rng))

    # Opening mines passed the gate: costs just below c*
    capacity = _residual(known, "opening", K_opening, name)
    n = _n_plants(capacity, n_operating, K_operating)
    if n:
        caps, _ = generate_cost_curve(n, capacity, rng=rng, capacity_log_sd=capacity_log_sd)
        rows += _plant_rows(name, "mine", "opening", caps, c_star*rng.uniform(0.9, 1.0, n),
                            reserves=lives(n)*caps, years_in_pool=_years_in_pool(n, params.construction_time, rng))

    # 7. Resource pool sized by reserves, costs from c* up
    resource_reserves = resource_years*q0 - sum(row["capacity"]*(row["reserves"] if not _missing(row["reserves"])
                                                                 else params.default_reserve_life)
                                                for row in known if row["pool"] == "resource")
    if resource_reserves > 0:
        # About as many mines per unit of reserves as the operating pool has
        n = max(1, int(round(n_operating*resource_reserves/(q0*np.mean(reserve_life_range)))))
        caps, pool_costs = generate_cost_curve(n, 1.0, rng=rng, capacity_log_sd=capacity_log_sd, cost_bounds=(c_star, resource_cost_max_factor*c_star))
        resource_lives = lives(n)
        caps = caps*resource_reserves/np.sum(caps*resource_lives)
        rows += _plant_rows(name, "mine", "resource", caps, pool_costs, reserves=resource_lives*caps)
    return rows


def _generate_processing(name, spec, q0, growth, parent_price, conversion_scale, n_operating, known, params, rng, *,
                         max_cost_margin, processing_cost_steepness, plant_capex_conversion_ratio, capacity_log_sd, **_):
    """Rows of every plant of a processing stage, its base price and its capex intensity (step 3b, 8).

    Returns:
        tuple: (rows, base price, capex_intensity).
    """
    eta = spec["eta"]
    intensity = spec.get("intensity", 0.0) if spec["parent"] else 0.0

    # 3b. Conversion costs around the scale target; the median is the starting frontier
    capacities, conversion = generate_cost_curve(n_operating, 1.0, rng=rng, capacity_log_sd=capacity_log_sd,
                                                 cost_quantiles=((25, 1.0), (90, processing_cost_steepness)))
    conversion = np.maximum(conversion, 1e-3)*conversion_scale/np.median(np.maximum(conversion, 1e-3))
    frontier = float(np.median(conversion))
    capex_intensity = plant_capex_conversion_ratio*frontier

    # Base price: the template plant's break-even at flat prices
    life = params.plant_life

    def npv_at(price):
        template = SupplyPlant(1.0, frontier, intensity=intensity, capex=capex_intensity)
        return plant_npv(template, np.full(life, price), np.full(life, parent_price), params)[0]
    input_cost = intensity*parent_price
    base_price = brentq(npv_at, input_cost + frontier, input_cost + 20*frontier + 1e-9)

    conversion = np.clip(conversion, MIN_GENERATED_COST_FRACTION*base_price,
                         (1 - max_cost_margin)*base_price - input_cost)
    totals = conversion + input_cost

    # Capacities scaled so that supply at the base prices equals q0
    known_operating = [row for row in known if row["pool"] == "operating" and row.get("kind", "plant") != "waste"]
    known_supply = sum(float(plant_quantities(base_price, np.array([row["capacity"]]), np.array([row["cost"]]), eta).sum())
                       for row in known_operating)
    scale = (q0 - known_supply)/float(plant_quantities(base_price, capacities, totals, eta).sum())
    if scale < 0:
        warnings.warn(f"Known operating plants of {name!r} already supply more than the base quantity; "
                      "no operating plants are generated.")
        scale = 0.0
    capacities = capacities*scale

    rows = [{**row, "synthetic": bool(row.get("synthetic", False))} for row in known]
    if scale > 0:
        rows += _plant_rows(name, "plant", "operating", capacities, totals)

    # 8. Construction pool from base growth, at the frontier
    K_operating = float(np.sum(capacities)) + _known_capacity(known, "operating")
    capacity = _residual(known, "construction", growth*K_operating*params.plant_construction_time, name)
    if capacity > 0 and scale > 0:
        typical = float(np.median(capacities))
        n = max(1, int(round(capacity/typical)))
        rows += _plant_rows(name, "plant", "construction", np.full(n, capacity/n), np.full(n, frontier + input_cost),
                            years_in_pool=_years_in_pool(n, params.plant_construction_time, rng))
    return rows, base_price, capex_intensity


def _costs_at_percentiles(raw, targets):
    """Map drawn costs onto target percentiles: linear between the draws' own P10, P50 and P90 and the
    targets, and extended beyond them with the end segments' slopes. Never below
    MIN_GENERATED_COST_FRACTION of the target median. The mapped costs' percentiles hit the targets
    exactly when no bend of the mapping falls between the two plants a percentile interpolates between,
    and closely otherwise.

    Args:
        raw (array): Drawn costs, any scale.
        targets (tuple): (P10, P50, P90) the costs should have.

    Returns:
        array: The costs.
    """
    c10, c50, c90 = (float(c) for c in targets)
    x10, x50, x90 = np.percentile(raw, [10, 50, 90])
    low = (c50 - c10)/(x50 - x10) if x50 > x10 else 0.0
    high = (c90 - c50)/(x90 - x50) if x90 > x50 else 0.0
    costs = np.where(raw <= x50, c50 + (raw - x50)*low, c50 + (raw - x50)*high)
    return np.maximum(costs, MIN_GENERATED_COST_FRACTION*c50)


def fill_from_summaries(model_input, params, seed=None, capacity_log_sd=0.5):
    """Fill pools from their stage_summaries rows (direct fill, no steady-state calibration).

    For each (stage, pool) row, the plants the plants table already has in that pool are kept, and the
    rest are drawn: total_capacity minus their capacity, in n_plants minus their number of plants (at
    least one), with sizes from generate_cost_curve and costs placed so that the drawn plants' P10, P50
    and P90 costs (by plant count) come out at the row's values (_costs_at_percentiles). Plants in timed pools get spread-out
    years_in_pool. Mines get default reserves (default_reserve_life) when the chain is built.

    Args:
        model_input (ModelInput): The inputs.
        params (ModelParams): Model parameters; the residence times of the timed pools.
        seed (int, optional): Seed of the random generator. Defaults to None.
        capacity_log_sd (float, optional): Spread of plant sizes (see generate_cost_curve). Defaults to 0.5.

    Returns:
        ModelInput: The inputs with the drawn plants added to `plants`.
    """
    rng = np.random.default_rng(seed)
    types = dict(zip(model_input.stages["stage"], model_input.stages["type"]))
    plants = _rows(model_input.plants)
    residence = {"developing": params.feasibility_time, "opening": params.construction_time,
                 "closed": params.closed_cooldown, "construction": params.plant_construction_time,
                 "idle": params.max_idle_years}
    drawn = []
    for row in _rows(model_input.stage_summaries):
        name, pool = row["stage"], row["pool"]
        known = [p for p in plants if p["stage"] == name and p["pool"] == pool]
        capacity = _residual(known, pool, float(row["total_capacity"]), name)
        if capacity <= 0:
            continue
        n = max(int(row["n_plants"]) - len(known), 1)
        caps, raw = generate_cost_curve(n, capacity, rng=rng, capacity_log_sd=capacity_log_sd)
        costs = _costs_at_percentiles(raw, (row["cost_p10"], row["cost_p50"], row["cost_p90"]))
        kind = "mine" if types[name] == "mining" else "plant"
        drawn += _plant_rows(name, kind, pool, caps, costs,
                             years_in_pool=_years_in_pool(n, residence.get(pool, 0), rng))
    return ModelInput(**{**{f.name: getattr(model_input, f.name) for f in fields(ModelInput)},
                         "plants": _table(plants + drawn, "plants")})


def _cell(sheet, i, column, rows=None):
    """Location of a table entry as it appears in a workbook (row 1 holds the headers). Plant rows are
    counted within their stage's sheet, so `rows` (the plants table's rows) is needed for them."""
    if sheet == "plants":
        stage = rows[i]["stage"]
        return f"{PLANTS_PREFIX}{stage}, row {sum(r['stage'] == stage for r in rows[:i]) + 2}, column {column!r}"
    return f"{TITLES[sheet][0]}, row {i + 2}, column {column!r}"


def _unit(text):
    return " ".join(str(text).split())


def validate_inputs(model_input):
    """Check that ModelInput tables describe a chain that can be built.

    Checks: the stages form one tree with a mining stage, if any, at the root; plants and stage summaries
    name real stages, pools and kinds, with non-negative capacities and costs; every leaf has a sector
    drawing from it, every stage with sectors has a base_price, every sector draws from at least one
    stage, and sectors, stages, recycling targets and technologies referred to exist; every mining
    stage has operating mines and a capex_intensity (unless every mine gives its own capex); every
    processing stage has a capex_intensity; parameters are ModelParams fields; and intensity units,
    where given, read '<parent quantity_unit>/<quantity_unit>' (stages) and
    '<stage quantity_unit>/<driver_unit>' (sectors), compared after normalising whitespace.

    Args:
        model_input (ModelInput): The inputs.

    Raises:
        ValueError: At the first failed check, naming the sheet, row and column.
    """
    stages = _rows(model_input.stages)
    plants = _rows(model_input.plants)
    sectors = _rows(model_input.demand_sectors)
    draws = _rows(model_input.sector_intensities)
    techs = {row["technology"] for row in _rows(model_input.recycling_technologies)}
    names = {}
    for i, row in enumerate(stages):
        if _missing(row["stage"]) or row["stage"] in names:
            raise ValueError(f"{_cell('stages', i, 'stage')}: stage names must be given and unique.")
        names[row["stage"]] = row
        if row["type"] not in PIPELINES:
            raise ValueError(f"{_cell('stages', i, 'type')}: {row['type']!r} is not 'mining' or 'processing'.")
    roots = [name for name, row in names.items() if _missing(row["parent"])]
    if len(roots) != 1:
        raise ValueError(f"Stages: the chain needs exactly one root stage (blank parent); found {roots}.")
    for i, row in enumerate(stages):
        parent, seen = row["parent"], {row["stage"]}
        if not _missing(parent) and parent not in names:
            raise ValueError(f"{_cell('stages', i, 'parent')}: {parent!r} is not a stage.")
        while not _missing(parent):
            if parent in seen:
                raise ValueError(f"{_cell('stages', i, 'parent')}: the parents of {row['stage']!r} form a loop.")
            seen.add(parent)
            parent = names[parent]["parent"]
        if row["type"] == "mining" and not _missing(row["parent"]):
            raise ValueError(f"{_cell('stages', i, 'parent')}: mining stage {row['stage']!r} must be the root.")
        if not _missing(row["parent"]) and not _missing(row["intensity_unit"]):
            expected = f"{names[row['parent']]['quantity_unit']}/{row['quantity_unit']}"
            if _unit(row["intensity_unit"]) != _unit(expected):
                raise ValueError(f"{_cell('stages', i, 'intensity_unit')}: {row['intensity_unit']!r} should read {expected!r}.")

    for sheet, rows in (("plants", plants), ("stage_summaries", _rows(model_input.stage_summaries))):
        for i, row in enumerate(rows):
            if row["stage"] not in names:
                where = f"{PLANTS_PREFIX}{row['stage']}" if sheet == "plants" else _cell(sheet, i, "stage")
                raise ValueError(f"{where}: {row['stage']!r} is not a stage.")
            if row["pool"] not in PIPELINES[names[row["stage"]]["type"]]:
                raise ValueError(f"{_cell(sheet, i, 'pool', rows)}: {row['pool']!r} is not a pool of a "
                                 f"{names[row['stage']]['type']} stage.")
            for column in ("capacity", "cost", "total_capacity", "cost_p10", "cost_p50", "cost_p90"):
                if column in row and (_missing(row[column]) or row[column] < 0):
                    raise ValueError(f"{_cell(sheet, i, column, rows)}: must be given and not negative.")
            if sheet == "plants":
                allowed = ("mine",) if names[row["stage"]]["type"] == "mining" else ("plant", "waste")
                if row["kind"] not in allowed:
                    raise ValueError(f"{_cell(sheet, i, 'kind', rows)}: {row['kind']!r} should be one of {allowed}.")

    sector_rows = {}
    for i, row in enumerate(sectors):
        sector_rows[row["sector"]] = row
        if not _missing(row["technology"]) and row["technology"] not in techs:
            raise ValueError(f"{_cell('demand_sectors', i, 'technology')}: {row['technology']!r} is not in Recycling technologies.")
    for i, row in enumerate(draws):
        if row["sector"] not in sector_rows:
            raise ValueError(f"{_cell('sector_intensities', i, 'sector')}: {row['sector']!r} is not in Demand sectors.")
        if row["stage"] not in names:
            raise ValueError(f"{_cell('sector_intensities', i, 'stage')}: {row['stage']!r} is not a stage.")
        if not _missing(row["recycling_target"]) and row["recycling_target"] not in names:
            raise ValueError(f"{_cell('sector_intensities', i, 'recycling_target')}: {row['recycling_target']!r} is not a stage.")
        if not _missing(row["intensity_unit"]):
            expected = f"{names[row['stage']]['quantity_unit']}/{sector_rows[row['sector']]['driver_unit']}"
            if _unit(row["intensity_unit"]) != _unit(expected):
                raise ValueError(f"{_cell('sector_intensities', i, 'intensity_unit')}: {row['intensity_unit']!r} should read {expected!r}.")
    for i, row in enumerate(sectors):
        if not any(d["sector"] == row["sector"] for d in draws):
            raise ValueError(f"{_cell('demand_sectors', i, 'sector')}: {row['sector']!r} draws from no stage (Sector intensities).")

    drawn = {d["stage"] for d in draws}
    for i, row in enumerate(stages):
        name = row["stage"]
        if name not in drawn and not any(other["parent"] == name for other in stages):
            raise ValueError(f"{_cell('stages', i, 'stage')}: leaf stage {name!r} has no sector drawing from it.")
    for i, row in enumerate(stages):
        name = row["stage"]
        if name in drawn and _missing(row["base_price"]):
            raise ValueError(f"{_cell('stages', i, 'base_price')}: sectors draw from {name!r}, so it needs a base_price.")
        own = [p for p in plants if p["stage"] == name]
        if row["type"] == "mining":
            if not any(p["pool"] == "operating" for p in own):
                raise ValueError(f"{_cell('stages', i, 'stage')}: mining stage {name!r} has no operating mines "
                                 "(plant sheet or Stage summaries).")
            if _missing(row["capex_intensity"]) and any(_missing(p["capex"]) for p in own):
                raise ValueError(f"{_cell('stages', i, 'capex_intensity')}: needed, since not every mine of {name!r} gives a capex.")
        elif _missing(row["capex_intensity"]):
            raise ValueError(f"{_cell('stages', i, 'capex_intensity')}: processing stage {name!r} needs one.")

    known = {f.name for f in fields(ModelParams)}
    for i, row in enumerate(_rows(model_input.parameters)):
        if row["name"] not in known:
            raise ValueError(f"{_cell('parameters', i, 'name')}: {row['name']!r} is not a ModelParams field.")


def _cast(value, kind):
    """A parameters-table value as the ModelParams field's type."""
    if kind is bool:
        return value.strip().lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
    if kind is int:
        return int(float(value))
    return kind(value)


def params_from_inputs(model_input):
    """ModelParams from the parameters table; parameters it leaves out keep their defaults.

    Args:
        model_input (ModelInput): The inputs.

    Returns:
        ModelParams: The parameters.
    """
    kinds = {f.name: f.type for f in fields(ModelParams)}
    return ModelParams(**{row["name"]: _cast(row["value"], kinds[row["name"]])
                          for row in _rows(model_input.parameters) if not _missing(row["value"])})


def chain_from_inputs(model_input, params=None):
    """Build a SupplyChain from ModelInput tables, after validate_inputs.

    Mining stages get a SupplyCurve of their operating mines; processing stages a SupplyCurve2D of
    their operating plants quoted at ref_parent_price (a SupplyCurve for a root). Plants in other pools
    keep their total cost at ref_parent_price, converted to conversion cost. Each sector draws from the
    stages in its sector_intensities rows; each stage's CompositeDemand lists its child stages and the
    sectors drawing from it.

    Args:
        model_input (ModelInput): The tables.
        params (ModelParams, optional): Model parameters. Defaults to the parameters table.

    Returns:
        SupplyChain: The chain, ready to run.
    """
    validate_inputs(model_input)
    params = params if params is not None else params_from_inputs(model_input)
    stages = {row["stage"]: row for row in _rows(model_input.stages)}
    plants = _rows(model_input.plants)
    draws = _rows(model_input.sector_intensities)
    children = {name: [n for n, row in stages.items() if row["parent"] == name] for name in stages}

    rates = {}
    for row in _rows(model_input.recycling_technologies):
        rates.setdefault(row["technology"], {})[row["stage"]] = float(row["recovery_rate"])
    technologies = {name: RecyclingTechnology(name, r) for name, r in rates.items()}

    sectors = []
    for row in _rows(model_input.demand_sectors):
        own = [d for d in draws if d["sector"] == row["sector"]]
        targets = {d["stage"]: d["recycling_target"] for d in own if not _missing(d["recycling_target"])}
        sectors.append(Sector(
            row["sector"], float(row["elasticity"]), float(row["V0"]), float(row["cagr"]),
            {d["stage"]: float(d["I0"]) for d in own}, phi=float(row["phi"]), chi=float(row["chi"]),
            other_cost=float(row["other_cost"]), lifetime=_value(row["lifetime"]), lifetime_sd=_value(row["lifetime_sd"]),
            collection_rate=float(row["collection_rate"]), disassembly_rate=float(row["disassembly_rate"]),
            technology=technologies.get(row["technology"]) if not _missing(row["technology"]) else None,
            recycling_targets=targets or None, history_growth=_value(row["history_growth"])))

    def build(name):
        row = stages[name]
        drawing = [sector for sector in sectors if name in sector.I0]
        demand = CompositeDemand([build(child) for child in children[name]] + drawing,
                                 base_price=_value(row["base_price"]))

        pools = {}
        mining = row["type"] == "mining"
        intensity = float(_value(row["intensity"], 0.0))
        ref = float(_value(row["ref_parent_price"], 0.0))
        for p in (p for p in plants if p["stage"] == name):
            if mining:
                plant = Mine(capacity=p["capacity"], cost=p["cost"], grade=p["grade"], recovery=p["recovery"],
                             reserves=_value(p["reserves"]), capex=_value(p["capex"]),
                             years_in_pool=int(p["years_in_pool"]), synthetic=bool(p["synthetic"]))
            elif p["kind"] == "waste":
                plant = WasteStream(capacity=p["capacity"], cost=p["cost"], capacity_ratio=p["capacity_ratio"],
                                    source=_value(p["source"]), mode=_value(p["mode"], "flow"))
            elif p["pool"] == "operating":
                plant = SupplyPlant(p["capacity"], p["cost"], capex=_value(p["capex"]),
                                    years_in_pool=int(p["years_in_pool"]), synthetic=bool(p["synthetic"]))
            else:
                plant = SupplyPlant(p["capacity"], p["cost"] - intensity*ref, intensity=intensity,
                                    capex=_value(p["capex"]), years_in_pool=int(p["years_in_pool"]),
                                    synthetic=bool(p["synthetic"]))
            pools.setdefault(p["pool"], []).append(plant)

        operating = pools.pop("operating", [])
        eta = float(row["eta"])
        if mining:
            return MiningStage(SupplyCurve(operating, eta=eta), demand, name=name,
                               capex_intensity=_value(row["capex_intensity"]), pools=pools)
        supply = (SupplyCurve2D(operating, intensity=intensity, p_u_ref=ref, eta=eta) if not _missing(row["parent"])
                  else SupplyCurve(operating, eta=eta))
        return ProcessingStage(supply, demand, name=name, capex_intensity=float(row["capex_intensity"]), pools=pools)

    root = next(name for name, row in stages.items() if _missing(row["parent"]))
    return SupplyChain([build(root)], params=params)


# ---------------------------------------------------------------------------
# Excel input and output. pandas and openpyxl are imported here only, so the model runs without them.
# ---------------------------------------------------------------------------

def _overview():
    """The first sheet: one row per sheet, saying what it holds."""
    import pandas as pd
    rows = [{"sheet": "Sheets", "contents": "This list."},
            {"sheet": "Definitions", "contents": "Every column of every sheet: its default (used when a cell is blank) "
                                                 "and its meaning and units."}]
    rows += [{"sheet": title, "contents": contents} for title, contents in TITLES.values()]
    return pd.DataFrame(rows)


def _definitions():
    """The second sheet: one row per column of every input sheet, with its default and meaning."""
    import pandas as pd
    rows = []
    for table, columns in SHEETS.items():
        if table == "plants":
            sheets = [(f"{PLANTS_PREFIX}<{kind} stage>", PLANT_SHEET_COLUMNS[kind]) for kind in PLANT_SHEET_COLUMNS]
        else:
            sheets = [(TITLES[table][0], list(columns))]
        for title, names in sheets:
            rows += [{"sheet": title, "column": name, "default": "" if _missing(columns[name][0]) else columns[name][0],
                      "description": columns[name][1]} for name in names]
    return pd.DataFrame(rows)


def _plant_sheet_name(stage):
    """The sheet name of a stage's plants, checked against Excel's rules for sheet names."""
    name = f"{PLANTS_PREFIX}{stage}"
    if len(name) > 31 or any(c in name for c in r":\/?*[]"):
        raise ValueError(f"Stage name {stage!r} can't make a sheet name {name!r}: Excel sheet names have at most "
                         "31 characters and none of : \\ / ? * [ ].")
    return name


def _write_tables(path, tables):
    """Write input tables: the Sheets and Definitions sheets, Stages, one plant sheet per stage (or two
    placeholders when there are no stages), then the other tables."""
    import pandas as pd

    def frame(table, columns):
        return pd.DataFrame({column: table[column] for column in columns})

    stage_types = dict(zip(tables["stages"]["stage"], tables["stages"]["type"]))
    plants = tables["plants"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _overview().to_excel(writer, sheet_name="Sheets", index=False)
        _definitions().to_excel(writer, sheet_name="Definitions", index=False)
        frame(tables["stages"], SHEETS["stages"]).to_excel(writer, sheet_name=TITLES["stages"][0], index=False)
        sheets = ([(_plant_sheet_name(stage), kind, stage) for stage, kind in stage_types.items()] if stage_types else
                  [(f"{PLANTS_PREFIX}<{kind} stage>", kind, None) for kind in PLANT_SHEET_COLUMNS])
        for name, kind, stage in sheets:
            rows = plants["stage"] == stage
            frame({column: plants[column][rows] for column in plants}, PLANT_SHEET_COLUMNS[kind]).to_excel(
                writer, sheet_name=name, index=False)
        for table in ("stage_summaries", "demand_sectors", "sector_intensities", "recycling_technologies", "parameters"):
            frame(tables[table], SHEETS[table]).to_excel(writer, sheet_name=TITLES[table][0], index=False)


def write_template(path):
    """Write an empty input workbook: the Sheets and Definitions sheets, every input sheet with its
    column headers (two placeholder plant sheets, for a mining and a processing stage), and the
    Parameters sheet with every ModelParams field at its default.

    Args:
        path (str or Path): The .xlsx file to write.
    """
    tables = {sheet: _empty(sheet) for sheet in SHEETS}
    tables["parameters"] = _table(_parameter_rows(ModelParams()), "parameters")
    _write_tables(path, tables)


def write_inputs(model_input, path):
    """Write ModelInput tables to an input workbook: one sheet per table, with the plants split into one
    sheet per stage, after the Sheets and Definitions sheets.

    Args:
        model_input (ModelInput): The inputs.
        path (str or Path): The .xlsx file to write.
    """
    _write_tables(path, {sheet: getattr(model_input, sheet) for sheet in SHEETS})


def _rows_from_frame(frame, title, sheet, columns):
    """The rows of a sheet read by pandas: blank cells take the column's default, and each column takes
    the type of its default (text, number, whole number or true/false)."""
    unknown = [c for c in frame.columns if c not in columns]
    if unknown:
        raise ValueError(f"{title}: unknown column(s) {unknown}; the sheet's columns are {list(columns)}.")
    rows = []
    for record in frame.to_dict("records"):
        row = {}
        for column in columns:
            default = SHEETS[sheet][column][0]
            value = record.get(column, default)
            if _missing(value):
                value = default
            elif isinstance(default, bool):
                value = value.strip().lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
            elif isinstance(default, str):
                value = str(value).strip()
            elif isinstance(default, int):
                value = int(value)
            else:
                value = float(value)
            row[column] = value
        rows.append(row)
    return rows


def load_inputs(path, params=None, seed=None):
    """Read an input workbook into ModelInput tables, fill the pools that Stage summaries describes
    (fill_from_summaries), and validate the result (validate_inputs).

    Missing sheets are empty, and empty plant sheets are ignored; a sheet with an unknown name or an
    unknown column raises. Each "Plants - <stage>" sheet becomes that stage's rows of the plants table
    (kind "mine" on a mining stage, "plant" where a processing stage's kind is blank).

    Args:
        path (str or Path): The .xlsx file.
        params (ModelParams, optional): Parameters for the stage-summary fill. Defaults to the workbook's
            Parameters sheet.
        seed (int, optional): Seed of the stage-summary fill. Defaults to None.

    Returns:
        ModelInput: The inputs.
    """
    import pandas as pd
    frames = pd.read_excel(path, sheet_name=None)
    by_title = {title: table for table, (title, _) in TITLES.items() if table != "plants"}
    unknown = [name for name in frames if name not in by_title and name not in ("Sheets", "Definitions")
               and not name.startswith(PLANTS_PREFIX)]
    if unknown:
        raise ValueError(f"Unknown sheet(s) {unknown}; the input sheets are {list(by_title)} and "
                         f"'{PLANTS_PREFIX}<stage>' sheets.")

    tables = {table: _table(_rows_from_frame(frames[title], title, table, SHEETS[table]) if title in frames else [], table)
              for title, table in by_title.items()}
    stage_types = dict(zip(tables["stages"]["stage"], tables["stages"]["type"]))
    plants = []
    for name, frame in frames.items():
        if not name.startswith(PLANTS_PREFIX) or frame.dropna(how="all").empty:
            continue
        stage = name[len(PLANTS_PREFIX):]
        if stage not in stage_types:
            raise ValueError(f"{name}: {stage!r} is not a stage in Stages.")
        kind = stage_types[stage] if stage_types[stage] in PLANT_SHEET_COLUMNS else "processing"
        for row in _rows_from_frame(frame.dropna(how="all"), name, "plants", PLANT_SHEET_COLUMNS[kind]):
            row["stage"] = stage
            row["kind"] = "mine" if kind == "mining" else (row["kind"] or "plant")
            plants.append(row)
    model_input = ModelInput(plants=_table(plants, "plants"), **tables)
    params = params if params is not None else params_from_inputs(model_input)
    model_input = fill_from_summaries(model_input, params, seed=seed)
    validate_inputs(model_input)
    return model_input


def write_results(chain, path):
    """Write a run's results to a workbook: sheets Prices (prices_df), Plants (records_df), Demand
    (demand_df, with end of life and the backcast years) and Parameters (the run's parameters).

    Args:
        chain (SupplyChain): A chain that has been run.
        path (str or Path): The .xlsx file to write.
    """
    import pandas as pd
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        chain.prices_df().to_excel(writer, sheet_name="Prices")
        chain.records_df().to_excel(writer, sheet_name="Plants", index=False)
        chain.demand_df().to_excel(writer, sheet_name="Demand", index=False)
        pd.DataFrame(_parameter_rows(chain.params)).to_excel(writer, sheet_name="Parameters", index=False)
