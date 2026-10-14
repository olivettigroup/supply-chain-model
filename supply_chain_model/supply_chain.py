import copy
import warnings
from dataclasses import dataclass

import numpy as np

from .utils import build_chain, solve_chain, SupplyCurve2D, MIN_COST_FACTOR
from .supply_chain_stages import MiningStage, ProcessingStage


@dataclass(frozen=True)
class ModelParams:
    """Global model parameters. Values that differ by stage are stage attributes, not parameters.

    Attributes:
        start_year (int): First model year.
        end_year (int): Last model year.
        price_smoothing_weight (float): Weight on this year's equilibrium price in the decision price.
            1 switches smoothing off.
        grade_decline_rate (float): Annual ore-grade decline of producing mines, used by the default grade_fn.
        discount_rate (float): Real discount rate in the mine NPV.
        mine_life (int): Years of production counted in the mine NPV.
        feasibility_time (int): Years a mine spends in "developing".
        construction_time (int): Years a mine spends in "opening".
        default_reserve_life (float): Reserves of a mine whose data has none, in years of capacity.
        incentive_ratio (float): Target for incentive + developing capacity, as a fraction of the mining
            stage's production last year.
        closed_destination (str): Pool a closed mine returns to.
        closed_cooldown (int): Years a mine spends in "closed" before returning.
        plant_construction_time (int): Years a processing plant spends in "construction".
        plant_life (int): Years of production counted in the template plant NPV.
        plant_learning_rate (float): Annual fall in the conversion cost of existing processing plants
            (construction and operating; idle plants don't learn).
        technology_rate (float): Annual fall in the frontier conversion cost of new processing plants.
        expansion_cost_percentile (float): Percentile of the starting regular plants' conversion costs
            that sets the starting frontier.
        expansion_speed (float): Share of last year's operating capacity built per unit of the template
            plant's profitability index.
        max_expansion_share (float): Cap on the capacity built in a year, as a share of last year's
            operating capacity.
        max_idle_years (int): Years a processing plant can stay idle before it closes.
        lifetime_sd_fraction (float): A sector's lifetime standard deviation as a fraction of its mean
            lifetime, when the sector gives none.
        recycling (bool): Offer the sectors' secondary material as recycled supply at the processing
            stages they target. End-of-life flows are worked out either way.
        recycling_cost_percentile (float): Percentile of a target stage's regular operating plants'
            total costs in the first year that sets the cost of its recycled supply.
    """
    start_year: int = 2024
    end_year: int = 2050
    price_smoothing_weight: float = 0.34
    grade_decline_rate: float = 0.01
    discount_rate: float = 0.15
    mine_life: int = 20
    feasibility_time: int = 4
    construction_time: int = 2
    default_reserve_life: float = 25
    incentive_ratio: float = 0.5
    closed_destination: str = "developing"
    closed_cooldown: int = 1
    plant_construction_time: int = 2
    plant_life: int = 25
    plant_learning_rate: float = 0.005
    technology_rate: float = 0.015
    expansion_cost_percentile: float = 50
    expansion_speed: float = 0.5
    max_expansion_share: float = 0.15
    max_idle_years: int = 3
    lifetime_sd_fraction: float = 0.25
    recycling: bool = False
    recycling_cost_percentile: float = 25


class SupplyChain:
    def __init__(self, input_stages, params=None, verbosity=0):
        """Simulates a supply chain with various supply stages througout time.

        The stages are mutable and advanced in place each year. The input stages are deepcopied on the
        way in, so two chains built from the same objects stay independent of each other and of the
        originals.

        Args:
            input_stages (array): List of SupplyChainStages
            params (ModelParams, optional): Model parameters. Defaults to ModelParams().
            verbosity (int, optional): Print the year as it runs if above 0. Defaults to 0.
        """
        # One deepcopy call over the whole list, so the copies share one memo and the graph
        # (parent links, child stages and sectors) stays wired up inside the copy
        own_stages = copy.deepcopy(list(input_stages))

        self.params = params if params is not None else ModelParams()
        self.stages, self.parent = build_chain(own_stages[0])
        for stage in self.stages:
            stage.set_params(self.params)
        self.mine_idx = [k for k, s in enumerate(self.stages) if isinstance(s, MiningStage)]
        self.plant_idx = [k for k, s in enumerate(self.stages) if not isinstance(s, MiningStage)]

        # Name -> index map. Stage order is stable across years, so this resolves names for every snapshot.
        self.stage_idx = {}
        for k, s in enumerate(self.stages):
            if s.name in self.stage_idx:
                raise ValueError(f"Duplicate stage name {s.name!r}; stage names must be unique.")
            self.stage_idx[s.name] = k

        # Every sector once, however many stages it draws from
        self.sectors = []
        for stage in self.stages:
            for sector in stage.demand.sectors:
                if not any(sector is known for known in self.sectors):
                    self.sectors.append(sector)
        for sector in self.sectors:
            for name in sector.I0:
                if name not in self.stage_idx or not any(sector is d for d in self.stages[self.stage_idx[name]].demand.sectors):
                    raise ValueError(f"Sector {sector.name!r} draws from {name!r}, which is not a stage whose "
                                     "CompositeDemand lists the sector.")
        for sector in self.sectors:
            sector.set_params(self.params)
        if self.params.recycling:
            self._add_recycled_supply()

        # Filled in as the run goes: deepcopy snapshots, prices by name, one record row per plant per year,
        # one demand row per sector per drawn stage per year
        self.stages_history = {}
        self.prices = {}
        self.records = []
        self.demand_records = []

        self._current_year = None
        self.verbosity = verbosity

    def _secondary_inflows(self):
        """This year's secondary material per target stage: the sum over sectors and drawn stages of
        secondary(s, e) whose recycling target is that stage.

        Returns:
            dict: {target stage name: secondary material}.
        """
        inflows = {}
        for sector in self.sectors:
            for stage, amount in sector.secondary.items():
                target = sector.recycling_targets[stage]
                inflows[target] = inflows.get(target, 0.0) + amount
        return inflows

    def _add_recycled_supply(self):
        """Give every stage that a sector's recycling targets name one recycled-supply plant, stocked
        with the first year's secondary material. Only processing stages can take recycled material.
        """
        inflows = self._secondary_inflows()
        for sector in self.sectors:
            for target in sector.recycling_targets.values():
                if target not in self.stage_idx:
                    raise ValueError(f"Sector {sector.name!r} sends recycled material to {target!r}, which is not a stage.")
                if not isinstance(self.stages[self.stage_idx[target]], ProcessingStage):
                    raise ValueError(f"Sector {sector.name!r} sends recycled material to {target!r}; only "
                                     "processing stages can take recycled material.")
        for target in dict.fromkeys(t for sector in self.sectors for t in sector.recycling_targets.values()):
            self.stages[self.stage_idx[target]].add_recycled_supply(inflows.get(target, 0.0))

    def snapshot(self, t):
        """Stages for a given year.

        Returns the frozen copy for a finished year, and the live objects for the year currently being
        built, so a scenario can read last year and write this year.

        Args:
            t (int): Year.

        Returns:
            list: SupplyChainStages, in canonical order.
        """
        if t in self.stages_history:
            return self.stages_history[t]
        if t == self._current_year:
            return self.stages
        raise KeyError(f"No stages for year {t}; the run covers {sorted(self.stages_history)}.")

    def stage(self, name, t):
        """Get a stage by name for a given year."""
        return self.snapshot(t)[self.stage_idx[name]]

    @property
    def stage_names(self):
        """Stage names in canonical (snapshot) order."""
        return list(self.stage_idx)

    def plant_stages(self, t):
        """Handy helper to get the plant stages for a given year."""
        return [self.snapshot(t)[k] for k in self.plant_idx]

    def mine_stages(self, t):
        """Handy helper to get the mine stages for a given year."""
        return [self.snapshot(t)[k] for k in self.mine_idx]

    def records_df(self):
        """The per-plant records as a DataFrame.

        One row per plant per pool per year, so pool stats are a groupby and a single plant's history
        is a filter on its uid.

        Returns:
            pandas.DataFrame: The recorded rows.
        """
        import pandas as pd
        return pd.DataFrame(self.records)

    def demand_df(self):
        """Sector demand as a DataFrame: one row per sector per drawn stage per year.

        Columns: year, sector, stage, nominal_driver (V_nom), driver (V at the solved prices),
        sector_price (P at the solved prices), p_nom (P_nom), intensity, consumption (intensity *
        driver, what the sector took from the stage), eol and secondary (that year's end-of-life and
        secondary material), and backcast. Backcast rows (backcast = True) cover the years before the
        run of sectors with a lifetime, with only nominal_driver, intensity and consumption filled in.

        Returns:
            pandas.DataFrame: The recorded rows.
        """
        import pandas as pd
        return pd.DataFrame(self.demand_records)

    def demand_elasticities(self, step=1e-4):
        """Price elasticities of sector demand at the latest solved prices, with no re-solve.

        Entry (i, j) is the % change in the sector demand on stage i per 1% change in stage j's price,
        all other prices held: the diagonal holds each stage's own-price elasticity, the rest the
        coupling through shared sectors. Only stages that sectors draw from are included.

        Args:
            step (float, optional): Log price step of the central difference. Defaults to 1e-4.

        Returns:
            pandas.DataFrame: Rows are the demand stages, columns the stages whose price moves.
        """
        import pandas as pd
        names = [s.name for s in self.stages if s.demand.sectors]
        prices = {s.name: s.p_eq for s in self.stages}

        def log_demand(stage_name, at):
            return np.log(sum(sector.demand_on(stage_name, at) for sector in self.stages[self.stage_idx[stage_name]].demand.sectors))

        table = {}
        for j in names:
            up = {**prices, j: prices[j]*np.exp(step)}
            down = {**prices, j: prices[j]*np.exp(-step)}
            table[j] = [(log_demand(i, up) - log_demand(i, down))/(2*step) for i in names]
        return pd.DataFrame(table, index=names)

    def prices_df(self):
        """Prices as a DataFrame indexed by year: one column per stage with the equilibrium price,
        then one "<stage> (decision)" column per stage with the decision price."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.prices, orient="index").rename_axis("year")

    def solve_prices(self):
        """Solve for the equilibrium prices of the live stages and write them onto the stages.

        Warns if the MIN_COST_FACTOR cost floor binds at the solved prices, since the floor is only
        meant to keep the solver's trial prices well defined.
        """

        empty = [s.name for s in self.stages if len(s.supply.plants) == 0]
        if empty:
            raise ValueError(
                f"Stage(s) {empty} have no plants left, so there is no supply curve to solve. "
                "For a MiningStage this usually means every operating mine closed or ran out and the "
                "pipeline did not open anything to replace them.")

        # Warm start from last year's prices when we have them. Once the chain has run a while these
        # sit much closer to the answer than the per-stage cost means the solver would otherwise use.
        p0 = [s.p_eq for s in self.stages]
        p0 = p0 if all(p is not None for p in p0) else None

        res = solve_chain(self.stages, self.parent, p0=p0)
        for node, pi in res.items():
            node.p_eq = pi

        for s in self.stages:
            if isinstance(s.supply, SupplyCurve2D) and s.parent is not None:
                unfloored = s.supply.conversion_costs + s.supply.plant_intensities*s.parent.p_eq
                n_floored = int(np.sum(unfloored < MIN_COST_FACTOR*s.supply.costs))
                if n_floored:
                    warnings.warn(f"{self._current_year}: the MIN_COST_FACTOR cost floor binds for "
                                  f"{n_floored} plant(s) of stage {s.name!r} at the solved prices.")

    def _smooth_decision_prices(self):
        """Update every stage's decision price from the price just solved.

        P̄_t = w * p_eq_t + (1 - w) * P̄_{t-1}, with w = params.price_smoothing_weight. In the first year
        the decision price is p_eq.
        """
        w = self.params.price_smoothing_weight
        for s in self.stages:
            s.decision_price = s.p_eq if s.decision_price is None else w*s.p_eq + (1 - w)*s.decision_price

    def run(self, n_iterations=None, scenario=None):
        """Run the supply chain simulation from params.start_year.

        Args:
            n_iterations (int, optional): Number of years to simulate after the initial one. Defaults to
                running through params.end_year.
            scenario (callable, optional): Scenario callback ``scenario(chain, t)`` invoked each year after that
                year's stages are built but before prices are solved. Defaults to None (baseline run).
        """
        if n_iterations is None:
            n_iterations = self.params.end_year - self.params.start_year

        for i in range(n_iterations + 1):
            t = self.params.start_year + i
            self._current_year = t
            if self.verbosity > 0: print(f"Year {t}")

            # The initial year is whatever was passed in; later years are stepped in place from it
            if i > 0:
                # Built once, eagerly, from last year's solved chain. Stages may only read this, never
                # each other, so it doesn't matter what order they get stepped in.
                ctx = StageContext(self.stages, t)
                for sector in self.sectors:
                    sector.step(ctx, self.params)
                ctx.add_secondary(self._secondary_inflows())
                for stage in self.stages:
                    stage.step(ctx)

            # Scenario hook can mutate this year's stages before they're solved
            if scenario is not None:
                scenario(self, t)

            self.solve_prices()
            self._smooth_decision_prices()

            # Freeze the year: one deepcopy call so the snapshot's internal graph stays consistent
            self.stages_history[t] = copy.deepcopy(self.stages)
            self._record(t)

    def _record(self, t):
        """Write this year's prices, one row per sector per drawn stage, and one row per plant into the
        flat records lists.

        Args:
            t (int): Year being recorded.
        """
        prices = {s.name: s.p_eq for s in self.stages}
        self.prices[t] = dict(prices)
        self.prices[t].update({f"{s.name} (decision)": s.decision_price for s in self.stages})

        if t == self.params.start_year:
            for sector in self.sectors:
                for year in sorted(y for y in sector.history if y < t):
                    driver, intensities = sector.history[year]
                    for name, intensity in intensities.items():
                        self.demand_records.append({
                            "year": year, "sector": sector.name, "stage": name, "nominal_driver": driver,
                            "intensity": intensity, "consumption": intensity*driver, "backcast": True,
                        })
        for sector in self.sectors:
            driver = sector.volume(prices)
            for name, intensity in sector.intensities.items():
                self.demand_records.append({
                    "year": t, "sector": sector.name, "stage": name,
                    "nominal_driver": sector.V, "driver": driver,
                    "sector_price": sector.price(prices), "p_nom": sector.P_nom,
                    "intensity": intensity, "consumption": intensity*driver,
                    "eol": sector.eol[name], "secondary": sector.secondary[name], "backcast": False,
                })

        for stage in self.stages:
            output = _plant_output(stage).ravel()
            produced = {p.uid: float(q) for p, q in zip(stage.supply.plants, output)}

            for pool_name, plants in stage.all_pools().items():
                for plant in plants:
                    self.records.append({
                        "year": t,
                        "stage": stage.name,
                        "pool": pool_name,
                        "uid": plant.uid,
                        "capacity": plant.capacity,
                        "cost": plant.cost,
                        # Only the operating pool actually produces anything
                        "production": produced.get(plant.uid, 0.0),
                        "grade": getattr(plant, "grade", None),
                        "reserves": getattr(plant, "reserves", None),
                        "years_in_pool": getattr(plant, "years_in_pool", 0),
                    })


def _plant_output(stage, prices=None):
    """Per-plant output of a solved stage, at its own price and its parent's.

    Args:
        stage (SupplyChainStage): A stage with p_eq set.
        prices (dict, optional): {stage name: price}, used to look up the parent's price. Defaults to
            reading it off the parent stage.

    Returns:
        array: Quantity produced by each of the stage's plants.
    """
    if not isinstance(stage.supply, SupplyCurve2D):
        return stage.supply._plant_quantities(stage.p_eq)

    if stage.parent is None:
        p_u = None
    elif prices is not None:
        p_u = prices[stage.parent.name]
    else:
        p_u = stage.parent.p_eq

    return stage.supply._plant_quantities(stage.p_eq, p_u)


class StageContext:
    def __init__(self, stages, year):
        """Read-only view of the just-solved chain, so a plant's step() can look up global info by name.

        Everything is worked out up front. The stages get mutated one by one afterwards, so anything
        read lazily would see a half-updated chain.

        Args:
            stages (list): Last year's solved SupplyChainStages (each with p_eq set).
            year (int): The year being stepped into.
        """
        self.year = year
        self._secondary = {}
        self._prices = {s.name: s.p_eq for s in stages}
        self._decision_prices = {s.name: s.decision_price for s in stages}

        # Per-plant realized output at equilibrium, and which stage each plant belongs to, keyed by uid
        self._production = {}
        self._produced = {}
        self._stage_name = {}
        for s in stages:
            quantities = _plant_output(s, self._prices).ravel()
            self._production[s.name] = float(quantities.sum())
            for plant, q in zip(s.supply.plants, quantities):
                self._produced[plant.uid] = float(q)
                self._stage_name[plant.uid] = s.name

    def price_of(self, name):
        """This year's equilibrium price of the named stage."""
        return self._prices[name]

    def decision_price_of(self, name):
        """This year's decision price (smoothed equilibrium price) of the named stage."""
        return self._decision_prices[name]

    def production_of(self, name):
        """This year's total production of the named stage."""
        return self._production[name]

    def stage_name_of(self, plant):
        """Name of the stage the given plant belongs to."""
        return self._stage_name[plant.uid]

    def add_secondary(self, inflows):
        """Give the context this year's secondary material per target stage. Called by SupplyChain.run
        once the sectors have stepped, before any stage steps.

        Args:
            inflows (dict): {target stage name: secondary material}.
        """
        self._secondary = dict(inflows)

    def secondary_into(self, name):
        """This year's secondary material sent to the named stage (0 if none)."""
        return self._secondary.get(name, 0.0)

    def consumed_by(self, plant):
        """This plant's own realized output at equilibrium (what it actually produced/consumed last year)."""
        return self._produced.get(plant.uid, 0.0)
