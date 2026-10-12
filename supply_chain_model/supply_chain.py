import copy

from .utils import build_chain, solve_chain, SupplyCurve2D
from .supply_chain_stages import MiningStage


class SupplyChain:
    def __init__(self, input_stages, initial_year=2026, verbosity=0):
        """Simulates a supply chain with various supply stages througout time.

        The stages are mutable and advanced in place each year. The input stages are deepcopied on the
        way in, so two chains built from the same objects stay independent of each other and of the
        originals.

        Args:
            input_stages (array): List of SupplyChainStages
            initial_year (int, optional): Year to start simulations for. Defaults to 2026.
            verbosity (int, optional): Print the year as it runs if above 0. Defaults to 0.
        """
        # One deepcopy call over the whole list, so the copies share one memo and the graph
        # (parent links, CompositeDemand children) stays wired up inside the copy
        own_stages = copy.deepcopy(list(input_stages))

        self.stages, self.parent, self.children = build_chain(own_stages[0])
        self.initial_year = initial_year
        self.mine_idx = [k for k, s in enumerate(self.stages) if isinstance(s, MiningStage)]
        self.plant_idx = [k for k, s in enumerate(self.stages) if not isinstance(s, MiningStage)]

        # Name -> index map. Stage order is stable across years, so this resolves names for every snapshot.
        self.stage_idx = {}
        for k, s in enumerate(self.stages):
            if s.name in self.stage_idx:
                raise ValueError(f"Duplicate stage name {s.name!r}; stage names must be unique.")
            self.stage_idx[s.name] = k

        # Filled in as the run goes: deepcopy snapshots, prices by name, one record row per plant per year
        self.stages_history = {}
        self.prices = {}
        self.records = []

        self._current_year = None
        self.verbosity = verbosity

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

    def prices_df(self):
        """The equilibrium prices as a DataFrame indexed by year, one column per stage."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.prices, orient="index").rename_axis("year")

    def solve_prices(self):
        """Solve for the equilibrium prices of the live stages and write them onto the stages."""

        empty = [s.name for s in self.stages if len(s.supply.plants) == 0]
        if empty:
            raise ValueError(
                f"Stage(s) {empty} have no plants left, so there is no supply curve to solve. "
                "For a MiningStage this usually means every mine retired and the pipeline did not "
                "open anything to replace them.")

        # Warm start from last year's prices when we have them. Once the chain has run a while these
        # sit much closer to the answer than the per-stage cost means the solver would otherwise use.
        p0 = [s.p_eq for s in self.stages]
        p0 = p0 if all(p is not None for p in p0) else None

        res = solve_chain(self.stages, self.parent, self.children, p0=p0)
        for node, pi in res.items():
            node.p_eq = pi

    def run(self, n_iterations, scenario=None):
        """Run the supply chain simulation for a given number of iterations (years).

        Args:
            n_iterations (int): Number of years to simulate after the initial one.
            scenario (callable, optional): Scenario callback ``scenario(chain, t)`` invoked each year after that
                year's stages are built but before prices are solved. Defaults to None (baseline run).
        """

        for i in range(n_iterations + 1):
            t = self.initial_year + i
            self._current_year = t
            if self.verbosity > 0: print(f"Year {t}")

            # The initial year is whatever was passed in; later years are stepped in place from it
            if i > 0:
                # Built once, eagerly, from last year's solved chain. Stages may only read this, never
                # each other, so it doesn't matter what order they get stepped in.
                ctx = StageContext(self.stages)
                for stage in self.stages:
                    stage.step(ctx)

            # Scenario hook can mutate this year's stages before they're solved
            if scenario is not None:
                scenario(self, t)

            self.solve_prices()

            # Freeze the year: one deepcopy call so the snapshot's internal graph stays consistent
            self.stages_history[t] = copy.deepcopy(self.stages)
            self._record(t)

    def _record(self, t):
        """Write this year's prices and one row per plant into the flat records list.

        Args:
            t (int): Year being recorded.
        """
        self.prices[t] = {s.name: s.p_eq for s in self.stages}

        for stage in self.stages:
            produced = {p.uid: float(q) for p, q in
                        zip(stage.supply.plants, _plant_output(stage).ravel())}

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
                        "ore_grade": getattr(plant, "ore_grade", None),
                        "years_unprofitable": getattr(plant, "years_unprofitable", 0),
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
    def __init__(self, stages):
        """Read-only view of the just-solved chain, so a plant's step() can look up global info by name.

        Everything is worked out up front. The stages get mutated one by one afterwards, so anything
        read lazily would see a half-updated chain.

        Args:
            stages (list): The current year's solved SupplyChainStages (each with p_eq set).
        """
        self._prices = {s.name: s.p_eq for s in stages}

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

    def production_of(self, name):
        """This year's total production of the named stage."""
        return self._production[name]

    def stage_name_of(self, plant):
        """Name of the stage the given plant belongs to."""
        return self._stage_name[plant.uid]

    def consumed_by(self, plant):
        """This plant's own realized output at equilibrium (what it actually produced/consumed last year)."""
        return self._produced.get(plant.uid, 0.0)
