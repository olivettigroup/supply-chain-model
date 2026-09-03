import itertools

import numpy as np

from utils import CompositeDemand, SupplyCurve2D, find_intersection

YEARS_UNPROFITABLE_THRESHOLD = 3
CHANGE = 0.01


class SupplyPlant:
    # Shared counter so every plant ever built gets a different uid
    _uid_counter = itertools.count()

    def __init__(self, capacity, cost, intensity=None, uid=None):
        """Generic type for a plant that produces a material.

        Args:
            capacity (float): Avaialble capacity.
            cost (float): Marginal cost
            intensity (float, optional): How much of the upstream material is needed for production. Defaults to None.
            uid (int, optional): Stable identifier, kept for the plant's whole life so it can be tracked
                across years in the chain's records. Defaults to the next value of the shared counter.
        """
        self.uid = next(SupplyPlant._uid_counter) if uid is None else uid
        self.capacity = capacity
        self.cost = cost
        self.intensity = intensity

    @classmethod
    def reset_uids(cls):
        """Restart the uid counter. Call before building a chain if you want uids to be reproducible."""
        SupplyPlant._uid_counter = itertools.count()

    def step(self, ctx, pool="operating"):
        """Advance this plant by one year, in place.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
            pool (str): Name of the pool the plant currently sits in.
        """
        self.cost /= (1 - CHANGE)


class Mine(SupplyPlant):
    def __init__(self, capacity, cost, ore_grade=None, years_unprofitable=0, years_in_pool=0, uid=None):
        """Subtype of plant specifically for a mining stage.

        Args:
            capacity (float): Avaialble capacity.
            cost (float): Marginal cost
            ore_grade (float, optional): Ore grade of the mine. Defaults to None.
            years_unprofitable (int): Number of years the mine has been unprofitable.
            years_in_pool (int): Number of years the mine has been in a given pool.
            uid (int, optional): Stable identifier. Defaults to the next value of the shared counter.
        """
        super().__init__(capacity, cost, uid=uid)
        self.ore_grade = ore_grade
        self.years_unprofitable = years_unprofitable
        self.years_in_pool = years_in_pool

    def step(self, ctx, pool="operating"):
        """Advance this mine by one year, in place.

        years_in_pool is deliberately not touched here: MinePipeline.step owns the ageing so there is
        exactly one place where it happens, for every pool at once.
        """
        self.cost /= (1 - CHANGE)

        # Only a producing mine depletes its reserves
        if pool == "operating":
            self.capacity *= (1 - CHANGE)


class WasteStream(SupplyPlant):
    def __init__(self, capacity, cost, capacity_ratio=None, source=None, mode="flow", stock=None, uid=None):
        """Subtype of plant specifically for a waste stream stage.

        Args:
            capacity (float): Available capacity. In stock mode this is the current reservoir.
            cost (float): Marginal cost.
            capacity_ratio (float): Fraction of the source stage's production that becomes collectable scrap.
            source (str, optional): Name of the stage whose production feeds this stream. By default it is the stream's own stage.
            mode (str): "flow" (capacity = this year's inflow) or "stock" (capacity = running reservoir).
            stock (float, optional): Current reservoir size (stock mode only). Defaults to the initial capacity.
            uid (int, optional): Stable identifier. Defaults to the next value of the shared counter.
        """
        # Recycled feedstock: doesn't consume the upstream input by default, so its cost is independent of the upstream price
        super().__init__(capacity, cost, intensity=0, uid=uid)
        self.capacity_ratio = capacity_ratio
        self.source = source
        self.mode = mode
        self.stock = capacity if stock is None else stock

    def step(self, ctx, pool="operating"):
        """Advance this waste stream by one year, in place."""

        # Inflow feeds off the source stage's production; source=None means this stream's own stage
        source = self.source if self.source is not None else ctx.stage_name_of(self)
        inflow = self.capacity_ratio * ctx.production_of(source)

        if self.mode == "stock":
            # Reservoir grows by inflow, shrinks by what this plant actually recycled last year
            self.stock = max(self.stock + inflow - ctx.consumed_by(self), 0.0)
            self.capacity = self.stock
        else:
            self.capacity = inflow

        self.cost /= (1 - CHANGE)


# ---------------------------------------------------------------------------
# Pool machinery
#
# Two different things that are easy to mix up:
#   - Pool / Transition / MinePipeline describe the RULES. They are the shape of the pipeline
#     and stay the same every year.
#   - stage.pools is a plain dict {pool name: list of plants} and holds the MEMBERSHIP.
# ---------------------------------------------------------------------------

class Transition:
    def __init__(self, select, to, name=None):
        """One rule for moving plants out of a pool and into another.

        Args:
            select (callable): select(plants, ctx) -> list of the plants that should move. It gets the
                whole list rather than one plant at a time so that rules based on the group (cumulative
                capacity against a gap, say) are expressible. Use per_mine() to build one out of a
                simple per-plant test.
            to (str): Name of the destination pool.
            name (str, optional): Label used in error messages. Defaults to "-> <to>".
        """
        self.select = select
        self.to = to
        self.name = name if name is not None else f"-> {to}"


class Pool:
    def __init__(self, name, transitions=(), frozen=False):
        """One pool in a pipeline, and the rules for leaving it.

        Args:
            name (str): Pool name, e.g. "incentive". The pool called "operating" is the one whose
                plants make up the stage's supply curve.
            transitions (iterable): Transitions out of this pool, tried in order.
            frozen (bool): If True the plants in this pool don't age and aren't stepped. Used for pools
                that are only a record of what happened, like "retired".
        """
        self.name = name
        self.transitions = list(transitions)
        self.frozen = frozen


class MinePipeline:
    def __init__(self, pools):
        """An ordered chain of pools plus the rules for moving between them.

        Pools are listed upstream-first (furthest from production first). Stepping happens in the
        opposite direction so a plant can only advance one pool per year.

        Args:
            pools (iterable): Pool objects, upstream-first.
        """
        self.pools = list(pools)

        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate pool names in pipeline: {names}")
        if "operating" not in names:
            raise ValueError("A pipeline needs a pool called 'operating' to feed the supply curve.")
        for pool in self.pools:
            for transition in pool.transitions:
                if transition.to not in names:
                    raise ValueError(f"Pool {pool.name} has a transition to unknown pool {transition.to}.")

    @property
    def names(self):
        """Pool names, upstream-first."""
        return [pool.name for pool in self.pools]

    def pool(self, name):
        """Get a Pool by name."""
        for pool in self.pools:
            if pool.name == name:
                return pool
        raise KeyError(name)

    def empty_pools(self):
        """A fresh membership dict with one empty list per pool."""
        return {name: [] for name in self.names}

    def step(self, pools, ctx):
        """Age every plant by a year and run all the transitions once.

        Args:
            pools (dict): Membership dict, mutated in place.
            ctx (PoolContext): What the transition rules get to look at.

        Returns:
            dict: The same membership dict.
        """

        # Age everything first, in one place, so that the "n years in this pool" rules can ever fire
        for pool in self.pools:
            if pool.frozen:
                continue
            for plant in pools[pool.name]:
                plant.years_in_pool += 1

        # Walk the chain backwards (closest to production first) so nothing skips a pool in one year
        for pool in reversed(self.pools):
            for transition in pool.transitions:
                movers = list(transition.select(pools[pool.name], ctx))
                if not movers:
                    continue

                # Rebuild the source list rather than removing from it while iterating over it
                moved = {plant.uid for plant in movers}
                pools[pool.name] = [plant for plant in pools[pool.name] if plant.uid not in moved]

                for plant in movers:
                    plant.years_in_pool = 0
                pools[transition.to].extend(movers)

        return pools


class PoolContext:
    def __init__(self, stage, ctx=None, year=None):
        """What a transition rule is allowed to look at.

        Args:
            stage (MiningStage): The stage being stepped.
            ctx (StageContext, optional): Read-only view of last year's solved chain.
            year (int, optional): Current year.
        """
        self.stage = stage
        self.ctx = ctx
        self.year = year
        self.p_eq = stage.p_eq
        self.pools = stage.pools

    def capacity_of(self, pool_name):
        """Total capacity sitting in one pool."""
        return sum(plant.capacity for plant in self.pools[pool_name])

    @property
    def incentive_target(self):
        """Capacity the incentive pool is meant to hold, from the stage's incentive ratio."""
        return self.stage.incentive_ratio * self.capacity_of("operating")

    @property
    def supply_gap(self):
        """How much capacity is missing to meet demand at the current price.

        TODO: placeholder, replace with the real demand-minus-supply calculation.
        """
        return 0.0


# ---------------------------------------------------------------------------
# Transition rules
# ---------------------------------------------------------------------------

def per_mine(predicate):
    """Turn a one-plant-at-a-time test into a selector.

    Args:
        predicate (callable): predicate(plant, ctx) -> bool.

    Returns:
        callable: A selector suitable for a Transition.
    """
    def select(plants, ctx):
        return [plant for plant in plants if predicate(plant, ctx)]
    return select


def after_years(n):
    """Selector: move every plant that has spent at least n years in the pool."""
    return per_mine(lambda plant, ctx: plant.years_in_pool >= n)


def cheaper_than_price(plants, ctx):
    """PLACEHOLDER selector: move every mine whose cost sits below the current price.

    TODO: swap for the NPV calculation.
    """
    return [plant for plant in plants if plant.cost < ctx.p_eq]


def persistently_unprofitable(plants, ctx):
    """Selector: move every mine that has been losing money for long enough to shut."""
    return [plant for plant in plants
            if plant.cost > ctx.p_eq and plant.years_unprofitable >= YEARS_UNPROFITABLE_THRESHOLD]


def fill_capacity_gap(gap_fn):
    """Selector: take the cheapest plants until their combined capacity covers a gap.

    Args:
        gap_fn (callable): gap_fn(ctx) -> float, the capacity that needs filling.

    Returns:
        callable: A selector suitable for a Transition.
    """
    def select(plants, ctx):
        gap = gap_fn(ctx)
        if gap <= 0 or len(plants) == 0:
            return []

        ordered = sorted(plants, key=lambda plant: plant.cost)
        capacities = np.array([plant.capacity for plant in ordered], dtype=float)

        # Capacity already committed before each plant, so the one straddling the gap is included
        committed = np.cumsum(capacities) - capacities
        return [plant for plant, before in zip(ordered, committed) if before <= gap]
    return select


def default_mine_pipeline():
    """The standard mine pipeline: resource -> developing -> incentive -> opening -> operating.

    Returns:
        MinePipeline: A fresh pipeline. Build your own and hand it to MiningStage to use a different
        number of pools, or different rules between them.
    """
    return MinePipeline([
        # TODO: placeholder rule, swap for the supply-gap calculation
        Pool("resource", [Transition(cheaper_than_price, to="developing")]),
        Pool("developing", [Transition(after_years(4), to="incentive")]),
        # TODO: placeholder rule, swap for the NPV calculation
        Pool("incentive", [Transition(cheaper_than_price, to="opening")]),
        Pool("opening", [Transition(after_years(2), to="operating")]),
        Pool("operating", [Transition(persistently_unprofitable, to="retired")]),
        Pool("retired", frozen=True),
    ])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class SupplyChainStage():
    def __init__(self, supply, demand, name):
        """Representation of one supply chain stage.

        Stages are mutable: the graph is built once, and every year each stage is advanced in place by
        step(). Past years are kept as deepcopy snapshots by SupplyChain.

        Args:
            supply (callable): Supply for that stage.
            demand (callable): Demand for that stage.
            name (str): Human-readable stage name (e.g. "midstream"). Used so plants can reference
                another stage's production by name.
        """

        self.supply = supply
        self.demand = demand
        self.name = name
        self.parent = None
        if isinstance(demand, CompositeDemand):
            for downstream in demand.downstreams:
                downstream.parent = self
        self.p_eq = None

    def all_pools(self):
        """Membership view used for recording.

        A plain stage has no pipeline, so everything it owns is producing.

        Returns:
            dict: {pool name: list of plants}.
        """
        return {"operating": list(self.supply.plants)}

    def step(self, ctx):
        """Advance the stage by one year, in place.

        Only reads its own state and ctx, never another live stage, so the order the stages are
        stepped in doesn't matter.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot step stage.")

        self._step_supply(ctx)
        self._step_demand()

    def set_plants(self, plants):
        """Replace this stage's plants, rebuilding the supply curve's cached arrays.

        Args:
            plants (list): The stage's new full plant list.
        """
        self.supply.set_plants(plants)

    def add_plants(self, new_plants):
        """Append plants to this stage, rebuilding the supply curve so its cached arrays stay consistent.

        Args:
            new_plants (list): Plants to add to the stage's supply.
        """
        self.supply.add_plants(new_plants)

    def _step_supply(self, ctx):
        """Advance every plant, follow the parent's price, then refresh the cached arrays."""

        for plant in self.supply.plants:
            plant.step(ctx)

        # The costs in a 2D curve are quoted against an upstream reference price that moves each year
        if isinstance(self.supply, SupplyCurve2D) and self.parent is not None:
            self.supply.p_u_ref = ctx.price_of(self.parent.name)

        self.supply.refresh()

    def _step_demand(self):
        """Advance demand, in place."""

        # A CompositeDemand already points at the live child stages, and those get stepped themselves,
        # so there is nothing to rebuild here
        if isinstance(self.demand, CompositeDemand):
            return

        # TODO: make demand curve update more sophisticated
        # If the stage is a leaf stage (all the way downstream), then we update the demand curve
        self.demand.p_nom = self.p_eq
        self.demand.q_nom *= (1 + CHANGE)
        self.demand.refresh()

    def get_equilibrium_price(self, p_u=None):
        """Find the equilibrium price (intersection of supply and demand) for this stage. At an upstream price p_u if need be.

        Args:
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            float: Equilibrium price.
        """

        if isinstance(self.supply, SupplyCurve2D):
            costs = self.supply.costs_at(p_u)
            s = lambda p: self.supply(p, p_u=p_u)
        else:
            costs = self.supply.costs
            s = self.supply
        bracket = (max(costs.min() * 0.1, 1e-12), costs.max() * 10.0)
        return find_intersection(s, self.demand, bracket=bracket)

    def plot_sd(self, ax, p_u=None):
        """Helpful function to plot the supply and demand curves for the stage.

        Args:
            ax (pyplot ax): Axes for the figure
            p_u (float, optional): Upstream price. Defaults to None.
        """

        if isinstance(self.supply, SupplyCurve2D):
            y = np.linspace(1e-20, 1.5*self.supply.costs_at(p_u).max(), 200)
            ax.plot(self.supply(y, p_u=p_u), y, lw=2, zorder=3, label="Supply")
        else:
            y = np.linspace(1e-20, 1.5*self.supply.costs.max(), 200)
            ax.plot(self.supply(y), y, lw=2, zorder=3, label="Supply")

        x_lims = ax.get_xlim()
        ax.plot(self.demand(y[::4]), y[::4], lw=2, zorder=3, label="Demand")

        ax.set_xlim(x_lims)
        ax.grid()

        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')


class MiningStage(SupplyChainStage):
    def __init__(self, supply, demand, name, pipeline=None, pools=None, incentive_ratio=None,
                 incentive_pool=None, resource_pool=None):
        """Subtype of a SupplyChainStage whose plants move through a chain of pools.

        The pools dict is the source of truth: the supply curve is rebuilt from pools["operating"] at
        the end of every step.

        Args:
            supply (callable): Supply for that stage. Its plants become the operating pool.
            demand (callable): Demand for that stage.
            name (str): Human-readable stage name.
            pipeline (MinePipeline, optional): The pools and the rules between them.
                Defaults to default_mine_pipeline().
            pools (dict, optional): Starting membership, {pool name: list of mines}. Any pool left out
                starts empty. Keys must be pool names that exist in the pipeline.
            incentive_ratio (float, optional): Capacity the incentive pool should hold, as a fraction of
                operating capacity. Defaults to whatever ratio the starting pools happen to have.
            incentive_pool (list, optional): Shorthand for pools={"incentive": ...}.
            resource_pool (list, optional): Shorthand for pools={"resource": ...}.
        """

        super().__init__(supply, demand, name)

        self.pipeline = pipeline if pipeline is not None else default_mine_pipeline()

        # The two shorthands are just entries in the same starting-membership dict
        starting = dict(pools or {})
        if incentive_pool is not None:
            starting["incentive"] = incentive_pool
        if resource_pool is not None:
            starting["resource"] = resource_pool

        self.pools = self.pipeline.empty_pools()
        for pool_name, mines in starting.items():
            if pool_name not in self.pools:
                raise ValueError(f"Unknown pool {pool_name}; this pipeline has {self.pipeline.names}.")
            self.pools[pool_name] = list(mines)

        # The supply curve's plants are the operating pool, always
        self.pools["operating"] = list(self.supply.plants)

        self.incentive_ratio = (incentive_ratio if incentive_ratio is not None
                                else self._measure_incentive_ratio())

    # Convenience read-only access, so stage.incentive_pool and friends still work
    @property
    def resource_pool(self):
        return self.pools["resource"]

    @property
    def developing_pool(self):
        return self.pools["developing"]

    @property
    def incentive_pool(self):
        return self.pools["incentive"]

    @property
    def opening_pool(self):
        return self.pools["opening"]

    @property
    def operating_pool(self):
        return self.pools["operating"]

    @property
    def retired_pool(self):
        return self.pools["retired"]

    def all_pools(self):
        """Membership view used for recording.

        Returns:
            dict: {pool name: list of mines}, including "operating".
        """
        return dict(self.pools)

    def add_plants(self, new_plants):
        """Add mines straight into production.

        Args:
            new_plants (list): Mines to add to the operating pool.
        """
        self.pools["operating"].extend(new_plants)
        self.set_plants(self.pools["operating"])

    def step(self, ctx):
        """Advance the stage by one year, in place.

        Order matters: every mine ages first, then we work out who is losing money, then mines move
        between pools, then the supply curve is rebuilt from whoever ended up operating.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot step stage.")

        for pool_name, mines in self.pools.items():
            if self.pipeline.pool(pool_name).frozen:
                continue
            for mine in mines:
                mine.step(ctx, pool=pool_name)

        self._update_profitability()

        self.pipeline.step(self.pools, PoolContext(self, ctx=ctx))

        # pools["operating"] is the truth, the supply curve just mirrors it
        self.set_plants(self.pools["operating"])

        self._step_demand()

    def _update_profitability(self):
        """Count how long each operating mine has been under water, resetting when it recovers.

        TODO: this is the simplest possible rule, change it if you want a smarter one.
        """
        for mine in self.pools["operating"]:
            if mine.cost > self.p_eq:
                mine.years_unprofitable += 1
            else:
                mine.years_unprofitable = 0

    def _measure_incentive_ratio(self):
        """Incentive pool capacity as a fraction of operating capacity.

        Zero if this pipeline has no pool called "incentive", which a custom one need not.
        """
        operating_capacity = self.supply.capacities.sum()
        if operating_capacity <= 0 or "incentive" not in self.pools:
            return 0.0
        return sum(mine.capacity for mine in self.pools["incentive"]) / operating_capacity
