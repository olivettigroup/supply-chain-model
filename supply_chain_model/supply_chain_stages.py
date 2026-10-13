import itertools

import numpy as np

from .utils import CompositeDemand, SupplyCurve2D, cheapest_covering, find_intersection

# Numerical guard: a mine whose remaining reserves fall below this fraction of its capacity is exhausted
EXHAUSTION_TOLERANCE = 1e-9


class SupplyPlant:
    # Shared counter so every plant ever built gets a different uid
    _uid_counter = itertools.count()

    def __init__(self, capacity, cost, intensity=None, capex=None, years_in_pool=0, synthetic=False, uid=None):
        """Generic type for a plant that produces a material.

        The plant's primitive cost is its conversion cost, the non-material cost per unit of output.
        Its total cost (the cost property) adds the upstream input at last year's upstream price. A
        plant with no upstream input (upstream_price = 0) has total cost = conversion cost.

        Args:
            capacity (float): Avaialble capacity.
            cost (float): Total marginal cost. When the plant is given to a SupplyCurve2D constructor or
                add_plants, it is taken as quoted at the curve's upstream price and split into
                conversion cost and input cost; otherwise it is the conversion cost.
            intensity (float, optional): How much of the upstream material is needed for production. Defaults to None.
            capex (float, optional): Total capex in currency. Only plants the model builds need one;
                an existing plant's capex is sunk and never used. Defaults to None.
            years_in_pool (int, optional): Full years spent in the current pool. Defaults to 0.
            synthetic (bool, optional): True for plants the model or the data generator created.
                Defaults to False.
            uid (int, optional): Stable identifier, kept for the plant's whole life so it can be tracked
                across years in the chain's records. Defaults to the next value of the shared counter.
        """
        self.uid = next(SupplyPlant._uid_counter) if uid is None else uid
        self.capacity = capacity
        self.intensity = intensity
        self.conversion_cost = cost
        self.upstream_price = 0.0
        self.capex = capex
        self.years_in_pool = years_in_pool
        self.synthetic = synthetic
        self.pool_history = []

    @property
    def cost(self):
        """Total cost per unit of output, at last year's upstream price."""
        return self.conversion_cost + (self.intensity or 0.0)*self.upstream_price

    @property
    def usable_capacity(self):
        """Capacity the supply curve can use this year."""
        return self.capacity

    @classmethod
    def reset_uids(cls):
        """Restart the uid counter. Call before building a chain if you want uids to be reproducible."""
        SupplyPlant._uid_counter = itertools.count()


class Mine(SupplyPlant):
    def __init__(self, capacity, cost, grade=1.0, recovery=1.0, reserves=None, capex=None, years_in_pool=0,
                 uid=None):
        """A deposit, described by its ore and its reserves.

        The primitives are ore_throughput, grade, recovery and ore_cost; capacity and cost are derived
        from them, so a change in grade moves both. Given only capacity and cost (grade and recovery 1),
        the mine behaves like a plain plant with that capacity and cost.

        Args:
            capacity (float): Nameplate capacity at the given grade and recovery (saleable product per year).
            cost (float): Cost per unit of saleable product at the given grade and recovery.
            grade (float, optional): Metal content of the ore. Defaults to 1.
            recovery (float, optional): Fraction of the contained metal that ends up as saleable product.
                Defaults to 1.
            reserves (float, optional): Saleable product still in the deposit. Defaults to
                params.default_reserve_life * capacity, filled in when the stage gets its parameters.
            capex (float, optional): Total capex in currency. Defaults to the stage's
                capex_intensity * capacity, filled in when the stage is built.
            years_in_pool (int, optional): Full years spent in the current pool. Defaults to 0.
            uid (int, optional): Stable identifier. Defaults to the next value of the shared counter.
        """
        self.uid = next(SupplyPlant._uid_counter) if uid is None else uid
        self.grade = grade
        self.recovery = recovery
        self.ore_throughput = capacity/(grade*recovery)
        self.ore_cost = cost*grade*recovery
        self.reserves = reserves
        self.capex = capex
        self.years_in_pool = years_in_pool
        # Set when the mine passes the NPV gate, and on every move between pools
        self.npv_at_decision = None
        self.pool_history = []

    @property
    def capacity(self):
        """Nameplate capacity: ore_throughput * grade * recovery."""
        return self.ore_throughput*self.grade*self.recovery

    @property
    def cost(self):
        """Cost per unit of saleable product: ore_cost / (grade * recovery)."""
        return self.ore_cost/(self.grade*self.recovery)

    @property
    def usable_capacity(self):
        """Nameplate capacity capped at the remaining reserves (uncapped while reserves are unknown)."""
        return self.capacity if self.reserves is None else min(self.capacity, self.reserves)


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

    def step(self, ctx):
        """Waste-stream capacity: set this year's capacity from the source stage's production last year.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """

        # Inflow feeds off the source stage's production; source=None means this stream's own stage
        source = self.source if self.source is not None else ctx.stage_name_of(self)
        inflow = self.capacity_ratio * ctx.production_of(source)

        if self.mode == "stock":
            # Reservoir grows by inflow, shrinks by what this plant actually recycled last year
            self.stock = max(self.stock + inflow - ctx.consumed_by(self), 0.0)
            self.capacity = self.stock
        else:
            self.capacity = inflow


# ---------------------------------------------------------------------------
# Pool machinery
#
# Pipeline describes the pools and moves plants between them; the stage's step() says which moves
# happen in which order. stage.pools is a plain dict {pool name: list of plants} and holds the
# membership.
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, names, frozen=()):
        """The pools a stage's plants move through.

        Args:
            names (iterable): Pool names, upstream first. The pool called "operating" makes up the
                stage's supply curve.
            frozen (iterable): Pools whose plants never age or move again.
        """
        self.names = list(names)
        self.frozen = set(frozen)
        if "operating" not in self.names:
            raise ValueError("A pipeline needs a pool called 'operating' to feed the supply curve.")
        if not self.frozen <= set(self.names):
            raise ValueError(f"Frozen pools {sorted(self.frozen - set(self.names))} are not in the pipeline.")

    def empty_pools(self):
        """A fresh membership dict with one empty list per pool."""
        return {name: [] for name in self.names}

    def move(self, pools, from_pool, to_pool, rule, ctx, moved):
        """Run one transition: move the plants that the rule selects from one pool to another.

        Plants that already moved in this step are not offered to the rule, so no plant moves twice in
        a year. Movers get years_in_pool = 0 and a (year, to_pool) entry in their pool_history.

        Args:
            pools (dict): Membership dict, mutated in place.
            from_pool (str): Pool to move plants out of.
            to_pool (str): Pool to move them into.
            rule (callable): rule(plants, ctx) -> list of the plants that move.
            ctx (PoolContext): What the rule gets to look at.
            moved (set): uids of plants that moved earlier in this step; updated in place.
        """
        if to_pool not in pools:
            raise ValueError(f"Unknown pool {to_pool!r}; this pipeline has {self.names}.")

        movers = rule([plant for plant in pools[from_pool] if plant.uid not in moved], ctx)
        if not movers:
            return

        uids = {plant.uid for plant in movers}
        pools[from_pool] = [plant for plant in pools[from_pool] if plant.uid not in uids]
        for plant in movers:
            plant.years_in_pool = 0
            plant.pool_history.append((ctx.year, to_pool))
        pools[to_pool].extend(movers)
        moved |= uids

    def age(self, pools, moved):
        """Ageing: add a year to every plant in a non-frozen pool, except plants that moved in this step,
        so years_in_pool always counts full years in the current pool.

        Args:
            pools (dict): Membership dict.
            moved (set): uids of plants that moved earlier in this step.
        """
        for name in self.names:
            if name in self.frozen:
                continue
            for plant in pools[name]:
                if plant.uid not in moved:
                    plant.years_in_pool += 1


class PoolContext:
    def __init__(self, stage, ctx):
        """What a transition rule or price_path_fn is allowed to look at.

        Args:
            stage (MiningStage): The stage being stepped.
            ctx (StageContext): Read-only view of last year's solved chain.
        """
        self.stage = stage
        self.ctx = ctx
        self.params = stage.params
        self.year = ctx.year
        # Last year's values for this stage, and its parent's decision price (0 for a stage with no parent)
        self.decision_price = ctx.decision_price_of(stage.name)
        self.production = ctx.production_of(stage.name)
        self.parent_decision_price = 0.0 if stage.parent is None else ctx.decision_price_of(stage.parent.name)

    def capacity_of(self, pool_name):
        """Total nameplate capacity sitting in one pool."""
        return sum(plant.capacity for plant in self.stage.pools[pool_name])


# ---------------------------------------------------------------------------
# Mine economics: grade path, price path, NPV
# ---------------------------------------------------------------------------

def default_grade_fn(grade, utilisation, params):
    """Default grade path: grade falls by params.grade_decline_rate per year of production at full
    capacity, in proportion to how much of its capacity the mine used.

    Args:
        grade (float): This year's grade.
        utilisation (float): Last year's production as a fraction of nameplate capacity. It can be
            slightly above 1, since the supply curve lets output exceed capacity when price > cost.
        params (ModelParams): Model parameters.

    Returns:
        float: Next year's grade.
    """
    return grade*(1 - params.grade_decline_rate*utilisation)


def flat_decision_price(ctx, stage_name, n_years):
    """Default NPV price path: last year's decision price of the named stage, for every year.

    Args:
        ctx (PoolContext): Context of the stage making the decision.
        stage_name (str): Stage whose price path is wanted (the stage itself, or its parent).
        n_years (int): Length of the path.

    Returns:
        array: Price in each production year.
    """
    return np.full(n_years, ctx.ctx.decision_price_of(stage_name))


def present_value(cash, rate):
    """Present value of yearly cash flows indexed from k = 0, each discounted by (1 + rate)**k.

    Args:
        cash (array): Cash flow in each year.
        rate (float): Discount rate.

    Returns:
        float: The present value.
    """
    return float(np.sum(cash/(1 + rate)**np.arange(len(cash))))


def mine_npv(mine, ctx, params, price_path_fn):
    """Net present value of deciding to open a mine now.

    Years are counted from the decision year, k = 0, matching the simulation calendar: a mine that
    passes the gate in year t produces from year t + construction_time.
    - Capex: -capex / construction_time in each of k = 0 ... construction_time - 1 (all of it at k = 0
      if construction_time is 0).
    - Production: k = construction_time ... construction_time + mine_life - 1, at full capacity, with
      cash flow (price - cost) * quantity. Capacity and cost follow the projected grade path (the
      stage's grade_fn at utilisation 1), and cumulative production stops at the mine's reserves.
    - Each year is discounted by (1 + discount_rate)**k.

    Args:
        mine (Mine): The mine. It is not modified.
        ctx (PoolContext): Context of the mine's stage; ctx.stage.grade_fn gives the grade path.
        params (ModelParams): Model parameters.
        price_path_fn (callable): price_path_fn(ctx, stage_name, n_years) -> price in each production year.

    Returns:
        float: The NPV.
    """
    build, life = params.construction_time, params.mine_life

    grades = np.empty(life)
    grade = mine.grade
    for j in range(life):
        grades[j] = grade
        grade = ctx.stage.grade_fn(grade, 1.0, params)

    capacity = mine.ore_throughput*grades*mine.recovery
    cost = mine.ore_cost/(grades*mine.recovery)
    produced_before = np.cumsum(capacity) - capacity
    quantity = np.clip(mine.reserves - produced_before, 0.0, capacity)

    capex_years = max(build, 1)
    cash = np.zeros(build + life)
    cash[:capex_years] -= mine.capex/capex_years
    cash[build:] += (price_path_fn(ctx, ctx.stage.name, life) - cost)*quantity
    return present_value(cash, params.discount_rate)


def plant_npv(plant, ctx, params, price_path_fn):
    """Net present value of building a processing plant now, and the present value of its capex.

    Years are counted from the decision year, k = 0: a plant built in year t operates from year
    t + plant_construction_time.
    - Capex: -capex / plant_construction_time in each of k = 0 ... plant_construction_time - 1 (all of
      it at k = 0 if plant_construction_time is 0).
    - Production: k = plant_construction_time ... plant_construction_time + plant_life - 1, at full
      capacity, with cash flow (p_k - c_k - intensity * p_parent_k) * capacity. The plant learns from
      construction on: c_k = conversion_cost * (1 - plant_learning_rate)**k. Prices come from
      price_path_fn for the stage and its parent (0 for a stage with no parent).
    - Each year is discounted by (1 + discount_rate)**k.

    Args:
        plant (SupplyPlant): The candidate plant, with capacity, conversion_cost, intensity and capex set.
        ctx (PoolContext): Context of the plant's stage.
        params (ModelParams): Model parameters.
        price_path_fn (callable): price_path_fn(ctx, stage_name, n_years) -> price in each production year.

    Returns:
        tuple: (NPV, present value of the capex).
    """
    build, life = params.plant_construction_time, params.plant_life
    k = np.arange(build + life)

    capex_years = max(build, 1)
    capex = np.zeros(build + life)
    capex[:capex_years] = plant.capex/capex_years

    price = price_path_fn(ctx, ctx.stage.name, life)
    parent_price = np.zeros(life) if ctx.stage.parent is None else price_path_fn(ctx, ctx.stage.parent.name, life)
    cost = plant.conversion_cost*(1 - params.plant_learning_rate)**k[build:]

    cash = -capex
    cash[build:] += (price - cost - (plant.intensity or 0.0)*parent_price)*plant.capacity
    return present_value(cash, params.discount_rate), present_value(capex, params.discount_rate)


# ---------------------------------------------------------------------------
# Transition rules: rule(plants, ctx) -> list of the plants that move
# ---------------------------------------------------------------------------

def after_years(n):
    """Rule: every plant that has spent at least n full years in its pool."""
    def rule(plants, ctx):
        return [plant for plant in plants if plant.years_in_pool >= n]
    return rule


def close_unprofitable(plants, ctx):
    """operating -> closed: every mine whose cost this year (after depletion) exceeds last year's
    decision price."""
    return [mine for mine in plants if mine.cost > ctx.decision_price]


def exhausted(plants, ctx):
    """operating -> exhausted: every mine whose reserves are used up."""
    return [mine for mine in plants if mine.reserves <= EXHAUSTION_TOLERANCE*mine.capacity]


def positive_npv(plants, ctx):
    """incentive -> opening: every mine with a positive NPV, with no cap. Stores npv_at_decision on
    each mover."""
    movers = []
    for mine in plants:
        npv = mine_npv(mine, ctx, ctx.params, ctx.stage.price_path_fn)
        if npv > 0:
            mine.npv_at_decision = npv
            movers.append(mine)
    return movers


def margin(plant, ctx):
    """A processing plant's running margin at last year's decision prices:
    P - (conversion_cost + intensity * P_parent)."""
    return ctx.decision_price - (plant.conversion_cost + (plant.intensity or 0.0)*ctx.parent_decision_price)


def negative_margin(plants, ctx):
    """operating -> idle: every plant whose margin at last year's decision prices is negative. Waste
    streams never go idle."""
    return [plant for plant in plants if not isinstance(plant, WasteStream) and margin(plant, ctx) < 0]


def margin_recovered(plants, ctx):
    """idle -> operating: every plant whose margin at last year's decision prices is back at 0 or above."""
    return [plant for plant in plants if margin(plant, ctx) >= 0]


def fill_incentive_gap(plants, ctx):
    """resource -> developing: the cheapest resource mines that cover the incentive gap.

    target = incentive_ratio * last year's production of the stage (realised output, not capacity)
    gap    = target - (K_incentive + K_developing), in nameplate capacity

    Counting developing capacity stops the same shortfall being refilled every year while earlier
    picks are still in development.
    """
    target = ctx.params.incentive_ratio*ctx.production
    gap = target - (ctx.capacity_of("incentive") + ctx.capacity_of("developing"))
    return cheapest_covering(plants, gap)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class SupplyChainStage():
    def __init__(self, supply, demand, name, pipeline, pools=None):
        """Base class for MiningStage and ProcessingStage: one supply chain stage whose plants move
        through a pipeline of pools.

        Stages are mutable: the graph is built once, and every year each stage is advanced in place by
        step(). Past years are kept as deepcopy snapshots by SupplyChain. The pools dict is the source
        of truth: the supply curve mirrors pools["operating"].

        Args:
            supply (callable): Supply for that stage. Its plants become the operating pool.
            demand (callable): Demand for that stage.
            name (str): Human-readable stage name (e.g. "midstream"). Used so plants can reference
                another stage's production by name.
            pipeline (Pipeline): The stage's pools.
            pools (dict, optional): Starting membership, {pool name: list of plants}, for pools other
                than operating. Any pool left out starts empty.
        """

        self.supply = supply
        self.demand = demand
        self.name = name
        self.parent = None
        if isinstance(demand, CompositeDemand):
            for downstream in demand.downstreams:
                downstream.parent = self
        self.p_eq = None
        # Set by SupplyChain: the smoothed price supply decisions respond to, and the model parameters
        self.decision_price = None
        self.params = None

        self.pipeline = pipeline
        self.pools = pipeline.empty_pools()
        for pool_name, plants in (pools or {}).items():
            if pool_name not in self.pools:
                raise ValueError(f"Unknown pool {pool_name}; this pipeline has {pipeline.names}.")
            self.pools[pool_name] = list(plants)
        self.pools["operating"] = list(self.supply.plants)

    def set_params(self, params):
        """Give the stage its model parameters. Called by SupplyChain when the chain is built.

        Args:
            params (ModelParams): Model parameters.
        """
        self.params = params

    def all_pools(self):
        """Membership view used for recording.

        Returns:
            dict: {pool name: list of plants}, including "operating".
        """
        return dict(self.pools)

    def set_plants(self, plants):
        """Replace this stage's supply-curve plants, rebuilding the curve's cached arrays.

        Args:
            plants (list): The stage's new full list of operating plants.
        """
        self.supply.set_plants(plants)

    def _step_demand(self, ctx):
        """Advance a leaf stage's demand to this year (SectorDemand.step), anchored at its own price last
        year. A CompositeDemand points at the live child stages, which are stepped themselves, so it
        needs nothing.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """
        if not isinstance(self.demand, CompositeDemand):
            self.demand.step(ctx.year, self.p_eq, self.params)

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
    POOLS = ("resource", "developing", "incentive", "opening", "operating", "closed", "exhausted")

    def __init__(self, supply, demand, name, capex_intensity=None, pools=None, incentive_pool=None,
                 resource_pool=None, grade_fn=default_grade_fn, price_path_fn=flat_decision_price):
        """Subtype of a SupplyChainStage whose mines move through a chain of pools.

        Pools, upstream first: resource -> developing -> incentive -> opening -> operating -> closed,
        plus a frozen exhausted pool. The pools dict is the source of truth: the supply curve is
        rebuilt from pools["operating"] at the end of every step.

        Args:
            supply (callable): Supply for that stage. Its plants become the operating pool.
            demand (callable): Demand for that stage.
            name (str): Human-readable stage name.
            capex_intensity (float, optional): Capex per unit of annual capacity, the same for every mine
                in the stage. Sets the capex of every mine that has none; required unless every mine
                has its own.
            pools (dict, optional): Starting membership, {pool name: list of mines}. Any pool left out
                starts empty.
            incentive_pool (list, optional): Shorthand for pools={"incentive": ...}.
            resource_pool (list, optional): Shorthand for pools={"resource": ...}.
            grade_fn (callable, optional): grade_fn(grade, utilisation, params) -> next year's grade of
                a mine that produced at the given utilisation. Defaults to default_grade_fn.
            price_path_fn (callable, optional): price_path_fn(ctx, stage_name, n_years) -> expected price in
                each production year, used in the NPV. Defaults to flat_decision_price.
        """

        # The two shorthands are just entries in the same starting-membership dict
        starting = dict(pools or {})
        if incentive_pool is not None:
            starting["incentive"] = incentive_pool
        if resource_pool is not None:
            starting["resource"] = resource_pool
        super().__init__(supply, demand, name, Pipeline(self.POOLS, frozen=("exhausted",)), starting)

        self.grade_fn = grade_fn
        self.price_path_fn = price_path_fn
        self.capex_intensity = capex_intensity

        # Capex is set once, here, and never recomputed
        for mines in self.pools.values():
            for mine in mines:
                if mine.capex is None:
                    if capex_intensity is None:
                        raise ValueError(f"Stage {name!r} needs a capex_intensity: mine {mine.uid} has no capex.")
                    mine.capex = capex_intensity*mine.capacity

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
    def closed_pool(self):
        return self.pools["closed"]

    @property
    def exhausted_pool(self):
        return self.pools["exhausted"]

    def set_params(self, params):
        """Give the stage its model parameters, and fill in the reserves of mines that have none
        (params.default_reserve_life years of capacity).

        Args:
            params (ModelParams): Model parameters.
        """
        super().set_params(params)
        if params.closed_destination not in self.pools:
            raise ValueError(f"closed_destination {params.closed_destination!r} is not one of {self.pipeline.names}.")

        for mines in self.pools.values():
            for mine in mines:
                if mine.reserves is None:
                    mine.reserves = params.default_reserve_life*mine.capacity
        self.set_plants(self.pools["operating"])

    def add_plants(self, new_plants):
        """Add mines straight into production.

        Args:
            new_plants (list): Mines to add to the operating pool.
        """
        self.pools["operating"].extend(new_plants)
        self.set_plants(self.pools["operating"])

    def step(self, ctx):
        """Advance the stage by one year, in place. Every decision uses last year's values, and prices
        are the decision price (smoothed equilibrium price).

        1. Depletion of every mine that was operating last year (_deplete).
        2. operating -> exhausted: reserves used up (exhausted).
        3. operating -> closed: cost after depletion above the decision price (close_unprofitable).
        4. Ageing of every mine that hasn't moved this step (Pipeline.age).
        5. opening -> operating after construction_time years.
        6. incentive -> opening: positive NPV (positive_npv).
        7. developing -> incentive after feasibility_time years.
        8. closed -> closed_destination after closed_cooldown years.
        9. resource -> developing: greedy fill of the incentive gap (fill_incentive_gap).
        10. The supply curve is rebuilt from the operating pool.

        A mine moves at most once per step. The minimum lead time from resource to operating is
        feasibility_time + 1 + construction_time years: a mine that reaches incentive in step 7 can
        first pass the NPV gate in step 6 of the following year.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot step stage.")

        params = self.params
        pool_ctx = PoolContext(self, ctx)
        moved = set()
        self._deplete(self.pools["operating"], ctx)
        self.pipeline.move(self.pools, "operating", "exhausted", exhausted, pool_ctx, moved)
        self.pipeline.move(self.pools, "operating", "closed", close_unprofitable, pool_ctx, moved)
        self.pipeline.age(self.pools, moved)
        self.pipeline.move(self.pools, "opening", "operating", after_years(params.construction_time),
                           pool_ctx, moved)
        self.pipeline.move(self.pools, "incentive", "opening", positive_npv, pool_ctx, moved)
        self.pipeline.move(self.pools, "developing", "incentive", after_years(params.feasibility_time),
                           pool_ctx, moved)
        self.pipeline.move(self.pools, "closed", params.closed_destination, after_years(params.closed_cooldown),
                           pool_ctx, moved)
        self.pipeline.move(self.pools, "resource", "developing", fill_incentive_gap, pool_ctx, moved)

        # pools["operating"] is the truth, the supply curve just mirrors it
        self.set_plants(self.pools["operating"])

        self._step_demand(ctx)

    def _deplete(self, mines, ctx):
        """Depletion: each mine loses last year's production from its reserves, and its grade moves
        along grade_fn at last year's utilisation (production / nameplate capacity).

        Args:
            mines (list): The mines that were operating last year.
            ctx (StageContext): Read-only view of last year's solved chain.
        """
        for mine in mines:
            produced = ctx.consumed_by(mine)
            mine.reserves = max(mine.reserves - produced, 0.0)
            mine.grade = self.grade_fn(mine.grade, produced/mine.capacity, self.params)


class ProcessingStage(SupplyChainStage):
    POOLS = ("construction", "operating", "idle", "closed")

    def __init__(self, supply, demand, name, capex_intensity, pools=None, price_path_fn=flat_decision_price):
        """Subtype of a SupplyChainStage for processing plants, which are replicable.

        Pools are construction -> operating <-> idle -> closed, and closed is frozen (plants never
        re-enter). New plants are built at the stage's technology frontier when a template plant's NPV
        is positive.

        Waste streams sit in operating for good: they never go idle, close or expand, but they follow
        the learning rule and count in the operating capacity that scales expansion.

        Args:
            supply (callable): Supply for that stage. Its plants become the operating pool.
            demand (callable): Demand for that stage.
            name (str): Human-readable stage name.
            capex_intensity (float): Capex of a new plant per unit of annual capacity.
            pools (dict, optional): Starting membership for "construction", "idle" or "closed",
                {pool name: list of plants}.
            price_path_fn (callable, optional): price_path_fn(ctx, stage_name, n_years) -> expected
                price in each production year, used in the template plant NPV. Defaults to
                flat_decision_price.
        """
        super().__init__(supply, demand, name, Pipeline(self.POOLS, frozen=("closed",)), pools)
        self.capex_intensity = capex_intensity
        self.price_path_fn = price_path_fn
        # Conversion cost of a plant built now; set from the starting plants in set_params
        self.frontier_cost = None

    def set_params(self, params):
        """Give the stage its model parameters, and set the starting technology frontier: the
        expansion_cost_percentile conversion cost of the starting regular operating plants.

        Args:
            params (ModelParams): Model parameters.
        """
        super().set_params(params)
        regular = [plant.conversion_cost for plant in self.pools["operating"] if not isinstance(plant, WasteStream)]
        self.frontier_cost = float(np.percentile(regular, params.expansion_cost_percentile)) if regular else None

    def add_plants(self, new_plants):
        """Add plants straight into production, with costs quoted at the stage's current upstream price.

        Args:
            new_plants (list): Plants to add to the operating pool.
        """
        self.supply.add_plants(new_plants)
        self.pools["operating"] = list(self.supply.plants)

    def step(self, ctx):
        """Advance the stage by one year, in place. Decisions use last year's decision prices.

        1. Ageing of every plant (Pipeline.age), before any move, so years_in_pool counts full years.
        2. idle -> operating: margin back at 0 or above (margin_recovered); the plant produces this year.
        3. idle -> closed after max_idle_years years idle.
        4. operating -> idle: negative margin (negative_margin).
        5. Learning on every construction and operating plant (_learn).
        6. construction -> operating after plant_construction_time years.
        7. The technology frontier moves on a year (_advance_frontier).
        8. Expansion with template plants at the frontier (_expand), measured on last year's pools.
        9. Waste-stream capacities from their source flows (WasteStream.step).
        10. The supply curve is rebuilt from the operating pool, at last year's parent price.

        A plant that goes idle in year t and never recovers closes in year t + max_idle_years, after
        exactly max_idle_years years without production. A plant that moves in a step can't move
        again in that step, so a plant that restarts can't go idle the same year.

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
        """
        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot step stage.")

        params = self.params
        pool_ctx = PoolContext(self, ctx)
        moved = set()
        # Last year's pools, before any move: expansion is measured against them
        last_operating = sum(plant.capacity for plant in self.pools["operating"])
        last_regular = [plant.capacity for plant in self.pools["operating"] if not isinstance(plant, WasteStream)]

        self.pipeline.age(self.pools, moved)
        self.pipeline.move(self.pools, "idle", "operating", margin_recovered, pool_ctx, moved)
        self.pipeline.move(self.pools, "idle", "closed", after_years(params.max_idle_years), pool_ctx, moved)
        self.pipeline.move(self.pools, "operating", "idle", negative_margin, pool_ctx, moved)
        self._learn()
        self.pipeline.move(self.pools, "construction", "operating", after_years(params.plant_construction_time),
                           pool_ctx, moved)
        self._advance_frontier()
        self._expand(pool_ctx, last_operating, last_regular)

        for plant in self.pools["operating"]:
            if isinstance(plant, WasteStream):
                plant.step(ctx)

        self.set_plants(self.pools["operating"])
        if isinstance(self.supply, SupplyCurve2D) and self.parent is not None:
            self.supply.set_upstream_price(ctx.price_of(self.parent.name))

        self._step_demand(ctx)

    def _learn(self):
        """Learning: conversion cost falls by plant_learning_rate for every plant in construction or
        operating, waste streams included. Idle plants don't learn."""
        for name in ("construction", "operating"):
            for plant in self.pools[name]:
                plant.conversion_cost *= (1 - self.params.plant_learning_rate)

    def _advance_frontier(self):
        """Technology frontier: the conversion cost of a newly built plant falls by technology_rate a year."""
        if self.frontier_cost is not None:
            self.frontier_cost *= (1 - self.params.technology_rate)

    def _expand(self, ctx, last_operating, last_regular):
        """Expansion: build template plants into construction when a new plant at the technology
        frontier has a positive NPV.

        The template plant has the frontier conversion cost, the median capacity of last year's regular
        (non-waste) operating plants, the stage's intensity and capex = capex_intensity * capacity. If
        its NPV (plant_npv) is positive:
            PI          = NPV / PV(capex)        (profitability index)
            build_share = min(max_expansion_share, expansion_speed * PI)
            n_plants    = build_share * last year's operating capacity / template capacity, rounded to
                          the nearest whole plant (waste streams count in the operating capacity)
        New plants are synthetic and get a (year, "construction") entry in their pool_history.

        Args:
            ctx (PoolContext): Context of this stage.
            last_operating (float): Operating capacity during last year's solve.
            last_regular (list): Capacities of last year's regular operating plants.
        """
        params = self.params
        if self.frontier_cost is None or not last_regular or last_operating <= 0:
            return

        capacity = float(np.median(last_regular))
        intensity = getattr(self.supply, "intensity", None) or 0.0
        template = SupplyPlant(capacity, self.frontier_cost, intensity=intensity,
                               capex=self.capex_intensity*capacity, synthetic=True)
        npv, pv_capex = plant_npv(template, ctx, params, self.price_path_fn)
        if npv <= 0:
            return

        build_share = min(params.max_expansion_share, params.expansion_speed*npv/pv_capex)
        n_plants = int(np.floor(build_share*last_operating/capacity + 0.5))
        for _ in range(n_plants):
            plant = SupplyPlant(capacity, self.frontier_cost, intensity=intensity,
                                capex=self.capex_intensity*capacity, synthetic=True)
            plant.pool_history.append((ctx.year, "construction"))
            self.pools["construction"].append(plant)
