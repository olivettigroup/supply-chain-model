import numpy as np
from scipy.optimize import root_scalar, least_squares

# Parameters to generate random cost curves. Taken from global Cu cost curve from S&P
MINE_DISTRIBUTION_MEAN = np.array([-7.6971, 0.323])
MINE_DISTRIBUTION_COV = np.array([[ 4.6362, -0.0874],
                                  [-0.0874,  0.0164]])
MINE_DISTRIBUTION_COST_MIN = 0.0647

MIN_PRICE_FACTOR = 0.95
SHARPNESS = 500

# Numerical guard only: a plant's total cost can never fall below this fraction of its cost at last
# year's upstream price, however low an upstream price the solver probes. Without it, a plant with a
# negative conversion cost would get a negative total cost and the curve would return nan.
# SupplyChain.solve_prices warns if it binds at a solved price.
MIN_COST_FACTOR = 0.05

def generate_cost_curve(n_plants, total_capacity=300, cost_bounds=(1000, 5000), rng=None):
    """Randomly generates a number of plants for any supply chain stage

    Args:
        n_plants (int): Number of plants to generate
        total_capacity (float, optional): Total capacity to distribute among the plants. Defaults to 300.
        cost_bounds (tuple, optional): Bounds for the generated costs. Defaults to (1000, 5000).
        rng (numpy.random.Generator, optional): Random generator, for reproducible draws. Defaults to a
            fresh unseeded one.

    Returns:
        arrays: Arrays of length n_plants for capacities and costs
    """

    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random samples in log space
    samples = rng.multivariate_normal(mean=MINE_DISTRIBUTION_MEAN, cov=MINE_DISTRIBUTION_COV, size=(n_plants))

    # Clip to bounds or transform back into normal space
    samples[:, 0] = np.exp(samples[:, 0])
    samples[:, 1] = np.clip(samples[:, 1], MINE_DISTRIBUTION_COST_MIN, None)

    # Rescale the samples to fit the desired total capacity and cost bounds
    cost_min, cost_max = cost_bounds
    samples[:, 1] = (samples[:, 1] - samples[:, 1].min()) / (samples[:, 1].max() - samples[:, 1].min() + 1e-12)
    samples[:, 1] = cost_min + samples[:, 1] * (cost_max - cost_min)

    return samples[:, 0]*total_capacity/samples[:, 0].sum(), samples[:, 1]

def plant_quantities(p, capacities, costs, eta):
    """Per-plant quantities supplied at the given price(s): the plant supply response.

    Each plant switches on steeply around MIN_PRICE_FACTOR * its cost (a logistic with SHARPNESS),
    and above that supplies capacity * (p / cost)**eta, scaled so it supplies its capacity at p = cost.

    Args:
        p (array): Prices to evaluate at.
        capacities (array): Capacity of each plant.
        costs (array): Total cost of each plant.
        eta (float): Elasticity of each plant's marginal price curve.

    Returns:
        array: Quantities per plant, shape (len(p), n_plants).
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))[:, np.newaxis]
    l = lambda x: 1.0 / (1.0 + np.exp(-SHARPNESS * (x / (MIN_PRICE_FACTOR * costs) - 1.0)))
    A = capacities / (l(costs) * costs ** eta)
    return l(p_arr) * A * p_arr ** eta  # (m, n_plants)

class SupplyCurve:
    def __init__(self, plants, eta=0.1):
        """Basic supply curve composed of a horizontal sum of marginal price curves.

        Args:
            plants (list): List of supply plants.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
        """
        self.eta = eta
        self.set_plants(plants)

    def refresh(self):
        """Recompute the cached arrays from the current plants.

        The plants are mutated in place every year, so anything derived from them
        (capacities, costs) goes stale unless this is called afterwards.
        """
        self.capacities = np.array([plant.usable_capacity for plant in self.plants], dtype=float)
        self.costs = np.array([plant.cost for plant in self.plants], dtype=float)

    def set_plants(self, plants):
        """Replace the plant list and refresh the cached arrays."""
        self.plants = np.array(list(plants), dtype=object)
        self.refresh()

    def add_plants(self, new_plants):
        """Append plants and refresh the cached arrays."""
        self.set_plants(list(self.plants) + list(new_plants))

    def _plant_quantities(self, p):
        """Per-plant quantities supplied at the given price(s).

        Args:
            p (array): Prices to evaluate the curve at.

        Returns:
            array: Quantities per plant, shape (len(p), n_plants).
        """
        return plant_quantities(p, self.capacities, self.costs, self.eta)

    def __call__(self, p):
        """Evaluation of the supply curve.

        Args:
            p (array): Prices to evaluate the curve at.

        Returns:
            array: Resulting quantities for each price.
        """

        supply = self._plant_quantities(p).sum(axis=-1)
        return supply.item() if np.isscalar(p) else supply

class SupplyCurve2D():
    def __init__(self, plants, intensity, p_u_ref, eta=0.1):
        """Supply curve with changing upstream price.

        Each plant's primitive cost is its conversion cost (non-material cost per unit of output). Its
        total cost at upstream price p_u is conversion_cost + intensity * p_u.

        Args:
            plants (list): List of supply plants. Their cost is the total cost quoted at p_u_ref.
            intensity (float): Default upstream input required per unit output, for plants that don't
                set their own (plant.intensity is None).
            p_u_ref (float): Upstream price the plants' costs are quoted at. Only used here, to split
                each total cost into conversion cost and input cost.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
        """
        self.intensity = intensity
        # Last year's upstream price; the stage moves it with the parent's price every year
        self.p_u = p_u_ref
        self.eta = eta
        plants = list(plants)
        self._quote(plants)
        self.set_plants(plants)

    def refresh(self):
        """Recompute the cached arrays from the current plants and upstream price.

        The plants are mutated in place every year, so every array derived from them has to be
        rebuilt afterwards.
        """
        self.capacities = np.array([plant.usable_capacity for plant in self.plants], dtype=float)
        self.conversion_costs = np.array([plant.conversion_cost for plant in self.plants], dtype=float)
        self.plant_intensities = np.array([plant.intensity for plant in self.plants], dtype=float)
        self.costs = self.conversion_costs + self.plant_intensities*self.p_u

    def _quote(self, plants):
        """Split each new plant's total cost, quoted at the curve's upstream price, into conversion
        cost and input cost. Plants without their own intensity get the curve's.

        Args:
            plants (list): Plants joining the curve for the first time.
        """
        for plant in plants:
            if plant.intensity is None:
                plant.intensity = self.intensity
            plant.conversion_cost = plant.cost - plant.intensity*self.p_u

    def set_plants(self, plants):
        """Replace the plant list and refresh the cached arrays. Conversion costs are kept as they are,
        so this also takes back plants that sat off the curve for a while (e.g. under construction).

        Args:
            plants (list): Plants whose conversion cost and intensity are already set.
        """
        self.plants = np.array(list(plants), dtype=object)
        for plant in self.plants:
            plant.upstream_price = self.p_u
        self.refresh()

    def add_plants(self, new_plants):
        """Append new plants, with costs quoted at the curve's current upstream price, and refresh the
        cached arrays.

        Args:
            new_plants (list): Plants joining the curve for the first time.
        """
        new_plants = list(new_plants)
        self._quote(new_plants)
        self.set_plants(list(self.plants) + new_plants)

    def set_upstream_price(self, p_u):
        """Move the curve and its plants to a new upstream price. Conversion costs are unchanged.

        Args:
            p_u (float): The new upstream price (last year's parent price).
        """
        self.p_u = p_u
        for plant in self.plants:
            plant.upstream_price = p_u
        self.refresh()

    def costs_at(self, p_u=None):
        """Per-plant total cost at a given upstream price.

        Floored at MIN_COST_FACTOR of the cost at the curve's own upstream price, so the curve stays
        defined (and positive) even when the solver probes a very low upstream price.

        Args:
            p_u (float, optional): Upstream price. Defaults to the curve's upstream price.

        Returns:
            array: Cost of each plant.
        """
        if p_u is None:
            p_u = self.p_u

        return np.maximum(self.conversion_costs + self.plant_intensities*p_u, MIN_COST_FACTOR*self.costs)

    def _plant_quantities(self, p_d, p_u=None):
        """Per-plant quantities supplied at the given price(s).

        Args:
            p_d (array): Prices to evaluate the curve at.
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            array: Quantities per plant, shape (len(p_d), n_plants).
        """
        return plant_quantities(p_d, self.capacities, self.costs_at(p_u), self.eta)

    def __call__(self, p_d, p_u=None):
        """Evaluation of the supply curve.

        Args:
            p_d (array): Prices to evaluate the curve at.
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            array: Resulting quantities for each price.
        """
        supply = self._plant_quantities(p_d, p_u).sum(axis=-1)

        return supply.item() if np.isscalar(p_d) else supply

    def upstream_demand(self, p_d, p_u=None):
        """Quantity of upstream input consumed to produce this stage's output at the given price(s).

        Plants with intensity=0 (e.g. a WasteStream that doesn't consume the upstream input) contribute nothing.

        Args:
            p_d (array): Prices to evaluate the curve at.
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            array: Resulting upstream quantities for each price.
        """
        demand = (self._plant_quantities(p_d, p_u) * self.plant_intensities).sum(axis=-1)

        return demand.item() if np.isscalar(p_d) else demand
    
class Sector():
    def __init__(self, name, elasticity, V0, cagr, intensities, phi=1.0, chi=1.0, other_cost=0.0):
        """A demand sector (e.g. a product): one driver volume that responds to the price of its
        inputs, drawing material from one or more stages.

        Its input price is P = other_cost + sum over drawn stages e of I_e * p_e, and its volume is
        V = V_nom * (P / P_nom)**elasticity. The curve passes through (P_nom, V_nom): P_nom is the
        input price at the drawn stages' base prices in the first year, and at last year's equilibrium
        prices after that. Stage e draws I_e * V.

        Args:
            name (str): Sector name, used in the records.
            elasticity (float): Price elasticity of the sector's volume to its input price.
            V0 (float): Driver volume in the first year (e.g. units sold).
            cagr (float): Driver growth rate at the start of the run.
            intensities (dict): {stage name: material drawn per driver unit in the first year}.
                Zero allowed.
            phi (float, optional): Factor by which driver growth is scaled by the end of the run.
                Defaults to 1 (constant growth).
            chi (float, optional): Factor by which every intensity has changed by the end of the run.
                Defaults to 1 (constant intensity).
            other_cost (float, optional): Non-material cost per driver unit, part of the input price.
                Defaults to 0.
        """
        self.name = name
        self.elasticity = elasticity
        self.V = float(V0)
        self.cagr = cagr
        self.phi = phi
        self.chi = chi
        self.other_cost = other_cost
        self.I0 = {stage: float(I) for stage, I in intensities.items()}
        self.k = 1.0
        # Prices of the drawn stages the curve is anchored at, filled in with base prices as the
        # stages are built; P_nom is set once every drawn stage has one
        self.prices = {}
        self.P_nom = None

    @property
    def intensities(self):
        """This year's intensity per drawn stage: I0 * k."""
        return {stage: I*self.k for stage, I in self.I0.items()}

    def price(self, prices):
        """The sector's input price at stage prices {stage name: price}."""
        return self.other_cost + sum(I*prices[stage] for stage, I in self.intensities.items())

    def volume(self, prices):
        """The sector's driver volume at stage prices {stage name: price}. A sector whose input price
        is zero (every intensity and other_cost zero) has nothing to respond to and stays at V_nom."""
        if self.P_nom == 0:
            return self.V
        return self.V*(self.price(prices)/self.P_nom)**self.elasticity

    def demand_on(self, stage_name, prices):
        """Quantity the sector draws from a stage at stage prices {stage name: price}."""
        return self.I0[stage_name]*self.k*self.volume(prices)

    def set_base_price(self, stage_name, base_price):
        """Record a drawn stage's base price; the first-year anchor is set once every drawn stage has one.

        Args:
            stage_name (str): A stage the sector draws from.
            base_price (float): That stage's price at which V0 applies.
        """
        self.prices[stage_name] = base_price
        if set(self.prices) == set(self.I0):
            self.P_nom = self.price(self.prices)

    def step(self, ctx, params):
        """Sector update for year ctx.year: advance the driver and the intensities, then re-anchor the
        curve at last year's equilibrium prices.

            x        = (t - t0) / (T - t0), clipped at 1, with t0 = start_year and T = end_year
            V_nom(t) = V_nom(t-1) * (1 + cagr * phi**x)
            I_e(t)   = I0_e * chi**x
            P_nom(t) = other_cost + sum of I_e(t) * p_e(t-1)

        Args:
            ctx (StageContext): Read-only view of last year's solved chain.
            params (ModelParams): Model parameters; uses start_year and end_year.
        """
        x = min((ctx.year - params.start_year)/max(params.end_year - params.start_year, 1), 1.0)
        self.V = self.V*(1 + self.cagr*self.phi**x)
        self.k = self.chi**x
        self.prices = {stage: ctx.price_of(stage) for stage in self.I0}
        self.P_nom = self.price(self.prices)

def find_intersection(supply_curve, demand_curve, bracket):
    """Finds the intersection of a supply and demand curve.

    Args:
        supply_curve (callable): Supply curve.
        demand_curve (callable): Demand curve.
        bracket (tuple): Initial bracket for the root finding.

    Returns:
        float: Intersection price.
    """
    f = lambda p: demand_curve(p) - supply_curve(p)
    p_lo, p_hi = bracket
    f_lo, f_hi = f(p_lo), f(p_hi)

    # Safeguard: expand outward if the cost-based bracket misses the root
    for _ in range(50):
        if f_lo * f_hi < 0:
            break
        if f_hi > 0:
            p_hi *= 2; f_hi = f(p_hi)
        elif f_lo < 0:
            p_lo *= 0.5; f_lo = f(p_lo)
        else:
            break
    else:
        raise ValueError(f"Could not bracket root in [{p_lo}, {p_hi}]: "
                         f"f_lo={f_lo}, f_hi={f_hi}")

    return root_scalar(f, bracket=(p_lo, p_hi), method="brentq").root

class CompositeDemand():
    def __init__(self, downstreams, base_price=None):
        """The demand on a stage: what its child stages and the sectors drawing from it take.

        Args:
            downstreams (list): Child SupplyChainStages and Sectors.
            base_price (float, optional): The stage's price at which its sectors' V0 apply. Required
                when sectors draw from the stage.
        """
        self.downstreams = list(downstreams)
        self.base_price = base_price

    @property
    def stages(self):
        """The child stages."""
        return [d for d in self.downstreams if not isinstance(d, Sector)]

    @property
    def sectors(self):
        """The sectors drawing from the stage."""
        return [d for d in self.downstreams if isinstance(d, Sector)]

def cheapest_covering(plants, gap):
    """Greedy gap fill: the cheapest plants whose combined capacity covers a gap.

    Plants are taken in order of cost until their nameplate capacity reaches the gap, including the
    plant that crosses it.

    Args:
        plants (list): Candidate plants.
        gap (float): Capacity to cover. Nothing is selected if it is not positive.

    Returns:
        list: The selected plants, cheapest first.
    """
    if gap <= 0 or len(plants) == 0:
        return []

    ordered = sorted(plants, key=lambda plant: plant.cost)
    capacities = np.array([plant.capacity for plant in ordered], dtype=float)

    # Capacity already committed before each plant, so the one straddling the gap is included
    committed = np.cumsum(capacities) - capacities
    return [plant for plant, before in zip(ordered, committed) if before < gap]

def build_chain(start_node):
    """Given any stage in the graph, list every stage with its parent's index, parents before children.

    Args:
        start_node (SupplyChainStage): Any stage in the supply chain graph.

    Returns:
        tuple: (stages, parent index of each stage, -1 for the root).
    """

    root_node = start_node
    while root_node.parent is not None:
        root_node = root_node.parent

    nodes, parent = [], []

    def visit(node, parent_idx):
        idx = len(nodes)
        nodes.append(node)
        parent.append(parent_idx)
        for child in node.demand.stages:
            visit(child, idx)

    visit(root_node, -1)

    return nodes, parent

def _residuals(logp, nodes, parent):
    """Price residuals for every stage.

    Args:
        logp (array): Log of prices.
        nodes (array): List of SupplyChainStages.
        parent (array): Parent index for every stage.

    Returns:
        array: Residuals between supply and demand for every stage at prices logp.
    """

    # Transform back into regular space
    p = np.exp(logp)
    prices = {node.name: pi for node, pi in zip(nodes, p)}

    # Get the supply at each stage
    S = np.array([nodes[i].supply(p[i]) if parent[i] < 0 else nodes[i].supply(p[i], p[parent[i]]) for i in range(len(nodes))])

    # For convergence, we scale the residuals
    scale = np.array([n.supply.capacities.sum() for n in nodes])

    # Demand on each stage: what its child stages and sectors draw from it at these prices
    res = np.empty(len(nodes))
    for i, node in enumerate(nodes):
        D = sum(d.demand_on(node.name, prices) for d in node.demand.downstreams)
        res[i] = (S[i] - D) / scale[i]

    return res

def solve_chain(nodes, parent, p0=None):
    """Solves the prices for one SupplyChain

    Args:
        nodes (array): List of SupplyChainStages.
        parent (array): Parent index for every stage.
        p0 (array, optional): Initial guesses. Defaults to None.

    Returns:
        dict: Dictionnary with SupplyChainStages as keys and their equilibrium prices as values.
    """

    # A price can't sit far below the stage's cheapest plant. For a child, take the cheapest cost at
    # the parent's own lower bound, so the bound holds however far the parent's price falls this year.
    # nodes lists every parent before its children, so the parent's bound is always ready.
    lb = np.empty(len(nodes))
    for i, n in enumerate(nodes):
        costs = n.supply.costs if parent[i] < 0 else n.supply.costs_at(lb[parent[i]])
        lb[i] = 0.5 * costs.min()
    lb = np.log(lb)
    
    # Get x0 as log or create it from the means. Passing last year's prices matters once the chain
    # has run for a while: the cost means can sit far enough from the answer that the solver starts
    # outside the region where every stage's curve is well behaved.
    if p0 is None:
        p0 = np.array([n.supply.costs.mean() for n in nodes])
    x0 = np.clip(np.log(np.asarray(p0, dtype=float)), lb, None)
    
    # Solve
    sol = least_squares(_residuals, x0, args=(nodes, parent), method="trf", bounds=(lb, np.inf), xtol=1e-12, ftol=1e-12)
    
    # Return prices and write them to the nodes
    p = np.exp(sol.x)
    
    return {n: pi for n, pi in zip(nodes, p)}