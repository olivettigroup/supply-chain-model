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
    
class SectorDemand():
    def __init__(self, base_price, elasticity, V0, I0, cagr, phi=1.0, chi=1.0, sectors=None):
        """Demand of a leaf stage, built bottom-up from sectors: D(p) = q_nom * (p / p_nom)**elasticity.

        Each sector s has a driver V_s (e.g. units sold) and an intensity I_s (material per driver
        unit); its nominal demand is D_s = V_s * I_s, and q_nom = sum of D_s. The curve passes through
        (p_nom, q_nom): p_nom is base_price in the first year and last year's equilibrium price after
        that. Sector arguments are arrays with one entry per sector, or scalars for a single sector.

        Args:
            base_price (float): Price at which the first year's nominal demand applies.
            elasticity (float): Price elasticity of the stage's demand.
            V0 (array): Drivers in the first year.
            I0 (array): Intensities in the first year (0 allowed).
            cagr (array): Driver growth rates at the start of the run.
            phi (array, optional): Factor by which driver growth is scaled by the end of the run.
                Defaults to 1 (constant growth).
            chi (array, optional): Factor by which intensity has changed by the end of the run. Defaults
                to 1 (constant intensity).
            sectors (list, optional): Sector names. Defaults to "sector 0", "sector 1", ...
        """
        V0, I0, cagr, phi, chi = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float))
                                                       for x in (V0, I0, cagr, phi, chi)))
        self.V = V0.copy()
        self.I0 = I0.copy()
        self.cagr = cagr.copy()
        self.phi = phi.copy()
        self.chi = chi.copy()
        self.k = np.ones_like(self.V)
        self.sectors = list(sectors) if sectors is not None else [f"sector {i}" for i in range(len(self.V))]
        if len(self.sectors) != len(self.V):
            raise ValueError(f"{len(self.sectors)} sector names for {len(self.V)} sectors.")
        self.elasticity = elasticity
        self.p_nom = base_price
        self.q_nom = float(np.sum(self.sector_demand))

    @property
    def intensity(self):
        """Each sector's intensity this year: I0 * k."""
        return self.I0*self.k

    @property
    def sector_demand(self):
        """Each sector's nominal demand this year: V * I."""
        return self.V*self.intensity

    def step(self, t, last_price, params):
        """Demand update for year t: advance every sector's driver and intensity, then re-anchor the
        curve at last year's equilibrium price.

            x        = (t - t0) / (T - t0), clipped at 1, with t0 = start_year and T = end_year
            V(s, t)  = V(s, t-1) * (1 + cagr_s * phi_s**x)
            I(s, t)  = I0_s * chi_s**x
            q_nom(t) = sum of V(s, t) * I(s, t),  p_nom(t) = last_price

        Args:
            t (int): Year being stepped into.
            last_price (float): The stage's equilibrium price last year.
            params (ModelParams): Model parameters; uses start_year and end_year.
        """
        x = min((t - params.start_year)/max(params.end_year - params.start_year, 1), 1.0)
        self.V = self.V*(1 + self.cagr*self.phi**x)
        self.k = self.chi**x
        self.q_nom = float(np.sum(self.sector_demand))
        self.p_nom = last_price

    def __call__(self, p):
        """Quantity demanded at price(s) p."""
        return self.q_nom*(np.asarray(p, dtype=float)/self.p_nom)**self.elasticity

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
    def __init__(self, downstreams, other=0):
        """Sums the demand from one or more downstream stages.

        Args:
            downstreams (SupplyChainStages): Supply chain stages that are downstream.
            other (float, optional): Additional demand from unsimulated stages. Defaults to 0.
        """
        self.downstreams = downstreams
        self.other = other

    def __call__(self, p):
        """Evaluation of the demand curve.

        Args:
            p (array): Prices to evaluate the curve at.

        Returns:
            array: Resulting quantities for each price.
        """
        scalar_input = np.isscalar(p)
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        supply = np.zeros_like(p_arr)

        for i, pi in enumerate(p_arr):
            for downstream in self.downstreams:
                supply[i] += downstream.supply.upstream_demand(downstream.get_equilibrium_price(pi), p_u=pi)

        supply += self.other
        
        return supply.item() if scalar_input else supply

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
    """Given any node in the graph, builds the parents and children index list with ordered nodes list.

    Args:
        start_node (SupplyChainStage): Any stage in the supply chain graph.

    Returns:
        lists: Lists of nodes, parent, and children.
    """

    root_node = start_node
    while root_node.parent is not None:
        root_node = root_node.parent

    nodes, parent, children = [], [], []

    def visit(node, parent_idx):
        idx = len(nodes)
        nodes.append(node)
        parent.append(parent_idx)
        children.append([])
        if parent_idx >= 0:
            children[parent_idx].append(idx)
        if isinstance(node.demand, CompositeDemand):
            for child in node.demand.downstreams:
                visit(child, idx)

    visit(root_node, -1)

    return nodes, parent, children

def _residuals(logp, nodes, parent, children):
    """Price residuals for every stage.

    Args:
        logp (array): Log of prices.
        nodes (array): List of SupplyChainStages.
        parent (array): Parent node for every stage.
        children (array): Children nodes for every stage.

    Returns:
        array: Residuals between supply and demand for every stage at prices logp.
    """

    # Transform back into regular space
    p = np.exp(logp)

    # Get the supply at each stage
    S = np.array([nodes[i].supply(p[i]) if parent[i] < 0 else nodes[i].supply(p[i], p[parent[i]]) for i in range(len(nodes))])

    # For convergence, we scale the residuals 
    scale = np.array([n.supply.capacities.sum() for n in nodes])
    
    # For every node, calculate the demand and subtract it from the supply to get the residual
    res = np.empty(len(nodes))
    for i, node in enumerate(nodes):
        if isinstance(node.demand, CompositeDemand):
            D = node.demand.other + sum(nodes[c].supply.upstream_demand(p[c], p[i]) for c in children[i])
        else:
            D = node.demand(p[i])

        # No matter the demand, subtract the supply and scale it for better convergence
        res[i] = (S[i] - D) / scale[i]
    
    return res

def solve_chain(nodes, parent, children, p0=None):
    """Solves the prices for one SupplyChain

    Args:
        nodes (array): List of SupplyChainStages.
        parent (array): Parent node for every stage.
        children (array): Children nodes for every stage.
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
    sol = least_squares(_residuals, x0, args=(nodes, parent, children), method="trf", bounds=(lb, np.inf), xtol=1e-12, ftol=1e-12)
    
    # Return prices and write them to the nodes
    p = np.exp(sol.x)
    
    return {n: pi for n, pi in zip(nodes, p)}