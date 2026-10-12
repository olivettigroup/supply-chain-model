import numpy as np
from scipy.optimize import root_scalar, least_squares

# Parameters to generate random cost curves. Taken from global Cu cost curve from S&P
MINE_DISTRIBUTION_MEAN = np.array([-7.6971, 0.323])
MINE_DISTRIBUTION_COV = np.array([[ 4.6362, -0.0874],
                                  [-0.0874,  0.0164]])
MINE_DISTRIBUTION_COST_MIN = 0.0647

MIN_PRICE_FACTOR = 0.95
SHARPNESS = 500

# A plant's cost can never fall below this fraction of its reference cost, however low the upstream
# price goes. Without a floor, costs_for_p_u can go negative (whenever min_costs is negative and the
# solver probes a low upstream price) and the curve returns nan.
MIN_COST_FACTOR = 0.05

def generate_cost_curve(n_plants, total_capacity=300, cost_bounds=(1000, 5000)):
    """Randomly generates a number of plants for any supply chain stage

    Args:
        n_plants (int): Number of plants to generate
        total_capacity (float, optional): Total capacity to distribute among the plants. Defaults to 300.
        cost_bounds (tuple, optional): Bounds for the generated costs. Defaults to (1000, 5000).

    Returns:
        arrays: Arrays of length n_plants for capacities and costs
    """

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

class SupplyCurve:
    def __init__(self, plants, eta=0.1):
        """Basic supply curve composed of a horizontal sum of marginal price curves.

        Args:
            plants (list): List of supply plants.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
            min_price (float, optional): Price floor for the marginal price curves. Defaults to 0.95.
        """
        self.eta = eta
        self.set_plants(plants)

    def refresh(self):
        """Recompute the cached arrays from the current plants.

        The plants are mutated in place every year, so anything derived from them
        (capacities, costs) goes stale unless this is called afterwards.
        """
        self.capacities = np.array([plant.capacity for plant in self.plants], dtype=float)
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
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))[:, np.newaxis]
        l = lambda x: 1.0 / (1.0 + np.exp(-SHARPNESS * (x / (MIN_PRICE_FACTOR * self.costs) - 1.0)))
        A = self.capacities / (l(self.costs) * self.costs ** self.eta)
        return l(p_arr) * A * p_arr ** self.eta  # (m, n_plants)

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

        Args:
            plants (list): List of supply plants.
            intensity (float): Default upstream input required per unit output, for plants that don't set their own (plant.intensity is None).
            p_u_ref (float): Reference price at which the costs are defined.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
            min_price (float, optional): Price floor for the marginal price curves. Defaults to 0.95.
        """
        self.intensity = intensity
        self.p_u_ref = p_u_ref
        self.eta = eta
        self.set_plants(plants)

    def refresh(self):
        """Recompute the cached arrays from the current plants and p_u_ref.

        The plants are mutated in place every year and p_u_ref moves with the parent's
        price, so every array derived from them has to be rebuilt afterwards.
        """
        self.capacities = np.array([plant.capacity for plant in self.plants], dtype=float)
        self.costs = np.array([plant.cost for plant in self.plants], dtype=float)
        self.plant_intensities = np.array([self.intensity if plant.intensity is None else plant.intensity
                                           for plant in self.plants], dtype=float)
        self.min_costs = self.costs - self.plant_intensities*self.p_u_ref

    def set_plants(self, plants):
        """Replace the plant list and refresh the cached arrays."""
        self.plants = np.array(list(plants), dtype=object)
        self.refresh()

    def add_plants(self, new_plants):
        """Append plants and refresh the cached arrays."""
        self.set_plants(list(self.plants) + list(new_plants))

    def costs_at(self, p_u=None):
        """Per-plant total cost at a given upstream price.

        Floored at MIN_COST_FACTOR of the reference cost so the curve stays defined (and positive)
        even when the solver probes an upstream price far below p_u_ref.

        Args:
            p_u (float, optional): Upstream price. Defaults to p_u_ref.

        Returns:
            array: Cost of each plant.
        """
        if p_u is None:
            p_u = self.p_u_ref

        return np.maximum(self.plant_intensities*p_u + self.min_costs, MIN_COST_FACTOR*self.costs)

    def _plant_quantities(self, p_d, p_u=None):
        """Per-plant quantities supplied at the given price(s).

        Args:
            p_d (array): Prices to evaluate the curve at.
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            array: Quantities per plant, shape (len(p_d), n_plants).
        """
        costs_for_p_u = self.costs_at(p_u)

        p_arr = np.atleast_1d(np.asarray(p_d, dtype=float))[:, np.newaxis]
        l = lambda x: 1.0 / (1.0 + np.exp(-SHARPNESS * (x / (MIN_PRICE_FACTOR * costs_for_p_u) - 1.0)))
        A = self.capacities / (l(costs_for_p_u) * costs_for_p_u ** self.eta)
        return l(p_arr) * A * p_arr ** self.eta  # (m, n_plants)

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
    
class DemandCurve():
    def __init__(self, p_nom, q_nom, elas):
        """Simple power law demand curve following A*p**elas.

        Args:
            A (float): Reference quantity
            elas (float): Price elasticity.
        """
        self.q_nom = q_nom
        self.p_nom = p_nom
        self.elas = elas
        self.refresh()

    def refresh(self):
        """Recompute A from the current p_nom / q_nom / elas.

        p_nom and q_nom are mutated in place each year, so A goes stale without this.
        """
        self.A = self.q_nom / self.p_nom**self.elas

    def __call__(self, p):
        return self.A*p**self.elas

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
            D = node.demand.A * p[i] ** node.demand.elas
        
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
    
    # Prices can't sit below cheapest plant
    lb = np.log(np.array([n.supply.costs.min() * 0.5 for n in nodes]))  
    
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