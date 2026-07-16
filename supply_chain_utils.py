import numpy as np
from scipy.optimize import root_scalar, least_squares

# Parameters to generate random cost curves. Taken from global Cu cost curve from S&P
MINE_DISTRIBUTION_MEAN = np.array([-7.6971, 0.323])
MINE_DISTRIBUTION_COV = np.array([[ 4.6362, -0.0874],
                                  [-0.0874,  0.0164]])
MINE_DISTRIBUTION_COST_MIN = 0.0647

MIN_PRICE_FACTOR = 0.95
SHARPNESS = 500

class SupplyPlant:
    def __init__(self, capacity, cost):
        self.capacity = capacity
        self.cost = cost

class Mine(SupplyPlant):
    def __init__(self, capacity, cost, ore_grade=None):
        super().__init__(capacity, cost)
        self.ore_grade = ore_grade

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

# def marginal_curve(p, eta, cost_ref, capacity_ref):
#     """Marginal cost curve function for a given capacity and cost.

#     Args:
#         p (array of float): Price to evaluate the function at.
#         eta (float): Price elasticity.
#         cost_ref (float): Reference cost for the plant.
#         capacity_ref (float): Reference capacity for the plant.
#         min_price (float, optional): Price floor below which no production will happen. Defaults to 0.95.
#         sharpness (float, optional): Steepness of the logistic modulation. Defaults to 500.

#     Returns:
#         array of float: Production quantities at a given marginal price.
#     """

#     # Logistic modulation to smooth out the transition to 0
#     sharpness = 500 #1000/p_min
#     l = lambda x: 1 / (1 + np.exp(-sharpness * (x / (MIN_PRICE_FACTOR * cost_ref) - 1)))
    
#     # Rescale A so that l(p_ref) * A * p_ref**eta == q_ref
#     l_ref = l(cost_ref)
#     A = capacity_ref / (l_ref * cost_ref**eta)
    
#     return l(p) * A * p**eta

class SupplyCurve:
    def __init__(self, plants, eta=0.1):
        """Basic supply curve composed of a horizontal sum of marginal price curves.

        Args:
            plants (list): List of supply plants.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
            min_price (float, optional): Price floor for the marginal price curves. Defaults to 0.95.
        """
        self.plants = np.array(plants)
        self.capacities = np.array([plant.capacity for plant in self.plants])
        self.costs = np.array([plant.cost for plant in self.plants])
        self.eta = eta

    def __call__(self, p):
        """Evaluation of the supply curve.

        Args:
            p (array): Prices to evaluate the curve at.

        Returns:
            array: Resulting quantities for each price.
        """

        p_arr = np.atleast_1d(np.asarray(p, dtype=float))[:, np.newaxis]  # (m, 1) broadcasts against costs (n_plants,)
        l = lambda x: 1.0 / (1.0 + np.exp(-SHARPNESS * (x / (MIN_PRICE_FACTOR * self.costs) - 1.0)))
        A = self.capacities / (l(self.costs) * self.costs ** self.eta)
        supply = np.sum(l(p_arr) * A * p_arr ** self.eta, axis=-1)  # (m,) -- one total per price
        return supply.item() if np.isscalar(p) else supply

class SupplyCurve2D():
    def __init__(self, plants, intensity, p_u_ref, eta=0.1):
        """Supply curve with changing upstream price.

        Args:
            plants (list): List of supply plants.
            intensity (float): Conversion factor between this stage and the upstream one.
            p_u_ref (float): Reference price at which the costs are defined.
            eta (float, optional): Elasticity for the marginal price curves. Defaults to 0.1.
            min_price (float, optional): Price floor for the marginal price curves. Defaults to 0.95.
        """
        self.plants = np.array(plants)
        self.capacities = np.array([plant.capacity for plant in self.plants])
        self.costs = np.array([plant.cost for plant in self.plants])
        self.intensity = intensity
        self.p_u_ref = p_u_ref
        self.min_costs = self.costs - intensity*p_u_ref
        self.eta = eta

    def __call__(self, p_d, p_u=None):
        """Evaluation of the supply curve.

        Args:
            p_d (array): Prices to evaluate the curve at.
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            array: Resulting quantities for each price.
        """
        if p_u is None:
            p_u = self.p_u_ref

        supply = np.zeros_like(p_d).astype(float)

        costs_for_p_u = self.intensity*p_u + self.min_costs

        p_arr = np.atleast_1d(np.asarray(p_d, dtype=float))[:, np.newaxis]  # (m, 1) broadcasts against costs (n_plants,)
        l = lambda x: 1.0 / (1.0 + np.exp(-SHARPNESS * (x / (MIN_PRICE_FACTOR * costs_for_p_u) - 1.0)))
        A = self.capacities / (l(costs_for_p_u) * costs_for_p_u ** self.eta)
        supply = np.sum(l(p_arr) * A * p_arr ** self.eta, axis=-1)  # (m,) -- one total per price
        
        return supply.item() if np.isscalar(p_d) else supply
    
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

class SupplyChainStage():
    def __init__(self, supply, demand, children=[]):
        """Representation of one supply chain stage.

        Args:
            supply (callable): Supply for that stage.
            demand (callable): Demand for that stage.
            children (list, optional): Downstream stages that depend on the price of this stage. Defaults to [].
        """
        self.supply = supply
        self.demand = demand
        self.children = children
        self.p_eq = None
        
    def get_equilibrium_price(self, p_u=None):
        """Find the equilibrium price (intersection of supply and demand) for this stage. At an upstream price p_u if need be.

        Args:
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            float: Equilibrium price.
        """

        # raise Exception("This function is deprecated. Use solve_chain() to solve the entire supply chain simultaneously.")

        if isinstance(self.supply, SupplyCurve2D):
            p_u_eff = p_u if p_u is not None else self.supply.p_u_ref
            costs = self.supply.intensity * p_u_eff + self.supply.min_costs
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
        if self.supply.__class__ == SupplyCurve2D:
            costs_for_p_u = self.supply.intensity * (p_u if p_u is not None else self.supply.p_u_ref) + self.supply.min_costs
            y = np.linspace(1e-20, 1.5*costs_for_p_u.max(), 200)
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

class CompositeDemand():
    def __init__(self, downstreams, other=0):
        """Summs the demand from one or more downstream stages.

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
                supply[i] += downstream.supply.intensity * downstream.supply(downstream.get_equilibrium_price(pi), p_u=pi)

        supply += self.other
        
        return supply.item() if scalar_input else supply

def build_chain(root_node):
    """
    Walk the supply chain stage graph to build relationships. Returns the ordered stage list
    as well as parent[] and children[] index lists.
    """

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
    """
    Flat market-clearing residuals (supply - demand) per stage, in log-price space.
    logp is an array of len(nodes).
    Normalised by fixed stage capacity so every residual is O(1).
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
            D = node.demand.other + sum(nodes[c].supply.intensity * S[c] for c in children[i])
        else:
            D = node.demand.A * p[i] ** node.demand.elas
        
        # No matter the demand, subtract the supply and scale it for better convergence
        res[i] = (S[i] - D) / scale[i]
    
    return res

def solve_chain(nodes, parent, children, p0=None):
    """Solve an entire supply chain simultaneously for all equilibrium prices.

    Returns ({stage: price}, scipy_result) and also writes each price to stage.p_eq.
    Robust to the initial guess; pass p0 (array over stages, root first) to override the
    default of each stage's mean plant cost."""
    
    # Prices can't sit below cheapest plant
    lb = np.log(np.array([n.supply.costs.min() * 0.5 for n in nodes]))  
    
    # Get x0 as log or create it from the means
    if p0 is None:
        p0 = np.array([n.supply.costs.mean() for n in nodes])
    x0 = np.clip(np.log(np.asarray(p0, dtype=float)), lb, None)
    
    # Solve
    sol = least_squares(_residuals, x0, args=(nodes, parent, children), method="trf", bounds=(lb, np.inf), xtol=1e-12, ftol=1e-12)
    
    # Return prices and write them to the nodes
    p = np.exp(sol.x)
    
    return {n: pi for n, pi in zip(nodes, p)}