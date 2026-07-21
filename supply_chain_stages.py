import numpy as np
from utils import CompositeDemand, SupplyCurve, SupplyCurve2D, DemandCurve, find_intersection


class SupplyPlant:
    def __init__(self, capacity, cost, intensity=None):
        self.capacity = capacity
        self.cost = cost
        self.intensity = intensity  # None -> inherits the stage's default intensity

    def update(self, **kwargs):
        return type(self)(self.capacity, self.cost * 1.025, intensity=self.intensity)

class Mine(SupplyPlant):
    def __init__(self, capacity, cost, ore_grade=None):
        super().__init__(capacity, cost)
        self.ore_grade = ore_grade

    def update(self, **kwargs):
        return Mine(self.capacity * 0.975, self.cost * 1.025, ore_grade=self.ore_grade)

class WasteStream(SupplyPlant):
    def __init__(self, capacity, cost, capacity_ratio=None):
        # Recycled feedstock: doesn't consume the upstream input by default, so its cost is independent of the upstream price
        super().__init__(capacity, cost, intensity=0)
        self.capacity_ratio = capacity_ratio

    def update(self, total_production=None, **kwargs):
        return WasteStream(self.capacity_ratio * total_production, self.cost * 1.025, capacity_ratio=self.capacity_ratio, intensity=self.intensity)


class SupplyChainStage():
    def __init__(self, supply, demand):
        """Representation of one supply chain stage.

        Args:
            supply (callable): Supply for that stage.
            demand (callable): Demand for that stage.
        """

        self.supply = supply
        self.demand = demand
        self.parent = None
        if isinstance(demand, CompositeDemand):
            for downstream in demand.downstreams:
                downstream.parent = self
        self.p_eq = None

    def update(self, new_children=None):
        """Upates the stage for the next iteration."""

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot update stage.")

        return type(self)(self._new_supply(new_children), self._new_demand(new_children))

    def _new_supply(self, new_children):
        """Depending on the stage type, update supply."""

        # If the stage is mid or downstream, delegate to each plant's own update rule
        if isinstance(self.supply, SupplyCurve2D):
            total_production = self.supply(self.p_eq)
            new_plants = [p.update(total_production=total_production) for p in self.supply.plants]
            return SupplyCurve2D(new_plants, intensity=self.supply.intensity, p_u_ref=self.parent.p_eq, eta=self.supply.eta)
        
        # If isn't, we keep the same supply
        return self.supply 

    def _new_demand(self, new_children):
        """Depending on the stage type, update demand."""

        # If the stage is up or midstream, we don't need to update demand
        if isinstance(self.demand, CompositeDemand):
            return CompositeDemand(new_children, other=self.demand.other)
        
        # TODO: make demand curve update more sophisticated
        # If the stage is a leaf stage (all the way downstream), then we update the demand curve
        return DemandCurve(p_nom=self.p_eq, q_nom=self.demand.q_nom * 1.025, elas=self.demand.elas)

    def get_equilibrium_price(self, p_u=None):
        """Find the equilibrium price (intersection of supply and demand) for this stage. At an upstream price p_u if need be.

        Args:
            p_u (float, optional): Upstream price. Defaults to None.

        Returns:
            float: Equilibrium price.
        """

        if isinstance(self.supply, SupplyCurve2D):
            p_u_eff = p_u if p_u is not None else self.supply.p_u_ref
            costs = self.supply.plant_intensities * p_u_eff + self.supply.min_costs
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
            costs_for_p_u = self.supply.plant_intensities * (p_u if p_u is not None else self.supply.p_u_ref) + self.supply.min_costs
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


class MiningStage(SupplyChainStage):
    def __init__(self, supply, demand, incentive_pool, resource_pool, incentive_ratio):
        """Subtype of a SupplyChainStage representing a mining stage.

        Args:
            supply (callable): Supply for that stage.
            demand (callable): Demand for that stage.
            incentive_pool (array): List of mines making up the incentive pool.
            resource_pool (array): List of mines making up the resource pool.
            incentive_ratio (float): Ratio of incentive pool capacity to total supply capacity.
        """

        super().__init__(supply, demand)
        self.incentive_pool = incentive_pool
        self.resource_pool = resource_pool
        self.incentive_ratio = incentive_ratio

        # Check that the incentive pool capacity is consistent with the specified incentive ratio
        real_incentive_ratio = sum(mine.capacity for mine in incentive_pool) / self.supply.capacities.sum()
        if np.around(real_incentive_ratio, 2) < np.around(incentive_ratio, 2):
            raise ValueError(f"Incentive pool capacity is smaller than the specified incentive ratio ({real_incentive_ratio:.2f} vs {incentive_ratio:.2f}).")

    def update(self, new_children):
        """Update the mines for the next iteration."""

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot update stage.")

        # Change mine parameters and filter out mines that are too expensive to operate at the current price
        new_upstream_mines = [mine.update() for mine in self.supply.plants]
        new_upstream_mines_left = np.array([mine for mine in new_upstream_mines if mine.cost <= self.p_eq])

        # Calculate supply gap based on temp supply and demand
        temp_supply_curve = SupplyCurve(new_upstream_mines_left)
        new_demand = CompositeDemand(new_children)
        supply_gap = new_demand(self.p_eq) - temp_supply_curve(self.p_eq)

        # Sort incentive mines and find which ones to open based on supply gap
        sorted_incentive_mines = np.array(sorted(self.incentive_pool, key=lambda x: x.cost))
        sorted_incentive_capacities = np.array([mine.capacity for mine in sorted_incentive_mines])
        to_open_mask = (np.cumsum(sorted_incentive_capacities) - sorted_incentive_capacities) <= supply_gap
        new_upstream_mines = np.concatenate([new_upstream_mines_left, sorted_incentive_mines[to_open_mask]])

        # Based on this, create the new upstream
        new_supply = SupplyCurve(new_upstream_mines)

        # Replenish the incentive curve with new mines
        incentive_mines_left = sorted_incentive_mines[~to_open_mask]
        incentive_gap = new_supply.capacities.sum()*self.incentive_ratio - sum(mine.capacity for mine in incentive_mines_left)
        sorted_resource_mines = np.array(sorted(self.resource_pool, key=lambda x: x.cost))
        sorted_resource_capacities = np.array([mine.capacity for mine in sorted_resource_mines])
        to_transfer_mask = (np.cumsum(sorted_resource_capacities) - sorted_resource_capacities) <= incentive_gap
        incentive_mines = np.concatenate([incentive_mines_left, sorted_resource_mines[to_transfer_mask]])
        resource_mines = sorted_resource_mines[~to_transfer_mask]

        # Create the new MiningStage with updated supply, demand, incentive pool, and resource pool
        new_stage = MiningStage(new_supply, new_demand, incentive_mines, resource_mines, incentive_ratio=self.incentive_ratio)

        return new_stage
