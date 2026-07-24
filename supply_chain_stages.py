import numpy as np
from utils import CompositeDemand, SupplyCurve, SupplyCurve2D, DemandCurve, find_intersection

CHANGE = 0.01

class SupplyPlant:
    def __init__(self, capacity, cost, intensity=None):
        """Generic type for a plant that produces a material.

        Args:
            capacity (float): Avaialble capacity.
            cost (float): Marginal cost
            intensity (float, optional): How much of the upstream material is needed for production. Defaults to None.
        """
        self.capacity = capacity
        self.cost = cost
        self.intensity = intensity

    def update(self, ctx):
        return type(self)(self.capacity, self.cost / (1-CHANGE), intensity=self.intensity)

class Mine(SupplyPlant):
    def __init__(self, capacity, cost, ore_grade=None):
        """Subtype of plant specifically for a mining stage.

        Args:
            capacity (float): Avaialble capacity.
            cost (float): Marginal cost
            ore_grade (float, optional): Ore grade of the mine. Defaults to None.
        """
        super().__init__(capacity, cost)
        self.ore_grade = ore_grade

    def update(self, ctx):
        return Mine(self.capacity * (1-CHANGE), self.cost / (1-CHANGE), ore_grade=self.ore_grade)

class WasteStream(SupplyPlant):
    def __init__(self, capacity, cost, capacity_ratio=None, source=None, mode="flow", stock=None):
        """Subtype of plant specifically for a waste stream stage.

        Args:
            capacity (float): Available capacity. In stock mode this is the current reservoir.
            cost (float): Marginal cost.
            capacity_ratio (float): Fraction of the source stage's production that becomes collectable scrap.
            source (str, optional): Name of the stage whose production feeds this stream. By default it is the stream's own stage.
            mode (str): "flow" (capacity = this year's inflow) or "stock" (capacity = running reservoir).
            stock (float, optional): Current reservoir size (stock mode only). Defaults to the initial capacity.
        """
        # Recycled feedstock: doesn't consume the upstream input by default, so its cost is independent of the upstream price
        super().__init__(capacity, cost, intensity=0)
        self.capacity_ratio = capacity_ratio
        self.source = source
        self.mode = mode
        self.stock = capacity if stock is None else stock

    def update(self, ctx):
        # Inflow feeds off the source stage's production; source=None means this stream's own stage
        source = self.source if self.source is not None else ctx.stage_name_of(self)
        inflow = self.capacity_ratio * ctx.production_of(source)

        if self.mode == "stock":
            # Reservoir grows by inflow, shrinks by what this plant actually recycled last year
            new_stock = max(self.stock + inflow - ctx.consumed_by(self), 0.0)
            new_capacity = new_stock
        else:
            new_stock = None
            new_capacity = inflow

        return WasteStream(new_capacity, self.cost / (1-CHANGE), capacity_ratio=self.capacity_ratio,
                           source=self.source, mode=self.mode, stock=new_stock)


class SupplyChainStage():
    def __init__(self, supply, demand, name):
        """Representation of one supply chain stage.

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

    def update(self, new_children=None, ctx=None):
        """Upates the stage for the next iteration."""

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot update stage.")

        return type(self)(self._new_supply(new_children, ctx), self._new_demand(new_children), name=self.name)

    def add_plants(self, new_plants):
        """Append plants to this stage, rebuilding the supply curve so its cached arrays stay consistent.

        Args:
            new_plants (list): Plants to add to the stage's supply.
        """
        plants = np.concatenate([self.supply.plants, new_plants])
        if isinstance(self.supply, SupplyCurve2D):
            self.supply = SupplyCurve2D(plants, intensity=self.supply.intensity,
                                        p_u_ref=self.supply.p_u_ref, eta=self.supply.eta)
        else:
            self.supply = SupplyCurve(plants, eta=self.supply.eta)

    def _new_supply(self, new_children, ctx=None):
        """Depending on the stage type, update supply."""

        # If the stage is mid or downstream, delegate to each plant's own update rule
        if isinstance(self.supply, SupplyCurve2D):
            new_plants = [p.update(ctx) for p in self.supply.plants]
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
        return DemandCurve(p_nom=self.p_eq, q_nom=self.demand.q_nom * (1+CHANGE), elas=self.demand.elas)

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
    def __init__(self, supply, demand, incentive_pool, resource_pool, incentive_ratio, name):
        """Subtype of a SupplyChainStage representing a mining stage.

        Args:
            supply (callable): Supply for that stage.
            demand (callable): Demand for that stage.
            incentive_pool (array): List of mines making up the incentive pool.
            resource_pool (array): List of mines making up the resource pool.
            incentive_ratio (float): Ratio of incentive pool capacity to total supply capacity.
            name (str): Human-readable stage name.
        """

        super().__init__(supply, demand, name)
        self.incentive_pool = incentive_pool
        self.resource_pool = resource_pool
        self.incentive_ratio = incentive_ratio

        # Check that the incentive pool capacity is consistent with the specified incentive ratio
        real_incentive_ratio = sum(mine.capacity for mine in incentive_pool) / self.supply.capacities.sum()
        if np.around(real_incentive_ratio, 2) < np.around(incentive_ratio, 2):
            raise ValueError(f"Incentive pool capacity is smaller than the specified incentive ratio ({real_incentive_ratio:.2f} vs {incentive_ratio:.2f}).")

    def update(self, new_children, ctx=None):
        """Update the mines for the next iteration."""

        if self.p_eq is None:
            raise ValueError("Equilibrium price not set, cannot update stage.")

        # Change mine parameters and filter out mines that are too expensive to operate at the current price
        new_upstream_mines = [mine.update(ctx) for mine in self.supply.plants]
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
        new_stage = MiningStage(new_supply, new_demand, incentive_mines, resource_mines,
                                incentive_ratio=self.incentive_ratio, name=self.name)

        return new_stage
