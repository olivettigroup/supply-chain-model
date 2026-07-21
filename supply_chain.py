from utils import build_chain, solve_chain
from supply_chain_stages import MiningStage

class SupplyChain:
    def __init__(self, input_stages, initial_year=2026, verbosity=0):
        """Simulates a supply chain with various supply stages througout time.

        Args:
            input_stages (array): List of SupplyChainStages
            initial_year (int, optional): Year to start simulations for. Defaults to 2026.
        """
        initial_stages, self.parent, self.children = build_chain(input_stages[0])
        self.initial_year = initial_year
        self.stages_history = {initial_year: initial_stages}
        self.mine_idx = [k for k, s in enumerate(initial_stages) if isinstance(s, MiningStage)]
        self.plant_idx = [k for k, s in enumerate(initial_stages) if not isinstance(s, MiningStage)]
        self.verbosity = verbosity

    def plant_stages(self, t):
        """Handy helper to get the plant stages for a given year."""
        return [self.stages_history[t][k] for k in self.plant_idx]

    def mine_stages(self, t):
        """Handy helper to get the mine stages for a given year."""
        return [self.stages_history[t][k] for k in self.mine_idx]

    def solve_prices(self, t):
        """Calls solve_chain to solve for the equilibrium prices at a given year and updates the stages with the new prices."""
        res = solve_chain(self.stages_history[t], self.parent, self.children)
        for node, pi in res.items():
            node.p_eq = pi

    def run(self, n_iterations):
        """Run the supply chain simulation for a given number of iterations (years)."""

        # For the initial year, no update needed, just price solving
        t = self.initial_year
        if self.verbosity > 0: print(f"Year {t}")
        self.solve_prices(t)

        # Every year after that, also update the stages
        for i in range(1, n_iterations+1):
            
            t = self.initial_year + i
            if self.verbosity > 0: print(f"Year {t}")

            # Update the stages
            new_snapshot = [None] * len(self.stages_history[t-1])
            for k in reversed(range(len(self.stages_history[t-1]))):
                new_children = [new_snapshot[c] for c in self.children[k]]
                new_snapshot[k] = self.stages_history[t-1][k].update(new_children)
            
            # Store the new stages and solve for the new equilibrium prices
            self.stages_history[t] = new_snapshot
            self.solve_prices(t)