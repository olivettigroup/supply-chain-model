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

        # Name -> index map. Stage order is stable across years, so this resolves names for every snapshot.
        self.stage_idx = {}
        for k, s in enumerate(initial_stages):
            if s.name in self.stage_idx:
                raise ValueError(f"Duplicate stage name {s.name!r}; stage names must be unique.")
            self.stage_idx[s.name] = k

        self.verbosity = verbosity

    def stage(self, name, t):
        """Get a stage by name for a given year."""
        return self.stages_history[t][self.stage_idx[name]]

    @property
    def stage_names(self):
        """Stage names in canonical (snapshot) order."""
        return list(self.stage_idx)

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

    def run(self, n_iterations, scenario=None):
        """Run the supply chain simulation for a given number of iterations (years).

        Args:
            n_iterations (int): Number of years to simulate after the initial one.
            scenario (callable, optional): Scenario callback ``scenario(chain, t)`` invoked each year after that
                year's stages are built but before prices are solved. Defaults to None (baseline run).
        """

        for i in range(n_iterations + 1):
            t = self.initial_year + i
            if self.verbosity > 0: print(f"Year {t}")

            # The initial year's snapshot already exists from __init__; later years are built from the previous one
            if i > 0:
                # Context built from last year's solved chain, so plants can look up other stages' production by name
                ctx = StageContext(self.stages_history[t-1])

                new_snapshot = [None] * len(self.stages_history[t-1])
                for k in reversed(range(len(self.stages_history[t-1]))):
                    new_children = [new_snapshot[c] for c in self.children[k]]
                    new_snapshot[k] = self.stages_history[t-1][k].update(new_children, ctx=ctx)
                self.stages_history[t] = new_snapshot

            # Scenario hook can mutate this year's stages before they're solved
            if scenario is not None:
                scenario(self, t)

            self.solve_prices(t)


class StageContext:
    def __init__(self, stages):
        """Read-only view of the just-solved chain, so a plant's update() can look up global info by name.

        Args:
            stages (list): The current year's solved SupplyChainStages (each with p_eq set).
        """
        self._by_name = {s.name: s for s in stages}

        # Per-plant realized output at equilibrium, and which stage each plant belongs to, keyed by plant identity
        self._produced = {}
        self._stage_name = {}
        for s in stages:
            quantities = s.supply._plant_quantities(s.p_eq).ravel()
            for plant, q in zip(s.supply.plants, quantities):
                self._produced[id(plant)] = q
                self._stage_name[id(plant)] = s.name

    def production_of(self, name):
        """This year's total production of the named stage."""
        stage = self._by_name[name]
        return stage.supply(stage.p_eq)

    def stage_name_of(self, plant):
        """Name of the stage the given plant belongs to."""
        return self._stage_name[id(plant)]

    def consumed_by(self, plant):
        """This plant's own realized output at equilibrium (what it actually produced/consumed last year)."""
        return self._produced[id(plant)]