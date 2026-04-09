import numpy as np
from malaria_pomdp import MalariaAction


# Behaves like random rollout
class RandomPolicy:
    def __init__(self, total_drugs, districts):
        self.total_drugs = total_drugs
        self.districts = districts
    def sample(self, belief):
        # Ignores belief and considers random observed infections
        observed_infections = {d: np.random.randint(1,10) for d in range(self.districts)}
        total_obs = sum(observed_infections.values())
        allocation = {d: int(self.total_drugs*count/total_obs) for d,count in observed_infections.items()}
        return MalariaAction(allocation)
    
    
class HeuristicPolicy:
    def __init__(self, total_drugs, districts, population_df,
                 alpha=1.0,   # weight on expected infections
                 beta=0.0):   # weight on uncertainty (std dev)
        """
        Parameters
        ----------
        total_drugs : int
        districts : int
        population_df : pd.DataFrame (must have 'district' column aligned with state indexing)
        alpha : float
            weight on expected infections
        beta : float
            weight on uncertainty (exploration bonus)
        """
        self.total_drugs = total_drugs
        self.districts = districts
        self.population_df = population_df
        self.alpha = alpha
        self.beta = beta

        # Precompute index lists per district for efficiency
        self.idxs_by_district = {
            d: [i for i in range(len(population_df))
                if population_df.at[i, 'district'] == d]
            for d in range(districts)
        }

    # -----------------------------
    # Utilities to extract belief stats
    # -----------------------------
    def _get_particles_and_weights(self, belief):
        """
        Supports:
        - Histogram belief: belief.items() -> (state, prob)
        - Particle belief: belief.particles (uniform weights assumed)
        """
        if hasattr(belief, "items"):
            items = list(belief.items())
            states = [s for s, _ in items]
            weights = np.array([w for _, w in items], dtype=float)
            weights = weights / weights.sum()
            return states, weights
            # list of particles 
        
        if isinstance(belief, list):
            states = belief
            weights = np.ones(len(states)) / len(states)
            return states, weights
        
        # pomdp_py Histogram
        if hasattr(belief, "get_histogram"):
            hist = belief.get_histogram()
            states = list(hist.keys())
            weights = np.array(list(hist.values()), dtype=float)
            weights = weights / weights.sum()
            return states, weights

        elif hasattr(belief, "particles"):
            states = list(belief.particles)
            n = len(states)
            weights = np.ones(n) / n
            return states, weights

        else:
            raise ValueError("Unsupported belief type")

    def _expected_and_std_infected(self, belief):
        """
        Returns per-district:
        - expected infected
        - std deviation (uncertainty)
        """
        states, weights = self._get_particles_and_weights(belief)

        exp_inf = np.zeros(self.districts)
        exp_inf_sq = np.zeros(self.districts)

        for s, w in zip(states, weights):
            # count infected per district for this state
            for d in range(self.districts):
                idxs = self.idxs_by_district[d]
                infected = sum(1 for i in idxs if s.individual_states[i] == 'I')

                exp_inf[d] += w * infected
                exp_inf_sq[d] += w * (infected ** 2)

        var = np.maximum(exp_inf_sq - exp_inf**2, 0.0)
        std = np.sqrt(var)

        return exp_inf, std

    # -----------------------------
    # Main policy
    # -----------------------------
    def sample(self, belief):
        # 1. Compute belief statistics
        exp_inf, std_inf = self._expected_and_std_infected(belief)

        # 2. Score districts
        #    exploitation (mean) + exploration (uncertainty)
        scores = self.alpha * exp_inf + self.beta * std_inf

        # Avoid all-zero case
        if scores.sum() == 0:
            scores = np.ones(self.districts)

        # 3. Normalize to allocation proportions
        proportions = scores / scores.sum()

        # 4. Initial integer allocation
        raw_alloc = proportions * self.total_drugs
        alloc = np.floor(raw_alloc).astype(int)

        # 5. Distribute remaining drugs (to match total exactly)
        remaining = self.total_drugs - alloc.sum()
        if remaining > 0:
            # assign leftover to highest fractional parts
            fractional = raw_alloc - alloc
            order = np.argsort(-fractional)
            for i in range(remaining):
                alloc[order[i]] += 1

        allocation = {d: int(alloc[d]) for d in range(self.districts)}

        return MalariaAction(allocation)