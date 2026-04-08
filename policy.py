import numpy as np
from malaria_pomdp import MalariaAction

class HeuristicPolicy:
    def __init__(self, total_drugs, districts):
        self.total_drugs = total_drugs
        self.districts = districts
    def sample(self, belief):
        # For simplicity: random observed infections
        observed_infections = {d: np.random.randint(1,10) for d in range(self.districts)}
        total_obs = sum(observed_infections.values())
        allocation = {d: int(self.total_drugs*count/total_obs) for d,count in observed_infections.items()}
        return MalariaAction(allocation)