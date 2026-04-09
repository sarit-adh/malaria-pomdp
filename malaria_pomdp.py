import pomdp_py
import numpy as np
from scipy.stats import binom, norm


class MalariaState(pomdp_py.State):
    def __init__(self, individual_states):
        self.individual_states = tuple(individual_states)
    def __hash__(self):
        return hash(self.individual_states)
    def __eq__(self, other):
        return self.individual_states == other.individual_states

class MalariaAction(pomdp_py.Action):
    def __init__(self, allocation_dict):
        """allocation_dict: {district: number_of_drugs}"""
        self.allocation = allocation_dict

class MalariaObservation(pomdp_py.Observation):
    def __init__(self, observed_infected):
        """observed_infected: {district: number of observed symptomatic cases}"""
        self.observed_infected = observed_infected

class MalariaTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, population_df):
        self.population_df = population_df.copy() 

    def probability(self, next_state, state, action):
        return 1.0

    def sample(self, state, action):
        pop_copy = self.population_df.copy()
        for district, drugs in action.allocation.items():
            district_pop = pop_copy[pop_copy['district']==district]
            susceptible = district_pop[district_pop['health_state']=='S']
            treatable = susceptible.sample(min(drugs, len(susceptible)))
            for idx in treatable.index:
                if np.random.rand() < pop_copy.at[idx, 'compliance']:
                    pop_copy.at[idx,'health_state'] = 'R'
        pop_copy = disease_step(pop_copy)
        return MalariaState(pop_copy['health_state'])


class MalariaObservation(pomdp_py.Observation):
    def __init__(self, observed_infected):
        """observed_infected: {district: number of observed symptomatic cases}"""
        self.observed_infected = observed_infected

    def __eq__(self, other):
        return isinstance(other, MalariaObservation) and \
               self.observed_infected == other.observed_infected

    def __hash__(self):
        return hash(tuple(sorted(self.observed_infected.items())))

    def __repr__(self):
        return f"MalariaObservation({self.observed_infected})"


class MalariaObservationModel(pomdp_py.ObservationModel):
    def __init__(self, population_df, n_districts, p_detect=0.5):
        """
        Parameters
        ----------
        p_detect : float or dict
            Detection probability (global or per-district)
        """
        self.population_df = population_df
        self.districts = n_districts
        self.p_detect = p_detect

    def _get_p_detect(self, d):
        """Supports global or per-district detection probability"""
        if isinstance(self.p_detect, dict):
            return self.p_detect.get(d, 0.5)
        return self.p_detect

    def probability(self, observation, next_state, action):
        """
        Computes P(o | s')
        Assumes:
            observed_infected[d] ~ Binomial(true_infected[d], p_detect)
        """
        prob = 1.0

        for d in range(self.districts):
            # Get individuals in district d
            idxs = [
                i for i in range(len(next_state.individual_states))
                if self.population_df.at[i, 'district'] == d
            ]

            states = [next_state.individual_states[i] for i in idxs]

            # Count true infected
            infected = sum(1 for s in states if s == 'I')

            # Observed count
            obs_val = observation.observed_infected.get(d, 0)

            p_detect = self._get_p_detect(d)

            # Binomial likelihood
            prob *= binom.pmf(obs_val, infected, p_detect)

        return prob

    def sample(self, next_state, action):
        """
        Sample observation from Binomial thinning
        """
        observed = {}

        for d in range(self.districts):
            idxs = [
                i for i in range(len(next_state.individual_states))
                if self.population_df.at[i, 'district'] == d
            ]

            states = [next_state.individual_states[i] for i in idxs]

            infected = sum(1 for s in states if s == 'I')

            p_detect = self._get_p_detect(d)

            # Sample observed infected
            observed_count = np.random.binomial(infected, p_detect)

            observed[d] = observed_count

        return MalariaObservation(observed)

class MalariaRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        healthy_count = next_state.individual_states.count('S') + next_state.individual_states.count('R')
        drugs_used = sum(action.allocation.values())
        return healthy_count - 0.1*drugs_used
    
def disease_step(pop, transmission_rate=0.02, recovery_rate=0.1):
    """Update health states based on transmission and recovery."""
    new_states = []
    for idx, row in pop.iterrows():
        if row['health_state'] == 'S':
            district_pop = pop[pop['district'] == row['district']]
            infected_count = (district_pop['health_state'] == 'I').sum()
            infection_prob = 1 - (1 - transmission_rate)**infected_count
            new_states.append('I' if np.random.rand() < infection_prob else 'S')
        elif row['health_state'] == 'I':
            new_states.append('R' if np.random.rand() < recovery_rate else 'I')
        else:
            new_states.append(row['health_state'])
    pop['health_state'] = new_states
    return pop