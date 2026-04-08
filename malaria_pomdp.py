import pomdp_py
import numpy as np


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


class MalariaObservationModel(pomdp_py.ObservationModel):
    def __init__(self, population_df, n_districts):
        self.population_df = population_df
        self.districts = n_districts

    def probability(self, observation, next_state, action):
        return 1.0

    def sample(self, next_state, action):
        observed = {}
        for d in range(self.districts):
            states_in_district = [
                s for i,s in enumerate(next_state.individual_states)
                if self.population_df.at[i,'district']==d
            ]
            infected_count = states_in_district.count('I')
            observed[d] = np.random.binomial(infected_count, 0.5)
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