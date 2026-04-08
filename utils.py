import pandas as pd
import numpy as np


def initialize_population(N=1000, districts=5, initial_prevalence=0.05):
    N = 1000  # population size
    districts = 5
    
    initial_prevalence = 0.05  # initial infected fraction

    # age sampling: 0-4:15%, 5-14:20%, 15+:65%
    age_probs_full = np.zeros(80)
    age_probs_full[0:5] = 0.15/5
    age_probs_full[5:15] = 0.20/10
    age_probs_full[15:80] = 0.65/65

    ages = np.random.choice(range(0,80), size=N, p=age_probs_full)
    risk_profile = np.where(ages < 5, 'high', np.where(ages < 15, 'medium', 'low'))
    health_state = np.random.choice(['S', 'I'], size=N, p=[1-initial_prevalence, initial_prevalence])
    district_assignment = np.random.choice(range(districts), size=N)
    compliance = np.clip(np.random.normal(loc=0.9, scale=0.05, size=N), 0, 1)

    population = pd.DataFrame({
        'id': range(N),
        'age': ages,
        'district': district_assignment,
        'risk': risk_profile,
        'health_state': health_state,
        'compliance': compliance
    })
    return population
