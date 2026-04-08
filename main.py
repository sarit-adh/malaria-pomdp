import matplotlib.pyplot as plt
from utils import initialize_population
from malaria_pomdp import MalariaState, MalariaTransitionModel, MalariaObservationModel, MalariaRewardModel
from policy import HeuristicPolicy



time_steps = 30
districts = 5
N = 1000
initial_prevalence = 0.05
population = initialize_population(N=N, districts=districts, initial_prevalence=initial_prevalence)


# Initialize models
trans_model = MalariaTransitionModel(population)
obs_model = MalariaObservationModel(population, districts)
reward_model = MalariaRewardModel()

# Define Policy
total_drugs_per_step = 50
policy = HeuristicPolicy(total_drugs_per_step, districts)


# -------------------------
# Simulation Loop
# -------------------------

# Initial state
current_state = MalariaState(population['health_state'])

infection_history = []

for t in range(time_steps):
    action = policy.sample(None)
    next_state = trans_model.sample(current_state, action)
    reward = reward_model.sample(current_state, action, next_state)
    observation = obs_model.sample(next_state, action)
    infection_history.append(
        sum([1 for s in next_state.individual_states if s=='I'])
    )
    current_state = next_state


# -------------------------
# Visualization
# -------------------------
plt.plot(range(time_steps), infection_history, marker='o')
plt.xlabel("Time step")
plt.ylabel("Number of Infected Individuals")
plt.title("Malaria Infection Dynamics under POMDP Allocation")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
