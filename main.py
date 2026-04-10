import matplotlib.pyplot as plt
from utils import initialize_population
from malaria_pomdp import MalariaState, MalariaTransitionModel, MalariaObservationModel, MalariaRewardModel
from policy import RandomPolicy, HeuristicPolicy
import pomdp_py
import numpy as np



time_steps = 20
districts = 5 
N = 100
initial_prevalence = 0.05 
population = initialize_population(N=N, districts=districts, initial_prevalence=initial_prevalence)

# Initialize models 
trans_model = MalariaTransitionModel(population) 
obs_model = MalariaObservationModel(population, districts) 
reward_model = MalariaRewardModel()

total_drugs_per_step = 50

num_particles = 100


def update_belief_particles(belief, action, observation,
                           trans_model, obs_model,
                           num_particles=100):

    new_particles = []
    weights = []
    

    # Step 1: propagate particles
    for s_prev in belief:
        #print("sampling next state from transition model for particle: ", s_prev)
        s_next = trans_model.sample(s_prev, action)

        # Step 2: weight by observation likelihood
        #print("computing observation likelihood for particle: ", s_next)
        w = obs_model.probability(observation, s_next, action)

        new_particles.append(s_next)
        weights.append(w)

    weights = np.array(weights)

    # Step 3: handle degenerate case
    if weights.sum() == 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()

    # Step 4: resample
    #print("resampling")
    indices = np.random.choice(len(new_particles),
                               size=num_particles,
                               p=weights)

    resampled_particles = [new_particles[i] for i in indices]

    return resampled_particles


def run_simulation(policy, population, trans_model, obs_model, reward_model,
                   time_steps, gamma=0.99):

    current_state = MalariaState(population['health_state'])

    belief = [current_state for _ in range(num_particles)]

    infection_history = []
    total_reward = 0.0
    discount = 1.0

    for t in range(time_steps):
        print("time step: ", t)
        
        if isinstance(policy, RandomPolicy):
            action = policy.sample(None)
        else:
            action = policy.sample(belief)

        # Environment step
        next_state = trans_model.sample(current_state, action)
        observation = obs_model.sample(next_state, action)

        # --- REWARD ---
        reward = reward_model.sample(current_state, action, next_state)
        total_reward += discount * reward
        discount *= gamma

        # Logging
        infected = sum(1 for s in next_state.individual_states if s == 'I')
        infection_history.append(infected)

        if isinstance(policy, RandomPolicy):
            current_state = next_state
            continue
        
        # Belief update
        belief = update_belief_particles(
            belief,
            action,
            observation,
            trans_model,
            obs_model,
            num_particles=num_particles
        )

        current_state = next_state

    return np.array(infection_history), total_reward


num_simulations = 10

random_runs = []
heuristic_runs = []

random_rewards = []
heuristic_rewards = []

for sim_num in range(num_simulations):

    population = initialize_population(
        N=N,
        districts=districts,
        initial_prevalence=initial_prevalence
    )

    trans_model = MalariaTransitionModel(population)
    obs_model = MalariaObservationModel(population, districts)
    reward_model = MalariaRewardModel()

    random_policy = RandomPolicy(total_drugs_per_step, districts)
    heuristic_policy = HeuristicPolicy(total_drugs_per_step, districts, population)

    print(f"Running simulation {sim_num+1}/{num_simulations} for random policy...")
    random_hist, random_reward = run_simulation(
        random_policy, population,
        trans_model, obs_model, reward_model,
        time_steps
    )

    print(f"Running simulation {sim_num+1}/{num_simulations} for heuristic policy...")
    heuristic_hist, heuristic_reward = run_simulation(
        heuristic_policy, population,
        trans_model, obs_model, reward_model,
        time_steps
    )

    random_runs.append(random_hist)
    heuristic_runs.append(heuristic_hist)

    random_rewards.append(random_reward)
    heuristic_rewards.append(heuristic_reward)


random_runs = np.array(random_runs)
heuristic_runs = np.array(heuristic_runs)

mean_random = random_runs.mean(axis=0)
std_random = random_runs.std(axis=0)

mean_heuristic = heuristic_runs.mean(axis=0)
std_heuristic = heuristic_runs.std(axis=0)


t = np.arange(time_steps)

plt.figure(figsize=(10,5))

# Random policy
plt.plot(t, mean_random, label="Random Policy", color='gray')
plt.fill_between(t,
                 mean_random - std_random,
                 mean_random + std_random,
                 color='gray', alpha=0.2)

# Heuristic policy
plt.plot(t, mean_heuristic, label="Heuristic Policy", color='blue')
plt.fill_between(t,
                 mean_heuristic - std_heuristic,
                 mean_heuristic + std_heuristic,
                 color='blue', alpha=0.2)

plt.xlabel("Time step")
plt.ylabel("Number of Infected Individuals")
plt.title("Random vs Heuristic Drug Allocation")
plt.legend()

plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()


print("\n=== POLICY COMPARISON (REWARD) ===")

print(f"Random Policy:    mean = {np.mean(random_rewards):.2f}, std = {np.std(random_rewards):.2f}")
print(f"Heuristic Policy: mean = {np.mean(heuristic_rewards):.2f}, std = {np.std(heuristic_rewards):.2f}")

plt.figure(figsize=(8,5))

plt.hist(random_rewards, alpha=0.5, label="Random Policy")
plt.hist(heuristic_rewards, alpha=0.5, label="Heuristic Policy")

plt.xlabel("Total Discounted Reward")
plt.ylabel("Frequency")
plt.title("Policy Comparison via Reward")
plt.legend()

plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()
