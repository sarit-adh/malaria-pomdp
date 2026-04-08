# Malaria POMDP Simulation

## Overview
This project implements a malaria control simulation using a **Partially Observable Markov Decision Process (POMDP)** framework with Python (`pomdp_py`).  
It models the effect of drug allocation on a population distributed across districts, incorporating:

- **Individual health states**: Susceptible (`S`), Infected (`I`), Recovered (`R`)  
- **Drug interventions**: Allocated per district with compliance probability  
- **Disease dynamics**: Infection spread and recovery (`disease_step`)  
- **Observations**: Noisy counts of infected individuals per district  
- **Reward model**: Incentivizes maximizing healthy population while minimizing drug usage

## POMDP Components

1. **`MalariaTransitionModel`** – Updates population state based on action and disease dynamics.
2. **`MalariaObservationModel`** – Simulates noisy observations of infection counts per district.
3. **`MalariaRewardModel`** – Computes reward based on healthy individuals and drug cost.
4. **`disease_step` function** – Simulates infection spread and recovery at each time step.

## Usage
1. Initialize your population DataFrame with `district` and `health_state` columns.
2. Define actions representing drug allocation per district.
3. Use `pomdp_py` to run simulations and evaluate policies.

