# Simulation-with-py


# Tool Rental Simulation Project

This project is a discrete-event simulation of a tool rental facility, implemented in Python. The simulation models the process of customers arriving, renting tools, returning them, and the subsequent maintenance and cleaning operations performed by staff members.

##  Overview

The simulation explores **three different staffing scenarios**, each with a unique assignment of roles among workers:

- **Scenario 1:** Basic two-worker setup with specific responsibilities.
- **Scenario 2:** Modified worker roles to test alternative task handling strategies.
- **Scenario 3:** Introduction of a third flexible worker ("Roham") who dynamically assists with rental and maintenance tasks.

Each scenario is simulated 30 times using different random seeds for statistical validity. The simulation tracks key performance indicators such as:

- Average customer wait time
- Rental durations by worker
- Maintenance and cleaning times
- Queue lengths and system load
- Number of operations performed by each worker

##  Simulation Features

- Custom Linear Congruential Generator (LCG) for pseudo-random number generation
- Normal distributions for simulating task durations
- Priority-based event scheduling (arrival, rental, return, maintenance, cleaning)
- Modular and extensible event handlers
- Detailed performance metrics output as CSV (`results_partX.csv`)

##  Files

- `part1.py`: Core simulation engine for scenario 1
- `part2.py`: Core simulation engine for scenario 2
- `part3.py`: Final scenario with an additional floating worker
- `results_part1.csv`, `results_part2.csv`, `results_part3.csv`: Output metrics from multiple simulation runs

