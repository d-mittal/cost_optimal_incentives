**"Targeting heuristics for cost-optimized institutional incentives in heterogeneous networked populations"**

**Authors:** Dhruv Mittal, Fátima González-Novo López, Sara Constantino, Shaul Shalvi, Xiaojie Chen, and Víctor V. Vasconcelos

This repository contains Python code for simulating and analyzing cost-optimized institutional incentives in heterogeneous networked populations. The code implements a game-theoretical framework in which individuals update their choices based on both their intrinsic preferences and the influence of their neighbors in a network. Through this model, various targeting heuristics are evaluated to determine which strategies minimize intervention costs while ensuring rapid and equitable adoption of new behaviors.


---


## Abstract
In a world facing urgent challenges—from climate change to pandemics—coordinated interventions are essential to catalyze large-scale behavioral shifts. Policy-makers often resort to incentive schemes to overcome resistance to change; however, the success of such interventions crucially depends on whom to target and how incentives are distributed. This project presents a computational framework that integrates individual heterogeneity, network structure, and dynamic decision-making. Our simulations reveal that optimal targeting strategies vary with preference distributions, network topology, and the dynamics of social influence, offering actionable insights for cost-effective and equitable policy design.

## Code Overview

The repository includes the following Python scripts:

1. `one_run_network.py`: This script simulates population dynamics with heterogenous preferences on networks. It implements a model where individuals update their choices based on their preferences and the choices of their neighbors on a fixed graph.

2. `one_run_well_mixed.py`: This script simulates population dynamics with heterogenous preferences connected probabilistically, mimicking a well-mixed limit. It models a scenario where individuals' choices are influenced by their preferences, the choices of randomly sampled neighbors.

3. `one_run_net_dynamic.py`: This script simulates population dynamics with evolving preferences on networks. It implements a model where individuals update their choices based on their evolving preferences (which are shared by all agents) and the choices of their neighbors on a fixed graph.

4. `adjacency_matrix_generator.py`: This function returns the adjacency matrix of the required graph (Barabasi-Albert or Erdos Renyi)

5. `conformity_placement_on_network.py`: This function assigns conformity to the nodes of a given network to achieve the required correlation between anti-conformity(non-conformity) and node degree.

6. `Manuscript_figures.ipynb`: This notebook can be used to generate figures from the main text. This notebook uses the above scripts to initialize and run simulations.

## Usage

To use the code:

1. Ensure you have Python installed on your system.
2. These scripts are completely based on Numpy. 


## Output of simulation scripts

`one_run_network.py` and `one_run_well_mixed.py` have the following output:

- Equilibrium fraction of choice A in the population
- Equilibrium alignment of choice and preference for the entire population.
- Volatility in choices at the end of the simulation

  
`one_run_net_dynamic.py` has the following output: 

- Time series of the fraction of individuals choosing option A in the conforming and the anti-conforming subpopulations, as well as the entire population.
- Average alignment of choice and preference for the conforming and the anti-conforming subpopulations.

## License

This GUI is distributed under the Creative Commons Attribution 4.0 International license. The DOI of the project is https://doi.org/10.5281/zenodo.12707357


## Acknowledgments

This code was developed by Dhruv Mittal.

---
