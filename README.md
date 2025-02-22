# Targeting heuristics for cost-optimized institutional incentives in heterogeneous networked populations

**Authors:** Dhruv Mittal, Fátima González-Novo López, Sara Constantino, Shaul Shalvi, Xiaojie Chen, and Vítor V. Vasconcelos

This repository contains Python code for simulating and analyzing cost-optimized institutional incentives in heterogeneous networked populations. The code implements a game-theoretical framework in which individuals update their choices based on both their intrinsic preferences and the influence of their neighbors in a network. Through this model, various targeting heuristics are evaluated to determine which strategies minimize intervention costs while ensuring rapid and equitable adoption of new behaviors.


---


## Abstract
In a world facing urgent challenges—from climate change to pandemics—coordinated interventions are essential to catalyze large-scale behavioral shifts. Policy-makers often resort to incentive schemes to overcome resistance to change; however, the success of such interventions crucially depends on whom to target and how incentives are distributed. This project presents a computational framework that integrates individual heterogeneity, network structure, and dynamic decision-making. Our simulations reveal that optimal targeting strategies vary with preference distributions, network topology, and the dynamics of social influence, offering actionable insights for cost-effective and equitable policy design.

## Code Overview

The repository includes the following Python scripts:

1. **`generate_network.py`**  
   Contains functions to generate network graphs using various models (e.g., Erdos-Rényi, Barabasi-Albert, Watts-Strogatz) and compute network metrics such as clustering and centrality which serve as criteria for targeted incentives.

2. **`generate_preference_distribution.py`**  
   Contains a function to generate a list of preferences based on a symmetric beta function

2. **`segregated_networks.py`**  
   Contains functions to generate segregated networks based on homophilous preferential attachment and the stochastic block model.

4. **`simulation.py`**  
   Implements the simulation of behavior adoption dynamics on a network in response to interventions. It outputs the the cost of incentives, the time taken to reach the required adoption level, the final adoption and the Gini coefficients of the incentive distribution

5. **`Manuscript_figures.ipynb`**  
   A Jupyter Notebook that generates the figures presented in the main manuscript. The notebook combines simulation outputs and visualization routines to illustrate the cost-effectiveness, timing, and equity of various targeting strategies.

6. **`Analytical_model.ipynb`**  
   A Jupyter Notebook that the analytical model based on Markov chain analysis and a mean field approach.


## Usage

To use the code:

1. Ensure you have Python installed on your system.
2. These scripts are based on Numpy, NetworkX and Scipy. 



## License

This GUI is distributed under the Creative Commons Attribution 4.0 International license. The DOI of the project is XXXXXX


## Acknowledgments

This code was developed by Dhruv Mittal.

---
