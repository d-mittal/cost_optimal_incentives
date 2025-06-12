# Targeted incentives for social tipping in heterogeneous networked populations

**Authors:** Dhruv Mittal, Fátima González-Novo López, Sara Constantino, Shaul Shalvi, Xiaojie Chen, and Vítor V. Vasconcelos

This repository contains Python code for simulating and analyzing cost-optimized institutional incentives in heterogeneous networked populations. The code implements a game-theoretical framework in which individuals update their choices based on both their intrinsic preferences and the influence of their neighbors in a network. Through this model, various targeting heuristics are evaluated to determine which strategies minimize intervention costs while ensuring rapid and equitable adoption of new behaviors. Representative Survey data from the US regarding support for climate change is also included, along with willingness-to-pay(WTP) analysis, synthetic populations, and associated networks from the survey sample, which help ground model assumptions.


---


## Abstract
Many societal challenges, such as climate change or disease outbreaks, require coordinated behavioral changes. For many behaviors, the tendency of individuals to adhere to social norms can reinforce the status quo. However, these same social processes can also result in rapid, self-reinforcing change. Interventions may be strategically targeted to initiate endogenous social change processes, often referred to as social tipping. While recent research has considered how the size and targeting of such interventions impact their effectiveness at bringing about change, they tend to overlook constraints faced by policymakers, including the cost, speed, and distributional consequences of interventions. To address this complexity, we introduce a game-theoretic framework that includes heterogeneous agents and networks of local influence. We implement various targeting heuristics based on information about individual preferences and commonly used local network properties to identify individuals to incentivize. Analytical and simulation results suggest that there is a trade-off between preventing backsliding among targeted individuals and promoting change among non-targeted individuals. Thus, where the change is initiated in the population and the direction in which it propagates is essential to the effectiveness of interventions. We identify cost-optimal strategies under different scenarios, such as varying levels of resistance to change, preference heterogeneity, and homophily. These results provide insights that can be experimentally tested and help policymakers to better direct incentives.

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
   A Jupyter Notebook with the analytical model based on Markov chain analysis and a mean field approach.

7. **`WTP_analysis.ipynb`**  
   A Jupyter Notebook that shows the WTP analysis of survey data and models the effect of demographic factors.

7. **synthetic_population.ipynb`**  
   A Jupyter Notebook that generates synthetic populations by sampling from the survey sample and creates associated homophilic networks using the SDA algorithm.
    



## Usage

To use the code:

1. Ensure you have Python installed on your system.
2. These scripts are based on Numpy, NetworkX and Scipy. 



## License

This GUI is distributed under the Creative Commons Attribution 4.0 International license. The DOI of the project is XXXXXX


## Acknowledgments

This code was developed by Dhruv Mittal.

---
