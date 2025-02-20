# Optimal Incentives: Targeting Heuristics for Cost-Optimized Institutional Incentives in Heterogeneous Networked Populations

This repository contains the code and data associated with the manuscript:

**"Targeting heuristics for cost-optimized institutional incentives in heterogeneous networked populations"**

**Authors:** Dhruv Mittal, Fátima González-Novo López, Sara Constantino, Shaul Shalvi, Xiaojie Chen, and Víctor V. Vasconcelos

---

## Overview

This project presents a game-theoretic and computational framework for optimizing the cost of incentivizing early adopters in networked populations. It explores various targeting heuristics based on individual preferences and local network properties, including network centrality and clustering metrics. The goal is to minimize intervention costs while ensuring a timely and equitable transition towards a new behavioral state.

Key features include:
- Simulation of collective adoption dynamics using network models (e.g., Erdos-Rényi, Barabási-Albert, Watts–Strogatz, and homophilous networks).
- Implementation of targeting strategies based on preference distributions and network structure.
- Analysis of cost-effectiveness, time to achieve a 90% adoption threshold, and incentive equity (as measured by the Gini coefficient).

---

## Repository Structure

- **Optimal_incentives.pdf:**  
  The manuscript describing the model, results, and implications of the study.

- **Manuscript_figures.ipynb:**  
  A Jupyter Notebook that generates the figures presented in the manuscript. This notebook contains:
  - Code cells for generating simulation plots.
  - Markdown cells with explanations and context derived from the manuscript.
  - Plots that illustrate key results such as cost-effectiveness curves, adoption time dynamics, and incentive distribution equality.

- **Simulation Scripts:**  
  Additional Python scripts implementing the simulation model, including functions for network generation, computing metrics (e.g., moving averages, Gini coefficients), and simulating the dynamics of interventions.

---
