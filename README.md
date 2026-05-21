# Comparative Analysis of Multi-Agent Q-Learning for Evolutionary Games

A reinforcement learning framework for studying evolutionary behavior in multi-agent environments using Q-Learning and the Iterated Prisoner’s Dilemma.

This project is based on the original Multiagent Q-Learning for Evolutionary Games project, extending the framework with constrained agents, strategic rule sets, comparative analysis, and behavioral metrics.

---

## Overview

The framework simulates populations of agents competing and learning in an evolutionary environment.

Agents may be:

* Fully free-learning agents
* Strategically constrained agents
* Fixed mentor strategies

The goal is to analyze how strategic restrictions affect:

* cooperation emergence
* learning convergence
* adaptation dynamics
* competitive performance
* behavioral diversity

---

## Features

* Multi-agent Q-Learning
* Constrained reinforcement learning agents
* Rule-based strategic filtering
* Evolutionary training system
* Mentor strategy populations
* Iterated Prisoner’s Dilemma environment
* Q-table persistence
* Cooperation and defection metrics
* Learning stability visualization
* Comparative analysis between free and constrained agents

---

## Rule Sets

The framework includes multiple strategic restriction systems:

* Conservative Rules
* Aggressive Rules
* Adaptive Rules
* Stochastic Rules

These rules constrain the action space available to specific agents during learning.

---

## Mentor Strategies

Implemented mentor agents include:

* Tit For Tat
* Generous Tit For Tat
* Win-Stay Lose-Shift
* Holds a Grudge
* Fool Me Once
* Omega TFT
* Gradual TFT

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install matplotlib numpy
```

---

## Running

Execute the main file:

```bash
python main.py
```

The framework will:

1. Initialize agent populations
2. Train agents through repeated interactions
3. Evaluate learned behaviors
4. Save learned Q-Tables
5. Generate convergence metrics and graphs

---

## Output

The system generates:

* Q-table export files
* Cooperation statistics
* Win/loss metrics
* Learning stability plots

Example output:

```text
Q_table_20260520_1430.txt
```

---

## Research Focus

This framework is designed for experimentation in:

* Multi-Agent Reinforcement Learning
* Evolutionary Game Theory
* Strategic Constraint Systems
* Emergent Cooperation
* Adaptive Behavior
* Artificial Social Dynamics

---

## Credits

Original project inspiration:

https://github.com/YuzukiWang/multiagent_q_learning_for_evolutionary_games

Extended and modified with:

* constrained agents
* strategic rule systems
* evolutionary comparisons
* additional metrics and analysis

---

## License

This project is intended for academic and research purposes.
