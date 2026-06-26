# Comparative Analysis of Multi-Agent Q-Learning for Evolutionary Games

A reinforcement learning framework for studying evolutionary behavior in multi-agent environments using Q-Learning and the Iterated Prisoner's Dilemma, with comprehensive batch experiment capabilities for scientific research.

## Overview

This framework simulates populations of agents competing and learning in an evolutionary environment, implementing the Multiagent Q-Learning for Evolutionary Games approach with significant extensions for systematic scientific experimentation.

### Key Components

- **Interactive Experiment Configuration**: Users can configure all experimental parameters at runtime
- **Three-Experiment Workflow**: Training, Evaluation, and Evolution experiments
- **Comprehensive Statistical Analysis**: Automated statistical testing and confidence intervals
- **Publication-Ready Visualizations**: Automated figure generation in violet-rose gradient
- **Final Comparison Tables**: Per-strategy and cross-experiment comparisons

## Features

### Core Capabilities
- Multi-agent Q-Learning implementation (AgentQ)
- MTBR (Multiagent Temporal Difference with Bayesian Regularization) strategy
- Mentor strategy populations (8 classic strategies)
- Iterated Prisoner's Dilemma environment with customizable payoff matrix
- Q-table persistence and export
- Cooperation and defection metrics
- Learning stability visualization

### Strategic Rule Sets
- Tit For Tat (TFT)
- Generous Tit For Tat (GTFT)
- Win-Stay Lose-Shift (WSLS)
- Grudge
- Fool Me Once
- Omega TFT
- Gradual TFT
- All Defect (AllD)

### Experimental Pipeline

1. **Training Experiments**: MTBR agents learn through repeated interactions
   - Configurable number of runs and iterations
   - Tracks payoff convergence and cooperation rates
   - Q-table size metrics

2. **Mentor Evaluation**: Head-to-head comparisons
   - MTBR agents vs 8 mentor strategies
   - Win rate, payoff, and cooperation statistics
   - Per-strategy performance analysis

3. **Evolutionary Simulations**: Population dynamics
   - Compare evolution with and without MTBR
   - Track strategy frequencies over generations
   - Analyze MTBR dominance and payoff advantages

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib
```

## Quick Start

Run the main experiment runner:

```bash
python experiment_runner.py
```

Follow the interactive prompts to configure:

```
Number of training runs (default: 30): 30
Training iterations per run (default: 50000): 50000
Number of evaluation runs (default: 50): 50
Episodes per evaluation (default: 1000): 1000
Number of evolution runs (default: 50): 50
Generations per evolution (default: 200): 200
Generate figures? (y/n, default: y): y
```

## Output Structure

The system generates a comprehensive output directory:

```
article_results/
├── training_results.json           # Training experiment data
├── evaluation_results.json          # Mentor evaluation data
├── evolution_results.json           # Evolution simulation data
├── statistical_analysis.json        # Statistical tests and metrics
├── comparison_table.csv             # Cross-experiment comparison
├── strategy_comparison.csv          # Per-strategy performance
└── figures/
    ├── fig1_training_results.png    # Training convergence plots
    ├── fig2_mentor_comparison.png   # Mentor strategy comparison
    └── fig3_evolutionary_dynamics.png # Evolution dynamics plots
```

### Output Example

```text
==============================================================
 FINAL COMPARISON
==============================================================

--------------------------------------------------------------
 COMPARISON TABLE
--------------------------------------------------------------
Experiment   Metric               Value           95% CI
Training     Avg Payoff           2.834 ± 0.287   [2.744, 2.924]
Training     Cooperation Rate     0.712 ± 0.083   [0.684, 0.740]
Evaluation   Avg Payoff           2.756 ± 0.312   [2.668, 2.844]
Evaluation   Win Rate             0.623 ± 0.047   [0.611, 0.635]
Evolution    MTBR Final Frequency 0.723 ± 0.114   [0.692, 0.754]
Evolution    Dominance Rate       84.0%           [73.0%, 95.0%]
```

## Scientific Contributions

This framework enables rigorous experimentation in:

- **Multi-Agent Reinforcement Learning**: Q-learning dynamics in competitive environments
- **Evolutionary Game Theory**: Strategy evolution and stability analysis
- **Behavioral Economics**: Cooperation emergence and strategic sophistication
- **Artificial Intelligence**: Learning convergence and adaptation
- **Complex Systems**: Population dynamics and emergent behavior

## Statistical Analysis

The framework automatically performs:

- **Descriptive Statistics**: Means, standard deviations, standard errors
- **Confidence Intervals**: 95% CI using t-distribution
- **Hypothesis Testing**: T-tests for comparing conditions
- **Effect Size**: Quantification of MTBR vs baseline differences

## Figure Generation

Publication-quality figures are automatically generated:

### Figure 1: Training Results
- Payoff distribution with confidence intervals
- Cooperation rate distribution with confidence intervals

### Figure 2: Mentor Comparison
- MTBR vs mentor payoffs across all strategies
- Win rates and cooperation rates
- Strategy-specific performance metrics

### Figure 3: Evolutionary Dynamics
- Strategy frequency evolution (with/without MTBR)
- Payoff evolution over generations
- MTBR final frequency distribution

## Customization

### Payoff Matrix

The default Iterated Prisoner's Dilemma uses:
- R = 3 (Reward for mutual cooperation)
- S = 0 (Sucker's payoff)
- T = 5 (Temptation to defect)
- P = 1 (Punishment for mutual defection)

Modify in the `InteractiveExperimentRunner` class initialization.

### Color Palette

The violet-rose gradient can be customized in the `VIOLET_ROSE_COLORS` list.

### Experiment Parameters

All experimental parameters are configurable through the interactive interface, including:
- Sample sizes (runs/repetitions)
- Iteration/generation counts
- Figure generation toggle

## Research Applications

This framework is designed for:

- **Comparative Studies**: Free vs constrained learning agents
- **Strategy Analysis**: Evaluating different strategic rule sets
- **Evolutionary Dynamics**: Studying strategy dominance and coexistence
- **Learning Behavior**: Understanding convergence and adaptation patterns
- **Game Theory Experiments**: Testing theoretical predictions in IPD

## Academic Use

When using this framework for research, please cite:

```
Original project: https://github.com/YuzukiWang/multiagent_q_learning_for_evolutionary_games
Extended framework: (cite your paper when published)
```

## Credits

Original project inspiration:
- https://github.com/YuzukiWang/multiagent_q_learning_for_evolutionary_games

Extensions and modifications:
- Interactive experiment runner
- Comprehensive statistical analysis
- Automated figure generation
- Final comparison tables
- Per-strategy evaluation
- Evolutionary dynamics with MTBR

## Requirements

- Python 3.7+
- NumPy ≥ 1.19.0
- SciPy ≥ 1.5.0
- Pandas ≥ 1.0.0
- Matplotlib ≥ 3.2.0

## License

This project is intended for academic and research purposes. Please attribute appropriately when using in research.
