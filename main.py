# ============================================================
# Multiagent Q-Learning for Evolutionary Games
#
# Framework Integrado com Sistema de Comparação de Resultados
# ============================================================

# ============================================================
# CONFIGURATION SECTION
# ============================================================

POPULATION_SIZE = 49
MENTOR_INSTANCES = 7
MENTOR_TYPES = 7
EPISODE_LENGTH = 20
reward_weighting_factor = 0.8
max_decisions = 450000
TESTING_EPISODES = 1000
STATISTICAL_EPISODES = 500
alpha = 0.2
gamma = 0.5
MEMORY_SIZE = 2
HISTORY_FILE = "experiment_history.json"

# ============================================================
# PRISONER'S DILEMMA REWARD MATRIX
# ============================================================

R = 2
S = 0
T = 3
P = 0.1

reward_matrix = [
    [[R, R], [S, T]],
    [[T, S], [P, P]]
]

# ============================================================
# IMPORTS
# ============================================================

import sys
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# COLOR PALETTE - VIOLET TO ROSE GRADIENT
# ============================================================

VIOLET_ROSE_COLORS = [
    '#4A004A',  # Deep violet
    '#6A006A',  # Dark violet
    '#8B00FF',  # Violet
    '#9B30FF',  # Purple
    '#B266FF',  # Light purple
    '#D4A0FF',  # Lavender
    '#FF6B9D',  # Rose
    '#FF1493',  # Deep rose
    '#FF69B4',  # Hot pink
    '#FFB6C1'   # Light pink
]

def get_color_gradient(n, cmap_name='cool'):
    """Retorna n cores em gradiente do mapa de cores especificado"""
    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(i / max(n-1, 1)) for i in range(n)]

# ============================================================
# HISTORY MANAGEMENT SYSTEM WITH CLEAR FUNCTION
# ============================================================

class ExperimentHistory:
    def __init__(self, history_file=HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_experiment(self, results, stats, config):
        experiment = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': config,
            'results': results,
            'stats': stats if isinstance(stats, dict) else stats.to_dict() if hasattr(stats, 'to_dict') else {},
            'free_score_mean': np.mean(results.get('free_scores', [0])),
            'restricted_score_mean': np.mean(results.get('restricted_scores', [0])),
            'mentor_score_mean': np.mean(results.get('mentor_scores', [0])),
            'free_score_std': np.std(results.get('free_scores', [0])),
            'restricted_score_std': np.std(results.get('restricted_scores', [0])),
            'mentor_score_std': np.std(results.get('mentor_scores', [0]))
        }
        self.history.append(experiment)
        self.save_history()
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def clear_history(self):
        """Clear all experiment history"""
        self.history = []
        self.save_history()
        print("\nExperiment history cleared successfully! ( ﾉ ﾟｰﾟ)ﾉ")
    
    def get_latest_experiments(self, n=5):
        return self.history[-n:] if self.history else []
    
    def get_all_experiments(self):
        return self.history
    
    def compare_experiments(self, n=None):
        experiments = self.get_latest_experiments(n) if n else self.get_all_experiments()
        if not experiments:
            print("\nNo previous experiments found. ¯\_(ツ)_/¯")
            return None
        
        print("\n" + "="*70)
        print(" COMPARISON OF PREVIOUS EXPERIMENTS ".center(70, "="))
        print("="*70)
        
        df_data = []
        for i, exp in enumerate(experiments):
            df_data.append({
                'Experiment': i + 1,
                'Timestamp': exp['timestamp'],
                'Free Q-Agent': f"{exp.get('free_score_mean', 0):.2f} ± {exp.get('free_score_std', 0):.2f}",
                'Restricted Q-Agent': f"{exp.get('restricted_score_mean', 0):.2f} ± {exp.get('restricted_score_std', 0):.2f}",
                'Evolutionary': f"{exp.get('mentor_score_mean', 0):.2f} ± {exp.get('mentor_score_std', 0):.2f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Plot comparison
        self.plot_comparison(experiments)
        
        return df
    
    def plot_comparison(self, experiments):
        if len(experiments) < 2:
            print("\nNeed at least 2 experiments for comparison plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Mean scores comparison
        n_exps = len(experiments)
        x = np.arange(n_exps)
        width = 0.25
        
        free_means = [exp.get('free_score_mean', 0) for exp in experiments]
        restricted_means = [exp.get('restricted_score_mean', 0) for exp in experiments]
        mentor_means = [exp.get('mentor_score_mean', 0) for exp in experiments]
        
        free_stds = [exp.get('free_score_std', 0) for exp in experiments]
        restricted_stds = [exp.get('restricted_score_std', 0) for exp in experiments]
        mentor_stds = [exp.get('mentor_score_std', 0) for exp in experiments]
        
        bars1 = axes[0].bar(x - width, free_means, width, label='Free Q-Agent', 
                           yerr=free_stds, capsize=3, color=VIOLET_ROSE_COLORS[2], alpha=0.7)
        bars2 = axes[0].bar(x, restricted_means, width, label='Restricted Q-Agent', 
                           yerr=restricted_stds, capsize=3, color=VIOLET_ROSE_COLORS[5], alpha=0.7)
        bars3 = axes[0].bar(x + width, mentor_means, width, label='Evolutionary', 
                           yerr=mentor_stds, capsize=3, color=VIOLET_ROSE_COLORS[7], alpha=0.7)
        
        axes[0].set_xlabel('Experiment Number')
        axes[0].set_ylabel('Average Score')
        axes[0].set_title('Score Comparison Across Experiments')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'Exp {i+1}' for i in range(n_exps)])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Performance trends
        exp_labels = [f"Exp {i+1}\n{exp['timestamp'][:10]}" for i, exp in enumerate(experiments)]
        
        axes[1].plot(exp_labels, free_means, 'o-', label='Free Q-Agent', 
                    color=VIOLET_ROSE_COLORS[2], linewidth=2, markersize=8)
        axes[1].plot(exp_labels, restricted_means, 's-', label='Restricted Q-Agent', 
                    color=VIOLET_ROSE_COLORS[5], linewidth=2, markersize=8)
        axes[1].plot(exp_labels, mentor_means, '^-', label='Evolutionary', 
                    color=VIOLET_ROSE_COLORS[7], linewidth=2, markersize=8)
        
        axes[1].set_xlabel('Experiment')
        axes[1].set_ylabel('Average Score')
        axes[1].set_title('Performance Trends')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: All scores distribution
        self.plot_distribution_comparison(experiments)
    
    def plot_distribution_comparison(self, experiments):
        if len(experiments) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        data = []
        labels = []
        
        for i, exp in enumerate(experiments):
            free_scores = exp.get('results', {}).get('free_scores', [0])
            restricted_scores = exp.get('results', {}).get('restricted_scores', [0])
            mentor_scores = exp.get('results', {}).get('mentor_scores', [0])
            
            data.extend([free_scores, restricted_scores, mentor_scores])
            labels.extend([f'Exp{i+1} Free', f'Exp{i+1} Restricted', f'Exp{i+1} Mentor'])
        
        # Create boxplot with violet-rose colors
        bp = ax.boxplot(data, patch_artist=True)
        
        # Color the boxes with violet-rose gradient
        colors = get_color_gradient(len(data), 'cool')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Score')
        ax.set_title('Score Distribution Comparison')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()

# ============================================================
# RULE SETS
# ============================================================

def conservative_rules(state, actions):
    if len(state) > 0:
        if state[-1][0] == 0:
            return [0]
    return actions

def aggressive_rules(state, actions):
    return [1]

def adaptive_rules(state, actions):
    if len(state) < 5:
        return [0]
    return actions

def stochastic_rules(state, actions):
    if random.random() < 0.8:
        return [0]
    return actions

def generous_rules(state, actions):
    if len(state) > 0 and state[-1][0] == 1 and random.random() < 0.3:
        return [0]
    return actions

# ============================================================
# HUMAN AGENT
# ============================================================

class AgentHuman:
    def pick_action(self, state):
        action = -1
        print(
            "State: "
            + str(state)
            + " ("
            + str(len(state))
            + "/"
            + str(EPISODE_LENGTH)
            + ")"
        )
        while action not in [0, 1]:
            try:
                action = int(input("Choose Cooperate/Defect (0/1): "))
            except ValueError:
                print("Please input a number.")
        return action

    def reward_action(self, state, action, reward):
        pass

# ============================================================
# Q-LEARNING AGENT
# ============================================================

class AgentQ:
    def __init__(self, memory=MEMORY_SIZE):
        self.wins = 0
        self.losses = 0
        self.Q = {}
        self.memory = memory
        self.epsilon_counter = 1
        self.delta = []
        self.avgdelta_list = []
        self.total_reward = 0
        self.steps = 0

    def get_q(self, state):
        state_key = str(state[-self.memory:])
        if state_key not in self.Q:
            self.Q[state_key] = [0, 0]
        return self.Q[state_key][0], self.Q[state_key][1]

    def set_q(self, state, quality1, quality2):
        state_key = str(state[-self.memory:])
        if state_key not in self.Q:
            self.Q[state_key] = [0, 0]
        self.Q[state_key][0] = quality1
        self.Q[state_key][1] = quality2

    def normalize_q(self, state):
        quality1, quality2 = self.get_q(state)
        normalization = min(quality1, quality2)
        self.set_q(state, 
                  (quality1 - normalization) * 0.95,
                  (quality2 - normalization) * 0.95)

    def epsilon_greedy(self, state):
        quality1, quality2 = self.get_q(state)
        epsilon = 1.0 / (1.0 + self.epsilon_counter / 1000.0)
        
        if random.random() < epsilon:
            return random.randint(0, 1)
        
        if quality1 > quality2:
            return 0
        elif quality2 > quality1:
            return 1
        else:
            return random.randint(0, 1)

    def pick_action(self, state):
        self.epsilon_counter += 1
        
        if self.epsilon_counter % 1000 == 1 and len(self.delta) > 0:
            self.avgdelta_list.append(self.delta_average())
        
        return self.epsilon_greedy(state)

    def reward_action(self, state, action, reward):
        if len(state) < self.memory + 1:
            return
        
        oldstate = state[-self.memory - 1:-1]
        newstate = state[-self.memory:]
        
        oldq1, oldq2 = self.get_q(oldstate)
        oldq = oldq1 if action == 0 else oldq2
        
        newq1, newq2 = self.get_q(newstate)
        maxqnew = max(newq1, newq2)
        
        new_q = oldq + alpha * (reward + gamma * maxqnew - oldq)
        
        if action == 0:
            self.set_q(oldstate, new_q, oldq2)
        else:
            self.set_q(oldstate, oldq1, new_q)
        
        self.delta.append(reward + gamma * maxqnew - oldq)
        if len(self.delta) > 1000:
            self.delta.pop(0)
        
        self.total_reward += reward
        self.steps += 1

    def delta_average(self):
        return sum(self.delta) / len(self.delta) if self.delta else 0

    def get_avgdelta_list(self):
        return self.avgdelta_list

    def analyse(self):
        times_cooperated = sum(1 for state in self.Q 
                             if self.epsilon_greedy(eval(state)) == 0)
        percent_cooperated = times_cooperated / len(self.Q) if self.Q else 0
        percent_won = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
        return self.wins, percent_won, percent_cooperated

    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def reset_analysis(self):
        self.wins = 0
        self.losses = 0

# ============================================================
# CONSTRAINED Q AGENT
# ============================================================

class ConstrainedAgentQ(AgentQ):
    def __init__(self, memory=MEMORY_SIZE, rule_set=None):
        super().__init__(memory)
        self.rule_set = rule_set

    def filter_actions(self, state):
        actions = [0, 1]
        if self.rule_set:
            return self.rule_set(state, actions)
        return actions

    def epsilon_greedy_restricted(self, state):
        quality1, quality2 = self.get_q(state)
        epsilon = 1.0 / (1.0 + self.epsilon_counter / 1000.0)
        allowed_actions = self.filter_actions(state)
        
        if not allowed_actions:
            allowed_actions = [0, 1]
        
        if random.random() < epsilon:
            return random.choice(allowed_actions)
        
        if 0 in allowed_actions and 1 in allowed_actions:
            if quality1 > quality2:
                return 0
            elif quality2 > quality1:
                return 1
            else:
                return random.choice(allowed_actions)
        elif 0 in allowed_actions:
            return 0
        else:
            return 1

    def pick_action(self, state):
        self.epsilon_counter += 1
        
        if self.epsilon_counter % 1000 == 1 and len(self.delta) > 0:
            self.avgdelta_list.append(self.delta_average())
        
        return self.epsilon_greedy_restricted(state)

# ============================================================
# DEFINED AGENTS (MENTORS)
# ============================================================

class AgentDefined:
    def __init__(self, strategy):
        self.wins = 0
        self.losses = 0
        self.strategy = strategy
        self.deadlock_threshold = 3
        self.randomness_threshold = 8
        self.randomness_counter = 0
        self.deadlock_counter = 0
        self.calm_count = 0
        self.punish_count = 0
        self.strategy_names = {
            0: "TFT",
            1: "GTFT0.3",
            2: "WSLS",
            3: "Holds a grudge",
            4: "Fool me once",
            5: "Omega TFT",
            6: "Gradual TFT"
        }

    def strategy_print(self):
        return self.strategy_names.get(self.strategy, "Unknown")

    def pick_action(self, state):
        if len(state) == 0:
            return random.randint(0, 1)
        
        if self.strategy == 0:  # TIT FOR TAT
            return state[-1][0]
        
        elif self.strategy == 1:  # GENEROUS TIT FOR TAT
            if state[-1][0] == 0:
                return 0
            return 0 if random.random() < 1/3 else 1
        
        elif self.strategy == 2:  # WIN STAY LOSE SHIFT
            if state[-1][0] == 0:
                return state[-1][1]
            return 1 - state[-1][1]
        
        elif self.strategy == 3:  # HOLDS A GRUDGE
            return 1 if any(action[0] == 1 for action in state) else 0
        
        elif self.strategy == 4:  # FOOL ME ONCE
            defect_count = sum(1 for action in state if action[0] == 1)
            return 1 if defect_count >= 2 else 0
        
        elif self.strategy == 5:  # OMEGA TFT
            if len(state) == 1:
                return state[-1][0]
            
            if self.deadlock_counter >= self.deadlock_threshold:
                if self.deadlock_counter == self.deadlock_threshold:
                    self.deadlock_counter += 1
                    return 0
                self.deadlock_counter = 0
                return self.pick_action(state[:-1])
            
            if len(state) >= 2 and state[-2][0] == state[-1][0] == 0:
                self.randomness_counter = max(0, self.randomness_counter - 1)
            
            if state[-2][0] != state[-1][0] or state[-1][0] != state[-1][1]:
                self.randomness_counter += 1
            
            if self.randomness_counter >= self.randomness_threshold:
                return 1
            
            move = state[-1][0]
            if state[-2][0] != state[-1][0]:
                self.deadlock_counter += 1
            else:
                self.deadlock_counter = 0
            
            return move
        
        elif self.strategy == 6:  # GRADUAL TFT
            if self.punish_count > 0:
                self.punish_count -= 1
                return 1
            
            if self.calm_count > 0:
                self.calm_count -= 1
                return 0
            
            if state[-1][0] == 1:
                defect_count = sum(1 for action in state if action[0] == 1)
                self.punish_count = defect_count - 1
                self.calm_count = 2
                return 1
            
            return 0
        
        return random.randint(0, 1)

    def reward_action(self, state, action, reward):
        pass

    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def analyse(self):
        percent_won = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
        return self.wins, percent_won

# ============================================================
# POPULATION INITIALIZATION
# ============================================================

def initialize_population(pop_size=POPULATION_SIZE):
    population = []
    
    for i in range(pop_size // 2):
        population.append(AgentQ(MEMORY_SIZE))
    
    rule_sets = [conservative_rules, aggressive_rules, 
                adaptive_rules, stochastic_rules, generous_rules]
    
    for i in range(pop_size // 2):
        chosen_rule = random.choice(rule_sets)
        population.append(ConstrainedAgentQ(MEMORY_SIZE, chosen_rule))
    
    return population

def initialize_mentors():
    mentors = []
    for i in range(MENTOR_TYPES):
        for j in range(MENTOR_INSTANCES):
            mentors.append(AgentDefined(i))
    return mentors

# ============================================================
# TRAINING PHASE
# ============================================================

def train_agents(population, mentors, verbose=True):
    while any(agent.epsilon_counter < max_decisions for agent in population):
        average_epsilon_counter = sum(agent.epsilon_counter for agent in population) / POPULATION_SIZE
        
        if verbose:
            progress = 100 * average_epsilon_counter / max_decisions
            sys.stdout.write(f'\rTraining [{"#" * int(progress / 5):19}] {int(min(100, progress + 5))}%')
            sys.stdout.flush()
        
        player1 = random.choice(population)
        if player1.epsilon_counter >= max_decisions:
            continue
        
        if isinstance(player1, ConstrainedAgentQ):
            agent_type_1 = "Restricted Q-Agent"
        else:
            agent_type_1 = "Free Q-Agent"
        
        player2 = random.choice(population + mentors)
        
        if isinstance(player2, AgentDefined):
            agent_type_2 = "Evolutionary Strategy"
        elif isinstance(player2, ConstrainedAgentQ):
            agent_type_2 = "Restricted Q-Agent"
        else:
            agent_type_2 = "Free Q-Agent"
        
        state1, state2 = [], []
        for i in range(EPISODE_LENGTH):
            action1 = player1.pick_action(state1)
            action2 = player2.pick_action(state2)
            state1.append([action2, action1])
            state2.append([action1, action2])
        
        rewards1, rewards2 = [], []
        total_reward1 = total_reward2 = 0
        
        for i in range(EPISODE_LENGTH):
            action1, action2 = state2[i][0], state1[i][0]
            r1, r2 = reward_matrix[action1][action2]
            rewards1.append(r1)
            rewards2.append(r2)
            total_reward1 += r1
            total_reward2 += r2
        
        if total_reward1 >= total_reward2:
            reward_chunk = total_reward1 / EPISODE_LENGTH * (1 - reward_weighting_factor)
            for i in range(EPISODE_LENGTH - 1):
                player1.reward_action(state1[:i+1], state2[i][0], 
                                    reward_chunk + rewards1[i] * reward_weighting_factor)
                player2.reward_action(state2[:i+1], state1[i][0], 
                                    rewards2[i] * reward_weighting_factor)
        else:
            reward_chunk = total_reward2 / EPISODE_LENGTH * (1 - reward_weighting_factor)
            for i in range(EPISODE_LENGTH - 1):
                player2.reward_action(state2[:i+1], state1[i][0], 
                                    reward_chunk + rewards2[i] * reward_weighting_factor)
                player1.reward_action(state1[:i+1], state2[i][0], 
                                    rewards1[i] * reward_weighting_factor)
    
    if verbose:
        print("\nTraining complete!")
    
    return population

# ============================================================
# TESTING PHASE
# ============================================================

def test_agents(population, mentors, episodes=TESTING_EPISODES):
    wins_agent = 0
    wins_mentor = 0
    ties = 0
    
    Nc_agent = Nc_mentor = 0
    Nd_agent = Nd_mentor = 0
    
    free_scores = []
    restricted_scores = []
    mentor_scores = []
    
    for episode in range(episodes):
        state1, state2 = [], []
        player1 = random.choice(population)
        player2 = random.choice(mentors)
        
        if isinstance(player1, ConstrainedAgentQ):
            agent_type_1 = "Restricted Q-Agent"
        else:
            agent_type_1 = "Free Q-Agent"
        
        if isinstance(player2, AgentDefined):
            agent_type_2 = "Evolutionary Strategy"
        elif isinstance(player2, ConstrainedAgentQ):
            agent_type_2 = "Restricted Q-Agent"
        else:
            agent_type_2 = "Free Q-Agent"
        
        for i in range(EPISODE_LENGTH):
            action1 = player1.pick_action(state1)
            action2 = player2.pick_action(state2)
            state1.append([action2, action1])
            state2.append([action1, action2])
        
        total_reward1 = total_reward2 = 0
        
        for i in range(EPISODE_LENGTH):
            action1, action2 = state2[i][0], state1[i][0]
            r1, r2 = reward_matrix[action1][action2]
            
            total_reward1 += r1
            total_reward2 += r2
            
            if action1 == 0:
                Nc_agent += 1
            else:
                Nd_agent += 1
            
            if action2 == 0:
                Nc_mentor += 1
            else:
                Nd_mentor += 1
        
        if isinstance(player1, ConstrainedAgentQ):
            restricted_scores.append(total_reward1)
        else:
            free_scores.append(total_reward1)
        mentor_scores.append(total_reward2)
        
        print(f"Score: {round(total_reward1,1)} to {round(total_reward2,1)} - {player2.strategy_print()}")
        
        if total_reward1 > total_reward2:
            wins_agent += 1
        elif total_reward2 > total_reward1:
            wins_mentor += 1
        else:
            ties += 1
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Agent wins: {wins_agent}")
    print(f"Mentor wins: {wins_mentor}")
    print(f"Ties: {ties}")
    print(f"Agent cooperation rate: {Nc_agent / (episodes * EPISODE_LENGTH):.3f}")
    print(f"Mentor cooperation rate: {Nc_mentor / (episodes * EPISODE_LENGTH):.3f}")
    
    print("\n===== PERFORMANCE MÉDIA =====")
    free_mean = np.mean(free_scores) if free_scores else 0
    restricted_mean = np.mean(restricted_scores) if restricted_scores else 0
    mentor_mean = np.mean(mentor_scores) if mentor_scores else 0
    
    print(f"Free Q-Agent: {free_mean:.2f} ± {np.std(free_scores):.2f}")
    print(f"Restricted Q-Agent: {restricted_mean:.2f} ± {np.std(restricted_scores):.2f}")
    print(f"Evolutionary Strategies: {mentor_mean:.2f} ± {np.std(mentor_scores):.2f}")
    
    return {
        'wins_agent': wins_agent,
        'wins_mentor': wins_mentor,
        'ties': ties,
        'coop_agent': Nc_agent / (episodes * EPISODE_LENGTH),
        'coop_mentor': Nc_mentor / (episodes * EPISODE_LENGTH),
        'free_scores': free_scores,
        'restricted_scores': restricted_scores,
        'mentor_scores': mentor_scores,
        'free_mean': free_mean,
        'restricted_mean': restricted_mean,
        'mentor_mean': mentor_mean,
        'free_std': np.std(free_scores) if free_scores else 0,
        'restricted_std': np.std(restricted_scores) if restricted_scores else 0,
        'mentor_std': np.std(mentor_scores) if mentor_scores else 0
    }

# ============================================================
# STATISTICAL ANALYSIS WITH FREE AGENTS INCLUDED
# ============================================================

def perform_statistical_analysis(population, mentors, free_scores, restricted_scores, mentor_scores, episodes=STATISTICAL_EPISODES):
    # Coletar scores das estratégias mentoras
    strategy_scores = {mentor.strategy_print(): [] for mentor in set(mentors)}
    strategy_names = {mentor: mentor.strategy_print() for mentor in mentors}
    
    for episode in range(episodes):
        state1, state2 = [], []
        player1 = random.choice(population)
        player2 = random.choice(mentors)
        
        for i in range(EPISODE_LENGTH):
            action1 = player1.pick_action(state1)
            action2 = player2.pick_action(state2)
            state1.append([action2, action1])
            state2.append([action1, action2])
        
        total_reward2 = 0
        for i in range(EPISODE_LENGTH):
            action1, action2 = state2[i][0], state1[i][0]
            r1, r2 = reward_matrix[action1][action2]
            total_reward2 += r2
        
        strategy_name = strategy_names[player2]
        strategy_scores[strategy_name].append(total_reward2)
    
    # Adicionar agentes livres como uma "estratégia" separada
    strategy_scores['Free Q-Agent'] = free_scores
    strategy_scores['Restricted Q-Agent'] = restricted_scores
    
    # Criar DataFrame
    rows = []
    for strategy, scores in strategy_scores.items():
        for score in scores:
            rows.append({'estrategia': strategy, 'score': score})
    
    df = pd.DataFrame(rows)
    stats = df.groupby('estrategia')['score'].agg(media='mean', desvio='std', count='count')
    
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS (Including Free Agents)")
    print("="*50)
    print(stats)
    
    create_visualizations_with_free_agents(df, stats)
    
    return df, stats

# ============================================================
# VISUALIZATIONS WITH FREE AGENTS INCLUDED - ALL VIOLET-ROSE
# ============================================================

def create_visualizations_with_free_agents(df, stats):
    # Sort strategies for consistent ordering
    strategies = sorted(stats.index)
    n_strategies = len(strategies)
    
    # Generate violet-rose gradient colors
    colors = get_color_gradient(n_strategies, 'cool')
    
    # ============================================================
    # GRÁFICO 1: Média e Desvio Padrão
    # ============================================================
    plt.figure(figsize=(14, 7))
    bars = plt.bar(strategies, stats.loc[strategies, 'media'], 
                   yerr=stats.loc[strategies, 'desvio'], 
                   capsize=8, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Aplicar gradiente adicional nas barras
    for i, bar in enumerate(bars):
        cmap = plt.cm.cool
        bar.set_color(cmap(i / max(n_strategies-1, 1)))
    
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.ylabel("Score Médio", fontsize=12, fontweight='bold')
    plt.title("Média e Desvio Padrão por Tipo de Agente/Estratégia", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Adicionar valores nas barras
    for bar, strategy in zip(bars, strategies):
        value = stats.loc[strategy, 'media']
        std = stats.loc[strategy, 'desvio']
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 2: Boxplot
    # ============================================================
    plt.figure(figsize=(14, 7))
    
    data_for_box = []
    labels_for_box = []
    
    for strategy in strategies:
        scores = df[df['estrategia'] == strategy]['score'].values
        data_for_box.append(scores)
        labels_for_box.append(strategy)
    
    bp = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Colorir os boxplots com violet-rose
    for i, patch in enumerate(bp['boxes']):
        if i < len(colors):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
    
    # Colorir os whiskers e medianas
    for i, (line, color) in enumerate(zip(bp['medians'], colors)):
        line.set_color(color)
    
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Distribuição dos Scores por Tipo de Agente/Estratégia", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 3: Histograma Comparativo
    # ============================================================
    plt.figure(figsize=(14, 7))
    
    # Criar histogramas para cada tipo com transparência
    for i, strategy in enumerate(strategies):
        scores = df[df['estrategia'] == strategy]['score'].values
        color = colors[i % len(colors)]
        plt.hist(scores, bins=20, alpha=0.5, label=strategy, 
                color=color, edgecolor='black', linewidth=0.5, density=True)
    
    plt.xlabel("Score", fontsize=12, fontweight='bold')
    plt.ylabel("Densidade", fontsize=12, fontweight='bold')
    plt.title("Distribuição de Scores por Tipo de Agente/Estratégia", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 4: Violin Plot
    # ============================================================
    plt.figure(figsize=(14, 7))
    
    parts = plt.violinplot(data_for_box, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
    
    parts['cmeans'].set_color(VIOLET_ROSE_COLORS[0])
    parts['cmedians'].set_color(VIOLET_ROSE_COLORS[7])
    
    plt.xticks(range(1, len(labels_for_box) + 1), labels_for_box, 
               rotation=25, ha='right', fontsize=10)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Distribuição de Scores (Violin Plot) - Todos os Agentes", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 5: Scatter Plot com Jitter
    # ============================================================
    plt.figure(figsize=(14, 7))
    
    for i, strategy in enumerate(strategies):
        scores = df[df['estrategia'] == strategy]['score'].values
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        color = colors[i % len(colors)]
        plt.scatter(x, scores, alpha=0.3, s=10, color=color, label=strategy)
    
    plt.xticks(range(1, len(strategies) + 1), strategies, rotation=25, ha='right', fontsize=10)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Distribuição de Scores (Scatter Plot com Jitter)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()

# ============================================================
# COMPARATIVE GRAPHS - ALL VIOLET-ROSE
# ============================================================

def plot_comparative_graphs(free_scores, restricted_scores, mentor_scores):
    nomes = ["Free Q", "Restricted Q", "Evolutionary"]
    medias = [np.mean(free_scores), np.mean(restricted_scores), np.mean(mentor_scores)]
    desvios = [np.std(free_scores), np.std(restricted_scores), np.std(mentor_scores)]
    
    # Violet-rose colors
    colors_gradient = [VIOLET_ROSE_COLORS[2], VIOLET_ROSE_COLORS[5], VIOLET_ROSE_COLORS[7]]
    
    # ============================================================
    # GRÁFICO 1: Barras com valores
    # ============================================================
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nomes, medias, color=colors_gradient, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bar, value in zip(bars, medias):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel("Average Score", fontsize=12, fontweight='bold')
    plt.title("Average Performance Comparison", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 2: Com desvio padrão
    # ============================================================
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nomes, medias, yerr=desvios, capsize=8,
                   color=colors_gradient, edgecolor='black', 
                   linewidth=1.5, alpha=0.8, ecolor=VIOLET_ROSE_COLORS[0])
    
    for bar, value, std in zip(bars, medias, desvios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.ylabel("Average Score", fontsize=12, fontweight='bold')
    plt.title("Mean Score ± Standard Deviation", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 3: Boxplot comparativo
    # ============================================================
    plt.figure(figsize=(10, 6))
    data = [free_scores, restricted_scores, mentor_scores]
    bp = plt.boxplot(data, labels=nomes, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Distribuição Comparativa dos Scores", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # GRÁFICO 4: Violin plot
    # ============================================================
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_gradient[i % len(colors_gradient)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    parts['cmeans'].set_color(VIOLET_ROSE_COLORS[0])
    parts['cmedians'].set_color(VIOLET_ROSE_COLORS[7])
    
    plt.xticks([1, 2, 3], nomes, fontsize=11)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Distribuição de Scores (Violin Plot)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.show()

# ============================================================
# LEARNING STABILITY VISUALIZATION
# ============================================================

def plot_learning_stability(population):
    agent = max(population, key=lambda a: len(a.get_avgdelta_list()))
    avg_deltas = agent.get_avgdelta_list()
    
    if avg_deltas:
        x = [i * 1000 for i in range(len(avg_deltas))]
        y = avg_deltas
        
        plt.figure(figsize=(12, 6))
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        for i in range(len(segments)):
            color_value = i / len(segments)
            plt.plot(segments[i, :, 0], segments[i, :, 1], 
                    color=plt.cm.cool(color_value * 0.8 + 0.2),
                    linewidth=2)
        
        scatter = plt.scatter(x, y, c=x, cmap='cool', s=10, alpha=0.5)
        plt.colorbar(scatter, label='Decision Count')
        
        plt.xlabel('Decision Count', fontsize=12, fontweight='bold')
        plt.ylabel('Strategy Change Magnitude', fontsize=12, fontweight='bold')
        plt.title('Strategy Change Magnitude Over Decision Count', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
        print("No learning data available for stability graph")

# ============================================================
# SAVE Q-TABLES
# ============================================================

def save_q_tables(population):
    Qtable_all = [sorted(agent.Q.items()) for agent in population]
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    with open(f'Q_table_{current_time}.txt', 'w') as f:
        for q_table in Qtable_all:
            f.write(json.dumps(q_table) + '\n')
    
    print(f"Q-tables saved to Q_table_{current_time}.txt")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main(show_history_comparison=True):
    print("="*60)
    print(" MULTIAGENT Q-LEARNING FOR EVOLUTIONARY GAMES ".center(60, "="))
    print("="*60)
    print("\nFramework Integrado: Agentes Livres vs Restritos vs Mentores")
    
    # Initialize history
    history = ExperimentHistory()
    
    # Ask if user wants to clear history
    if history.history:
        print(f"\nFound {len(history.history)} previous experiments in memory. (•_•)")
        clear = input("Clear experiment history? (y/n): ").lower()
        if clear == 'y':
            history.clear_history()
    
    # Show previous experiments if available
    if show_history_comparison:
        prev_experiments = history.get_latest_experiments(3)
        if prev_experiments:
            print("\nPrevious experiments found:")
            for i, exp in enumerate(prev_experiments):
                print(f"  Experiment {i+1}: {exp['timestamp']}")
                print(f"    Free: {exp.get('free_score_mean', 0):.2f} ± {exp.get('free_score_std', 0):.2f}")
                print(f"    Restricted: {exp.get('restricted_score_mean', 0):.2f} ± {exp.get('restricted_score_std', 0):.2f}")
                print(f"    Evolutionary: {exp.get('mentor_score_mean', 0):.2f} ± {exp.get('mentor_score_std', 0):.2f}")
            
            compare = input("\nCompare with previous experiments? (y/n): ").lower()
            if compare == 'y':
                history.compare_experiments()
    
    # Initialize
    print("\nInitializing population and mentors...")
    population = initialize_population()
    mentors = initialize_mentors()
    print(f"Population: {len(population)} agents")
    print(f"Mentors: {len(mentors)} agents")
    
    # Train
    print("\nStarting training phase...")
    population = train_agents(population, mentors)
    
    # Save Q-tables
    save_q_tables(population)
    
    # Test
    print("\nStarting testing phase...")
    test_results = test_agents(population, mentors)
    
    # Statistical analysis with free agents included
    print("\nPerforming statistical analysis including free agents...")
    df, stats = perform_statistical_analysis(
        population, 
        mentors,
        test_results['free_scores'],
        test_results['restricted_scores'],
        test_results['mentor_scores']
    )
    
    # Comparative graphs
    print("\nGenerating comparative graphs...")
    plot_comparative_graphs(
        test_results['free_scores'],
        test_results['restricted_scores'],
        test_results['mentor_scores']
    )
    
    # Learning stability
    print("\nGenerating learning stability graph...")
    plot_learning_stability(population)
    
    # Save to history
    config = {
        'population_size': POPULATION_SIZE,
        'mentor_instances': MENTOR_INSTANCES,
        'mentor_types': MENTOR_TYPES,
        'episode_length': EPISODE_LENGTH,
        'max_decisions': max_decisions,
        'alpha': alpha,
        'gamma': gamma,
        'memory_size': MEMORY_SIZE
    }
    
    history.save_experiment(test_results, stats, config)
    print("\nExperiment saved to history. (⌐■_■)")
    
    # Show comparison after saving
    if show_history_comparison:
        show = input("\nShow full experiment history comparison? (y/n): ").lower()
        if show == 'y':
            history.compare_experiments()
    
    print("\n" + "="*60)
    print(" EXPERIMENT COMPLETE ".center(60, "="))
    print("="*60)
    
    return population, mentors, test_results, df, stats

if __name__ == "__main__":
    population, mentors, test_results, df, stats = main()
