# ============================================================
# Multiagent Q-Learning for Evolutionary Games
#
# Framework Integrado
#
# Esta versão une:
#
# agentes livres
# agentes restritos
# estratégias mentoras
# métricas evolucionárias
# documentação completa
#
# Objetivo:
#
# comparar agentes totalmente livres
# contra agentes com restrições estratégicas
# dentro do Dilema do Prisioneiro iterado
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

alpha = 0.2

gamma = 0.5



# ============================================================
# PRISONER'S DILEMMA REWARD MATRIX
#
# R = recompensa cooperação mútua
# S = punição cooperador enganado
# T = tentação de trair
# P = punição dupla traição
# ============================================================

R = 2
S = 0
T = 3
P = 0.1


reward_matrix = [[[R, R],
                  [S, T],
                  [T, S],
                  [P, P]]]



# ============================================================
# IMPORTS
# ============================================================

import sys
import random
from time import time
from matplotlib import pyplot as plt
import json
from collections import OrderedDict
import numpy as np
import pickle
from datetime import datetime



# ============================================================
# RULE SETS
#
# Rule sets limitam as ações permitidas
# para agentes restritos.
#
# state:
# histórico observado
#
# actions:
# ações disponíveis
#
# retorno:
# lista de ações permitidas
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



# ============================================================
# HUMAN AGENT
#
# Permite interação manual
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

                action = int(
                    input(
                        "Choose Cooperate/Defect (0/1): "
                    )
                )

            except ValueError:

                print("Please input a number.")

        return action


    def reward_action(self, state, action, reward):

        pass



# ============================================================
# Q-LEARNING AGENT
#
# Agente principal do sistema.
#
# Aprende:
#
# Q(s,a)
#
# s = estado
# a = ação
# ============================================================

class AgentQ:

    def __init__(self, memory):

        self.wins = 0

        self.losses = 0

        self.Q = {}

        self.memory = memory

        self.epsilon_counter = 1

        self.delta = []

        self.avgdelta_list = []


    # ========================================================
    # Retorna Q-values do estado atual
    # ========================================================

    def get_q(self, state):

        quality1 = self.Q[
            str(state[-self.memory:])
        ][0]

        quality2 = self.Q[
            str(state[-self.memory:])
        ][1]

        return quality1, quality2


    # ========================================================
    # Atualiza Q-values
    # ========================================================

    def set_q(
        self,
        state,
        quality1,
        quality2
    ):

        self.Q[
            str(state[-self.memory:])
        ][0] = quality1

        self.Q[
            str(state[-self.memory:])
        ][1] = quality2


    # ========================================================
    # Normalização da tabela Q
    #
    # evita crescimento excessivo
    # ========================================================

    def normalize_q(self, state):

        quality1, quality2 = self.get_q(state)

        normalization = min(
            quality1,
            quality2
        )

        self.set_q(
            state,
            (quality1 - normalization) * 0.95,
            (quality2 - normalization) * 0.95
        )


    # ========================================================
    # Política epsilon-greedy
    #
    # mistura:
    #
    # exploração
    # exploração gulosa
    # ========================================================

    def max_q(self, state):

        if state == []:

            quality1, quality2 = 0, 0

        else:

            quality1, quality2 = self.get_q(state)

        if (
            quality1 == quality2
            or random.random() < (500 / self.epsilon_counter)
        ):

            return random.randint(0, 1)

        elif quality1 > quality2:

            return 0

        else:

            return 1


    # ========================================================
    # Escolha de ação
    # ========================================================

    def pick_action(self, state):

        self.epsilon_counter += 1

        if self.epsilon_counter % 1000 == 1:

            if len(self.delta) > 0:

                self.avgdelta_list.append(
                    self.delta_average()
                )

        if str(state[-self.memory:]) not in self.Q:

            self.Q[
                str(state[-self.memory:])
            ] = [0, 0]

        return self.max_q(state)


    # ========================================================
    # Atualização Q-Learning
    #
    # Q(s,a) ← Q(s,a) + α[r + γmaxQ(s') − Q(s,a)]
    # ========================================================

    def reward_action(
        self,
        state,
        action,
        reward
    ):

        oldstate = state[-self.memory -1 : -1]

        newstate = state[-self.memory : len(state)]

        oldq = self.Q[str(oldstate)][action]

        maxqnew = max(
            self.Q[str(newstate)]
        )

        self.Q[str(oldstate)][action] = (
            oldq
            + alpha * (
                reward
                + gamma * maxqnew
                - oldq
            )
        )

        self.delta.append(
            reward
            + gamma * maxqnew
            - oldq
        )

        if len(self.delta) > 1000:

            del self.delta[0]


    def mark_victory(self):

        self.wins += 1


    def mark_defeat(self):

        self.losses += 1


    # ========================================================
    # Estatísticas de convergência
    # ========================================================

    def delta_average(self):

        return sum(self.delta) / len(self.delta)


    def get_avgdelta_list(self):

        return self.avgdelta_list


    # ========================================================
    # Análise comportamental
    # ========================================================

    def analyse(self):

        percent_won = 0

        if self.wins > 0:

            percent_won = float(self.wins) / (
                self.wins + self.losses
            )

        times_cooperated = 0

        times_defected = 0

        for state in self.Q:

            action = self.max_q(eval(state))

            if action == 0:

                times_cooperated += 1

            else:

                times_defected += 1

        percent_cooperated = 0

        if times_cooperated > 0:

            percent_cooperated = (
                float(times_cooperated)
                / len(self.Q)
            )

        return (
            self.wins,
            percent_won,
            percent_cooperated
        )


    def reset_analysis(self):

        self.wins = 0

        self.losses = 0



# ============================================================
# CONSTRAINED Q AGENT
#
# Variante do AgentQ
#
# Aprende normalmente
#
# porém:
#
# possui limitações estratégicas
# ============================================================

class ConstrainedAgentQ(AgentQ):

    def __init__(
        self,
        memory,
        rule_set=None
    ):

        super().__init__(memory)

        self.rule_set = rule_set


    # ========================================================
    # Filtra ações permitidas
    # ========================================================

    def filter_actions(
        self,
        state,
        actions
    ):

        if self.rule_set is None:

            return actions

        return self.rule_set(
            state,
            actions
        )


    # ========================================================
    # Política epsilon-greedy restrita
    # ========================================================

    def max_q_restricted(self, state):

        if state == []:

            quality1, quality2 = 0, 0

        else:

            quality1, quality2 = self.get_q(state)

        actions = [0, 1]

        allowed_actions = self.filter_actions(
            state,
            actions
        )

        if len(allowed_actions) == 0:

            allowed_actions = actions

        if (
            quality1 == quality2
            or random.random() < (
                500 / self.epsilon_counter
            )
        ):

            return random.choice(
                allowed_actions
            )

        if (
            quality1 > quality2
            and 0 in allowed_actions
        ):

            return 0

        if (
            quality2 > quality1
            and 1 in allowed_actions
        ):

            return 1

        return random.choice(
            allowed_actions
        )


    # ========================================================
    # Escolha de ação restrita
    # ========================================================

    def pick_action(self, state):

        self.epsilon_counter += 1

        if self.epsilon_counter % 1000 == 1:

            if len(self.delta) > 0:

                self.avgdelta_list.append(
                    self.delta_average()
                )

        if str(state[-self.memory:]) not in self.Q:

            self.Q[
                str(state[-self.memory:])
            ] = [0, 0]

        return self.max_q_restricted(state)



# ============================================================
# DEFINED AGENTS
#
# Estratégias clássicas
#
# usadas como mentores evolucionários
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


    # ========================================================
    # Nome da estratégia
    # ========================================================

    def strategy_print(self):

        if self.strategy == 0:
            return "TFT"

        elif self.strategy == 1:
            return "GTFT0.3"

        elif self.strategy == 2:
            return "WSLS"

        elif self.strategy == 3:
            return "Holds a grudge"

        elif self.strategy == 4:
            return "Fool me once"

        elif self.strategy == 5:
            return "Omega TFT"

        elif self.strategy == 6:
            return "Gradual TFT"


    # ========================================================
    # Estratégias fixas
    # ========================================================

    def pick_action(self, state):

        # ====================================================
        # TIT FOR TAT
        # ====================================================

        if self.strategy == 0:

            if len(state) == 0:

                return random.randint(0, 1)

            else:

                return state[-1][0]


        # ====================================================
        # GENEROUS TIT FOR TAT
        # ====================================================

        elif self.strategy == 1:

            if len(state) == 0:

                return random.randint(0, 1)

            elif state[-1][0] == 0:

                return state[-1][0]

            elif state[-1][0] == 1:

                Pc = 1/3

                if random.random() < Pc:

                    return 0

                else:

                    return 1


        # ====================================================
        # WIN STAY LOSE SHIFT
        # ====================================================

        elif self.strategy == 2:

            if len(state) == 0:

                return random.randint(0, 1)

            elif state[-1][0] == 0:

                return state[-1][1]

            elif state[-1][0] == 1:

                return 1 - state[-1][1]


        # ====================================================
        # HOLDS A GRUDGE
        # ====================================================

        elif self.strategy == 3:

            state = np.array(state)

            if len(state) == 0:

                return random.randint(0, 1)

            elif 1 in state[:,0]:

                return 1

            else:

                return 0


        # ====================================================
        # FOOL ME ONCE
        # ====================================================

        elif self.strategy == 4:

            state = np.array(state)

            if len(state) == 0:

                return random.randint(0, 1)

            defect_count = 0

            for i in range(len(state[:,0])):

                if state[i,0] == 1:

                    defect_count += 1

            if defect_count >= 2:

                return 1

            elif defect_count == 1 and state[-1][0] == 1:

                return 1

            else:

                return 0


        # ====================================================
        # OMEGA TFT
        # ====================================================

        elif self.strategy == 5:

            state = np.array(state)

            if len(state) == 0:

                self.deadlock_threshold = 3

                self.randomness_threshold = 8

                self.randomness_counter = 0

                self.deadlock_counter = 0

                return random.randint(0, 1)

            if len(state[:,0]) == 1:

                return state[-1,0]

            if self.deadlock_counter >= self.deadlock_threshold:

                move = 0

                if self.deadlock_counter == self.deadlock_threshold:

                    self.deadlock_counter = (
                        self.deadlock_threshold + 1
                    )

                else:

                    self.deadlock_counter = 0

            else:

                if (
                    state[-2,0] == [0]
                    and state[-1,0] == [0]
                ):

                    self.randomness_counter -= 1

                if state[-2,0] != state[-1,0]:

                    self.randomness_counter += 1

                if state[-1,0] != state[-1,1]:

                    self.randomness_counter += 1

                if (
                    self.randomness_counter
                    >= self.randomness_threshold
                ):

                    move = 1

                else:

                    move = state[-1][0]

                    if state[-2,0] != state[-1,0]:

                        self.deadlock_counter += 1

                    else:

                        self.deadlock_counter = 0

            return move


        # ====================================================
        # GRADUAL TFT
        # ====================================================

        elif self.strategy == 6:

            state = np.array(state)

            if len(state) == 0:

                self.calm_count = 0

                self.punish_count = 0

                return random.randint(0, 1)

            if self.punish_count > 0:

                self.punish_count -= 1

                return 1

            if self.calm_count > 0:

                self.calm_count -= 1

                return 0

            if state[-1,0] == 1:

                defect_count = 0

                for i in range(len(state[:,0])):

                    if state[i,0] == 1:

                        defect_count += 1

                self.punish_count = defect_count - 1

                self.calm_count = 2

                return 1

            return 0


    def reward_action(self, state, action, reward):

        pass


    def mark_victory(self):

        self.wins += 1


    def mark_defeat(self):

        self.losses += 1


    def analyse(self):

        percent_won = 0

        if self.wins > 0:

            percent_won = float(self.wins) / (
                self.wins + self.losses
            )

        return self.wins, percent_won



# ============================================================
# POPULATION INITIALIZATION
# ============================================================

population = []

mentors = []


# ============================================================
# AGENTES LIVRES
# ============================================================

for i in range(int(POPULATION_SIZE / 2)):

    population.append(
        AgentQ(random.randint(2,2))
    )



# ============================================================
# AGENTES RESTRITOS
# ============================================================

rule_sets = [
    conservative_rules,
    aggressive_rules,
    adaptive_rules,
    stochastic_rules
]

for i in range(int(POPULATION_SIZE / 2)):

    chosen_rule = random.choice(rule_sets)

    population.append(
        ConstrainedAgentQ(
            random.randint(2,2),
            chosen_rule
        )
    )



# ============================================================
# MENTORES
# ============================================================

for i in range(MENTOR_TYPES):

    for j in range(MENTOR_INSTANCES):

        mentors.append(AgentDefined(i))



# ============================================================
# TRAINING PHASE
# ============================================================

while any(
    agent.epsilon_counter < max_decisions
    for agent in population
):

    average_epsilon_counter = (
        sum(
            agent.epsilon_counter
            for agent in population
        )
        / POPULATION_SIZE
    )

    progress = (
        100
        * average_epsilon_counter
        / max_decisions
    )

    sys.stdout.write(
        '\rTraining [{0}] {1}%'.format(
            ('#' * int(progress / 5)).ljust(19),
            int(min(100, progress + 5))
        )
    )

    sys.stdout.flush()

    state1 = []

    state2 = []

    player1 = random.choice(population)

    if player1.epsilon_counter >= max_decisions:

        continue

    player2 = random.choice(
        population + mentors
    )



    # ========================================================
    # EXECUTA EPISÓDIO
    # ========================================================

    for i in range(EPISODE_LENGTH):

        action1 = player1.pick_action(state1)

        action2 = player2.pick_action(state2)

        state1.append([action2, action1])

        state2.append([action1, action2])



    # ========================================================
    # RECOMPENSAS
    # ========================================================

    total_reward1 = 0

    total_reward2 = 0

    reward1 = [0] * EPISODE_LENGTH

    reward2 = [0] * EPISODE_LENGTH


    for i in range(EPISODE_LENGTH):

        action1 = state2[i][0]

        action2 = state1[i][0]

        if action1 == 0 and action2 == 0:

            reward1[i] = reward_matrix[0][0][0]

            reward2[i] = reward_matrix[0][0][1]

        elif action1 == 0 and action2 == 1:

            reward1[i] = reward_matrix[0][1][0]

            reward2[i] = reward_matrix[0][1][1]

        elif action1 == 1 and action2 == 0:

            reward1[i] = reward_matrix[0][2][0]

            reward2[i] = reward_matrix[0][2][1]

        elif action1 == 1 and action2 == 1:

            reward1[i] = reward_matrix[0][3][0]

            reward2[i] = reward_matrix[0][3][1]

        total_reward1 += reward1[i]

        total_reward2 += reward2[i]



    # ========================================================
    # DISTRIBUIÇÃO DE RECOMPENSAS
    # ========================================================

    if total_reward1 >= total_reward2:

        reward_chunk = (
            total_reward1
            / EPISODE_LENGTH
            * (1 - reward_weighting_factor)
        )

        for i in range(EPISODE_LENGTH - 1):

            action1 = state2[i][0]

            player1.reward_action(
                state1[:i+1],
                action1,
                reward_chunk
                + reward1[i]
                * reward_weighting_factor
            )

            action2 = state1[i][0]

            player2.reward_action(
                state2[:i+1],
                action2,
                reward2[i]
                * reward_weighting_factor
            )


    else:

        reward_chunk = (
            total_reward2
            / EPISODE_LENGTH
            * (1 - reward_weighting_factor)
        )

        for i in range(EPISODE_LENGTH - 1):

            action2 = state1[i][0]

            player2.reward_action(
                state2[:i+1],
                action2,
                reward_chunk
                + reward2[i]
                * reward_weighting_factor
            )

            action1 = state2[i][0]

            player1.reward_action(
                state1[:i+1],
                action1,
                reward1[i]
                * reward_weighting_factor
            )



# ============================================================
# TRAINING COMPLETE
# ============================================================

print("")



# ============================================================
# SAVE Q TABLES
# ============================================================

Qtable_all = []

for Agent in population:

    Qtable_all.append(
        sorted(Agent.Q.items())
    )

current_time = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

with open(
    f'Q_table_{current_time}.txt',
    'w'
) as f:

    for d in Qtable_all:

        f.write(json.dumps(d) + '\n')



# ============================================================
# TESTING PHASE
# ============================================================

wins1 = 0

wins2 = 0

tie = 0

Nc_agent = 0

Nc_mentor = 0

Nd_agent = 0

Nd_mentor = 0



for i in range(TESTING_EPISODES):

    state1 = []

    state2 = []

    player1 = random.choice(population)

    player2 = random.choice(mentors)


    for i in range(EPISODE_LENGTH):

        action1 = player1.pick_action(state1)

        action2 = player2.pick_action(state2)

        state1.append([action2, action1])

        state2.append([action1, action2])



    total_reward1 = 0

    total_reward2 = 0


    for i in range(EPISODE_LENGTH):

        action1 = state2[i][0]

        action2 = state1[i][0]

        reward1 = 0

        reward2 = 0


        if action1 == 0 and action2 == 0:

            reward1 = reward_matrix[0][0][0]

            reward2 = reward_matrix[0][0][1]

            Nc_agent += 1

            Nc_mentor += 1


        elif action1 == 0 and action2 == 1:

            reward1 = reward_matrix[0][1][0]

            reward2 = reward_matrix[0][1][1]

            Nc_agent += 1

            Nd_mentor += 1


        elif action1 == 1 and action2 == 0:

            reward1 = reward_matrix[0][2][0]

            reward2 = reward_matrix[0][2][1]

            Nd_agent += 1

            Nc_mentor += 1


        elif action1 == 1 and action2 == 1:

            reward1 = reward_matrix[0][3][0]

            reward2 = reward_matrix[0][3][1]

            Nd_agent += 1

            Nd_mentor += 1


        total_reward1 += reward1

        total_reward2 += reward2



    print(
        "Score: "
        + str(round(total_reward1,1))
        + " to "
        + str(round(total_reward2,1))
        + "      "
        + player2.strategy_print()
    )


    if total_reward1 > total_reward2:

        wins1 += 1

    elif total_reward2 > total_reward1:

        wins2 += 1

    else:

        tie += 1



# ============================================================
# FINAL METRICS
# ============================================================

print("")

print("Player 1 won " + str(wins1) + " times")

print("Player 2 won " + str(wins2) + " times")

print("tie          " + str(tie) + " times")

print(
    "phoC_agent: ",
    Nc_agent / TESTING_EPISODES / EPISODE_LENGTH,
    "  phoC_mentor: ",
    Nc_mentor / TESTING_EPISODES / EPISODE_LENGTH
)



# ============================================================
# LEARNING STABILITY GRAPH
#
# mostra magnitude média de atualização Q
# ============================================================

during = population[0].get_avgdelta_list()

x = [i * 1000 for i in range(len(during))]

y = during

fig = plt.plot(x, y)

plt.xlabel('Decision Count')

plt.ylabel('Strategy Change Magnitude')

plt.title(
    'Strategy Change Magnitude Over Decision Count'
)

plt.show()