# Formal Framework

This document defines the Behavioral Grounding Framework (BGF) mathematically and introduces two quantitative metrics for evaluating simulation realism.

---

## 1. BGF Instance Definition

A BGF simulation instance is a tuple:

    BGF = (A, E, G, P, Phi, T)

where:

- **A = {a_1, ..., a_N}** is a set of N agents. Each agent a_i has an immutable profile pi_i = Phi(x_i) where x_i is a record sampled from the empirical distribution D_ESS, and a mutable state s_i(t) = (wealth, stress, satisfaction, trust_map) at time step t.

- **E = (S, u)** is the economic environment, with state space S and payoff function u: Action x S -> R defined by:
  - u(work, s) = (+8 wealth, +0.1 stress)
  - u(save, s) = (+4 wealth, -0.05 stress)
  - u(cooperate, s) = (-3 wealth from self, +12/|cooperators| to public pool, -0.05 stress)

- **G = (V, E_G, theta)** is the social graph, where V = A, E_G are directed edges representing cooperation history, and theta are topology parameters (Watts-Strogatz rewiring probability beta, mean degree k). The graph evolves: cooperation events add edges.

- **P: Profile x State x Memory x Context -> Action** is the decision policy. For LLM-based policies, P(pi, s, m, c) = parse(LLM(prompt(pi, s, m, c))) where prompt() constructs a token-budgeted message from agent state and RAG-retrieved context.

- **Phi: D_ESS -> Profile** is the empirical grounding function that maps European Social Survey microdata records to agent profiles, preserving joint distributions of trust, risk tolerance, political orientation, income, education, and 10+ additional sociodemographic attributes.

- **T** is the simulation horizon (number of rounds).

---

## 2. Action Space

At each round t, agent a_i selects an action from:

    Action = {work, save, cooperate}

subject to constraints:
- cooperate requires a valid target agent in the agent's network neighborhood
- amounts are bounded [0, 20]
- confidence scores are bounded [0, 1]

Adversarial agents (bad apples) are hard-constrained to steal, which is outside the normal action space and handled by the EconomyEngine.

---

## 3. Behavioral Realism Metric (BRM)

The BRM quantifies how closely a simulated distribution matches the empirical target.

### 3.1 BRM-JSD (Single Dimension)

For a single distributional quantity (e.g., wealth):

    BRM_JSD(sim, emp) = 1 - JSD(D_sim || D_ESS)

where JSD is the Jensen-Shannon Divergence computed with base-2 logarithm and shared histogram bins.

Properties:
- BRM_JSD in [0, 1]
- BRM_JSD = 1 when D_sim = D_ESS (identical distributions)
- BRM_JSD = 0 when D_sim and D_ESS have disjoint support
- BRM_JSD is symmetric: BRM_JSD(sim, emp) = BRM_JSD(emp, sim)

### 3.2 Composite BRM

The composite metric aggregates four sub-dimensions:

    BRM_composite = w_1 * BRM_JSD(wealth)
                  + w_2 * (1 - |Gini_sim - Gini_ESS|)
                  + w_3 * (1 - |coop_sim - coop_ESS|)
                  + w_4 * (1 - JSD_temporal)

where:
- w_1 = 0.30 (wealth distribution match)
- w_2 = 0.25 (inequality calibration)
- w_3 = 0.25 (cooperation rate accuracy)
- w_4 = 0.20 (temporal behavioral stability)
- sum(w_i) = 1.0

Each component is individually bounded in [0, 1] (1 = better), and the composite is therefore also in [0, 1].

The default weights reflect the research priorities of the BGF: distributional fidelity (w_1) is weighted highest because it is the most direct measure of population realism, while stability (w_4) is weighted lowest because some behavioral variation across rounds is expected.

---

## 4. RLHF Bias Index

The RLHF Bias Index quantifies how far an LLM policy's action distribution deviates from a uniform (unbiased) prior:

    B_RLHF(pi) = TV(pi, pi_uniform) = 0.5 * sum_{a in A} |pi(a) - 1/|A||

where:
- pi(a) is the probability of action a under the observed policy
- pi_uniform(a) = 1/|A| = 1/3 for the BGF action space
- TV is the total variation distance

Properties:
- B_RLHF in [0, 1]
- B_RLHF = 0 when the policy is perfectly uniform (no bias)
- B_RLHF = 2/3 when the policy deterministically selects one action
- Higher B_RLHF indicates stronger alignment-induced behavioral bias

### Central Claim

For a BGF instance with grounding function Phi derived from D_ESS:

    BRM(Condition_B) > BRM(Condition_A)
    B_RLHF(Condition_B) < B_RLHF(Condition_A)

where Condition A is the ungrounded LLM baseline and Condition B is the ESS-grounded configuration with dual RAG pipelines.

In words: grounding reduces the RLHF cooperative bias while simultaneously increasing behavioral realism.

---

## 5. Notation Summary

| Symbol | Definition |
|--------|-----------|
| D_ESS | Empirical distribution from European Social Survey Round 11 |
| D_sim | Simulated distribution produced by a BGF run |
| Phi | Grounding function: D_ESS -> AgentProfile |
| pi | Policy action distribution |
| BRM_JSD | 1 - JSD(D_sim, D_ESS), in [0, 1] |
| B_RLHF | TV(pi, pi_uniform), in [0, 1] |
| JSD | Jensen-Shannon Divergence (base-2, symmetric) |
| TV | Total Variation distance |
| G | Gini coefficient |
| T | Simulation horizon (rounds) |
| N | Number of agents |
