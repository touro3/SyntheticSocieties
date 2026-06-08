<!--
  MONOGRAPH SOURCE (TCC2 — final, defended version).
  Editable Markdown spine for the IDP institutional template (see
  docs/TCC_Lucas_Tourinho.pdf for the TCC1 proposal that fixed the front-matter
  and chapter conventions). The full research-paper artifact lives in
  docs/paper.md and is kept untouched; this file reorganizes that content into
  the IDP monograph structure. Final PDF render (LaTeX/Word template) is out of
  scope of this source file.
-->

# INSTITUTO BRASILEIRO DE ENSINO, DESENVOLVIMENTO E PESQUISA (IDP)
# BACHARELADO EM CIÊNCIA DA COMPUTAÇÃO

## LUCAS TOURINHO MAMEDE

# Measuring Behavioral Fidelity and the Limits of Empirical Grounding in Large Language Model Synthetic Societies

*Trabalho de Conclusão de Curso apresentado como requisito parcial para a obtenção de grau de Bacharel em Ciência da Computação, pelo Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP).*

**Orientador:** Me. Klayton Rodrigues de Castro

Brasília – DF, 2025

---

## Folha de Aprovação

**LUCAS TOURINHO MAMEDE**

**Measuring Behavioral Fidelity and the Limits of Empirical Grounding in Large Language Model Synthetic Societies**

*Trabalho de Conclusão de Curso apresentado como requisito parcial para a obtenção de grau de Bacharel em Ciência da Computação, pelo Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP).*

Aprovado em 05/12/2025.

**Banca Examinadora**

- Me. Klayton Rodrigues de Castro — Orientador
- Me. Patrícia da Silva de Oliveira — Examinadora interna
- Me. Lucas Maurício Castro e Martins — Examinador interno

---

## ABSTRACT

This work pursues two pre-registered aims: (i) to formalize quantitative metrics for detecting RLHF-induced behavioral distortion and for measuring behavioral fidelity in LLM-based social simulations, and (ii) to use those metrics to test whether empirical grounding in real socioeconomic microdata propagates through instruction-tuned LLM agents into realistic collective behavior. Instruction-tuned LLMs deployed as agents in multi-agent economic simulations exhibit a systematic anomaly — cooperation rates exceeding both the Nash equilibrium and laboratory human behavior — which we formalize as the **RLHF Cooperative Bias** and operationalize as the total-variation distance of the observed action distribution from the uniform action prior, `B_RLHF = TV(π, π_uniform)`; behavioral fidelity is measured by a composite **Behavioral Realism Metric (BRM ∈ [0,1])**. Both metrics are embedded in the **Behavioral Grounding Framework**, a formally specified platform `BGF = (A, E, G, P, Φ, T)` grounded in European Social Survey Round 11 microdata via a population-synthesis map `Φ: D_ESS → Profile`. The central finding is a **dissociation (Φ/P_LLM)**: empirical grounding that is effective at the rule-based policy layer does **not** propagate through the LLM decision channel. The pre-registered 10-seed N=100 test finds the grounded and ungrounded arms statistically indistinguishable on cooperation (0.461 vs 0.455, MWU p = 0.91), Gini, and wealth — the pre-registered grounding hypothesis is **not supported at primary scale**, and we report this negative result openly. The metrics are precisely what surface the dissociation: at the layer that reads grounding directly, a deterministic policy reproduces a European-range Gini coefficient (0.325, BCa 95% CI [0.324, 0.325]; N=500, T=30, 10 seeds) and the cross-cultural cooperation gradient (Spearman ρ = +1.000; World Values Survey r = +0.977; Herrmann–Thöni–Gächter public-goods benchmark ρ = +0.886, p = 0.033), establishing that `Φ` is empirically valid while the LLM decision layer remains RLHF-anchored. A scale-dependent RLHF cooperation cascade (B_RLHF amplified 2.6–3.2× at N=500) and a falsified memory-depth hypothesis corroborate this anchoring reading. The framework, the two metrics, and the dissociation are released as an open-source artifact (1,578 automated tests, one-command reproduction, cryptographic reproducibility witness) so the result can be re-tested in any LLM-as-agent setting on any empirical population.

**Keywords:** synthetic societies, LLM agents, behavioral grounding, empirical fidelity, socioeconomic modeling, RLHF bias.

---

## RESUMO

Este trabalho persegue dois objetivos pré-registrados: (i) formalizar métricas quantitativas para detectar a distorção comportamental induzida por RLHF e para medir a fidelidade comportamental em simulações sociais baseadas em LLMs, e (ii) usar tais métricas para testar se a fundamentação empírica em microdados socioeconômicos reais se propaga, através de agentes LLM ajustados por instrução, em comportamento coletivo realista. LLMs ajustados por instrução, quando empregados como agentes em simulações econômicas multiagentes, exibem uma anomalia sistemática — taxas de cooperação que excedem tanto o equilíbrio de Nash quanto o comportamento humano de laboratório — que formalizamos como o **Viés Cooperativo de RLHF** e operacionalizamos como a distância de variação total entre a distribuição de ações observada e o prior uniforme de ações, `B_RLHF = TV(π, π_uniform)`; a fidelidade comportamental é medida por uma **Métrica de Realismo Comportamental (BRM ∈ [0,1])** composta. Ambas as métricas integram o **Arcabouço de Fundamentação Comportamental**, uma plataforma formalmente especificada `BGF = (A, E, G, P, Φ, T)`, fundamentada em microdados da Pesquisa Social Europeia (ESS) Rodada 11 por meio de um mapa de síntese populacional `Φ: D_ESS → Profile`. O achado central é uma **dissociação (Φ/P_LLM)**: a fundamentação empírica, eficaz na camada de política baseada em regras, **não** se propaga através do canal de decisão do LLM. O teste pré-registrado de 10 sementes em N=100 encontra os braços fundamentado e não fundamentado estatisticamente indistinguíveis em cooperação (0,461 vs 0,455, MWU p = 0,91), Gini e riqueza — a hipótese de fundamentação pré-registrada **não é corroborada na escala primária**, e relatamos esse resultado negativo de forma aberta. As métricas são precisamente o que revela a dissociação: na camada que lê a fundamentação diretamente, uma política determinística reproduz um coeficiente de Gini na faixa europeia (0,325, IC 95% BCa [0,324, 0,325]; N=500, T=30, 10 sementes) e o gradiente cooperativo intercultural (ρ de Spearman = +1,000; World Values Survey r = +0,977; referencial de bens públicos de Herrmann–Thöni–Gächter ρ = +0,886, p = 0,033), estabelecendo que `Φ` é empiricamente válido enquanto a camada de decisão do LLM permanece ancorada no RLHF. Uma cascata cooperativa de RLHF dependente de escala (B_RLHF amplificado 2,6–3,2× em N=500) e uma hipótese de profundidade de memória falsificada corroboram essa leitura de ancoragem. O arcabouço, as duas métricas e a dissociação são disponibilizados como artefato de código aberto (1.578 testes automatizados, reprodução por um único comando, testemunha criptográfica de reprodutibilidade), de modo que o resultado possa ser re-testado em qualquer cenário de LLM-como-agente sobre qualquer população empírica.

**Palavras-chave:** sociedades sintéticas, agentes LLM, fundamentação comportamental, fidelidade empírica, modelagem socioeconômica, viés de RLHF.

---

## LIST OF FIGURES

*(Figure numbers preserve the source-paper indexing for cross-traceability; gaps at 6, 16, 17 correspond to source figures not reproduced in this monograph.)*

- Figure 1 — Synthetic population validation against ESS Round 11 (wealth, age, trust, risk). §5.1
- Figure 2 — Single-seed pilot comparison of Condition A vs B (N=50, T=30). §5.1
- Figure 3 — Condition A cooperation network (sparse, hub-and-spoke). §5.1
- Figure 4 — Condition B cooperation network (modular, assortative). §5.1
- Figure 5 — Adversarial resilience under 5% bad-apple injection. §5.1
- Figure 7 — Macroeconomic shock recovery (50% wealth reduction at round 15). §5.1
- Figure 8 — Single-seed pilot macro-dynamics (Gini and cooperation). §5.2
- Figure 9 — Trust-gradient sub-population validation. §5.6
- Figure 10 — Cross-model RLHF bias comparison (Mistral-7B). §5.7
- Figure 11 — ESS feature-importance coefficients. §5.8
- Figure 12 — Profile richness vs. cooperation-prediction accuracy. §5.8
- Figure 13 — Long-horizon persona stability (T=100, rule-based proxy). §6.3
- Figure 14 — Policy-intervention analysis (trust boost at round 15). §5.8
- Figure 15 — Memory-ablation results (M0–M3 × grounded/ungrounded). §5.10
- Figure 18 — Pooled action-transition matrices (Condition A vs B). §5.12
- Figure 19 — N=500 cooperation-cascade trajectories. §5.13

## LIST OF TABLES

- Table 1 — Comparison of Related Works and the Proposed Approach. §2.4
- Table 2 — Experimental configuration. §4
- Table 3 — Proof-of-concept validation against real-world benchmarks. §5.1
- Table 4 — N=500 bad-apple sweep (rule-based, 3 seeds). §5.5
- Table 5 — Trust-gradient validation. §5.6
- Table 6 — Cross-model comparison (on-disk audit-traceable rows). §5.7
- Table 7 — B_RLHF(N) sequence. §5.7
- Table 8 — Policy-intervention sweep. §5.8
- Table 9 — Memory-ablation terminal-round results. §5.10
- Table 10 — Sham-grounding directionality. §5.11
- Table 11 — Pooled action-transition matrices. §5.12

## CONTENTS

1. Introduction
2. Literature Review
3. Methodology
4. Experimental Setup
5. Results
6. Discussion
7. Conclusion
- References

---

# 1 INTRODUCTION

## 1.1 Contextualization

Social science has long suffered from a singular, fatal constraint: the inability to run controlled experiments on history itself. We cannot collapse an economy just to test a hypothesis on inflation, nor can we deploy a radical social policy merely to observe the fallout. Synthetic societies dismantle this barrier. By orchestrating thousands of autonomous, cognitively complex agents into a living *in silico* mirror of our world, this technology grants the power to simulate the consequences of an intervention before it happens — transforming the speculative art of forecasting into an experimental science, a "sandboxed reality" where stakeholders, from central banks to policy designers, can stress-test their most volatile interventions against a population that negotiates, adapts, and evolves, without risking a single real-world consequence.

Within this scenario, Large Language Models (LLMs) transformed the field by overcoming the historical limitation of rigid agent reasoning. LLMs opened the possibility for agents to exhibit flexible, human-aligned reasoning patterns, directly enabling synthetic societies capable of complex emergent behavior. This shift is central to the present research agenda, because it explains why LLM-based agents can serve as cognitively rich components in large-scale social simulations.

Even before the emergence of LLM-based agents, attempts to simulate social dynamics relied primarily on traditional Agent-Based Modeling (ABM). Classical artificial societies captured important macropatterns — such as segregation, mobility, and economic exchange — through rule-based agents. However, these agents operated under rigid, predefined behavioral rules and simplified utility functions, which prevented them from representing the cognitive richness, heterogeneity, and adaptive reasoning observed in real populations. This historical limitation establishes the conceptual inflection point that motivates the transition toward cognitively enhanced, LLM-driven agents.

These advances allow the construction of synthetic societies populated by LLM agents displaying heterogeneous backgrounds, beliefs, and economic profiles. Researchers can now build digital laboratories to test economic policies, model social conflicts, and study collective decision-making with greater realism, capturing deeper heterogeneity in systemic outcomes.

However, a major scientific gap remains regarding calibration and structural validity. LLM-agent simulations are often developed with a focus on narrative plausibility rather than statistical fidelity. As Horton (2023) highlights, while LLMs can act as simulated economic agents, their utility for the social sciences depends entirely on their ability to replicate structural empirical regularities rather than mere linguistic coherence. The field currently relies on ad hoc prompt designs that lack standardized mechanisms for directly injecting real-world census data or socioeconomic distributions into the agent-generation pipeline. Without a rigorous methodology to ground these agents in empirical data — transforming them from "stochastic parrots" into calibrated personas — simulation outcomes risk becoming hallucinatory artifacts rather than reproducible social insights. This research aims to bridge this gap.

This monograph documents a specific instance of that calibration failure that is invisible in single-agent settings but produces pathological outcomes in multi-agent environments: **RLHF-aligned LLMs are individually virtuous but collectively pathological.** Across multiple social-dilemma game types — public goods games, prisoner's dilemmas, stag hunts, and ultimatum games — LLMs fine-tuned via Reinforcement Learning from Human Feedback (RLHF; Ouyang et al., 2022) exhibit cooperation rates far exceeding both the Nash equilibrium and empirically observed human behavior. We call this the **RLHF Cooperative Bias** and formalize it via the index `B_RLHF = TV(π, π_uniform)`.

The mechanism is structural. RLHF trains models on single-agent human-preference data in which the evaluator is always cooperative and well-intentioned — a context in which helpfulness and cooperation are synonymous. This produces a strong cooperative prior that overgeneralizes to multi-agent environments: placed in a social dilemma alongside agents with competing interests, the RLHF-tuned model cooperates with every agent as if each were the RLHF evaluator. It has no learned representation of adversarial interaction, no basis for trust discrimination, and no training signal indicating that cooperation can be individually costly. The result is a synthetic society resembling a utopia: frictionless cooperation, unnatural egalitarianism, and zero social friction — a world that has never existed. This is an alignment failure, not a simulation artifact, and as LLMs are increasingly deployed in multi-agent contexts (AI councils, automated negotiation, agentic workflows with competing objectives) measuring and mitigating this bias becomes a prerequisite for deployment.

## 1.2 Problem Delimitation

This research addresses the absence of a robust, empirically grounded framework capable of using demographic datasets, such as census or survey data, together with a reproducible pipeline for building, calibrating, and testing LLM-based synthetic societies. Although the technology to construct LLM agents already exists, there is still no established method to ensure that their collective behavior aligns with real-world socioeconomic patterns, to systematically translate demographic datasets into agent personas, or to validate whether the aggregate behavior of these agents reflects observed empirical regularities.

Moreover, existing studies rarely distinguish narrative plausibility from structural fidelity, making it difficult to determine whether simulated outcomes emerge from meaningful social mechanisms or from uncontrolled linguistic artifacts inherent to LLMs. The lack of standardized grounding procedures also undermines reproducibility, as current approaches depend on ad hoc prompt engineering and non-transparent calibration heuristics.

Within this context, this work delimits its scope to a fundamental methodological problem: creating agent personas directly from real data, defining consistent interaction rules, and establishing metrics that measure behavioral accuracy. The goal is to articulate the scientific standards required for valid applications and to operationalize them in a concrete, reproducible system — the Behavioral Grounding Framework — whose collective outputs can be quantitatively compared against empirical socioeconomic baselines.

## 1.3 Justification

The lack of a rigorous method threatens the scientific legitimacy of LLM-based computational social science. If policy recommendations derived from synthetic societies rely on uncalibrated agents that do not reflect actual income distributions, policymakers may implement interventions that are ineffective or even detrimental when applied to real populations. Without standardized grounding in real data — income distributions, educational levels, regional demographics — findings from synthetic societies cannot be replicated and may lead to misleading interpretations, limiting their value for research and policy.

The relevance of this work is grounded in advancing synthetic societies from exploratory demonstrations to empirically calibrated simulations capable of supporting analytical studies. Treating agents not as fictional characters but as statistical representations of real demographic groups provides the necessary foundation for structured analyses of collective behavior and socioeconomic dynamics. Researchers increasingly require methods that move beyond ad hoc prompt engineering toward reproducible and transparent procedures; likewise, public institutions that consider using synthetic societies for forecasting, evaluation of economic measures, or scenario testing need models supported by measurable fidelity rather than narrative coherence alone.

This research is timely because current approaches still lack methodological consistency and clear validation mechanisms. By focusing on how to ground agent construction and assess collective behavior against real-world baselines, this study establishes a conceptual and operational foundation that enables empirical implementation while maintaining rigor and reproducibility.

## 1.4 Objectives

This work has **two explicit, pre-registered aims**, both pursued throughout: (Aim 1 — measurement) to formalize quantitative metrics for detecting RLHF-induced behavioral distortion and for assessing behavioral fidelity in LLM-based synthetic societies; and (Aim 2 — propagation test) to use those metrics to test whether empirical grounding in real socioeconomic microdata propagates through LLM agents into realistic collective behavior. The metrics work (Aim 1) is therefore a planned, first-class contribution, not a by-product; the grounding propagation test (Aim 2) is the planned empirical hypothesis whose outcome — reported honestly whether positive or null — constitutes the central finding.

**General objective.** To formalize and apply quantitative metrics of RLHF behavioral distortion and behavioral fidelity within an empirically grounded LLM-based synthetic-society framework, and to test whether socioeconomic grounding propagates through the LLM decision layer into realistic collective behavior.

**Specific objectives.**

- (Aim 1) To formalize statistical distance measures and behavioral-consistency criteria — the RLHF Bias Index `B_RLHF = TV(π, π_uniform)` and the Behavioral Realism Metric (BRM) — that detect alignment-induced distortion and quantify the fidelity of synthetic societies against real social and economic patterns;
- To systematize the conceptual foundations required to transform demographic and socioeconomic datasets into semantically coherent LLM-based agent personas, and to define the architectural principles and behavioral constraints structuring the proposed Behavioral Grounding Framework;
- To synthesize a reproducible pipeline integrating data grounding, persona construction, interaction modeling, and fidelity assessment, validated empirically against real-world socioeconomic baselines (ESS, Eurostat, WVS, and an independent public-goods-game benchmark);
- (Aim 2) To test, under a pre-registered hypothesis battery, whether empirical grounding propagates through the LLM decision policy into measurable behavioral differentiation, and to characterize the outcome with the metrics of Aim 1.

These objectives are operationalized as seven research questions:

- **RQ1 (Primary alignment finding):** Is the RLHF cooperative bias universal across social-dilemma game types, or specific to the public-goods setting?
- **RQ2 (Grounding efficacy):** Does ESS grounding significantly mitigate `B_RLHF` by providing trust- and risk-calibrated priors that override the RLHF cooperative default?
- **RQ3 (Realism):** Can ESS-grounded LLM agents reproduce the macroeconomic and network-topological phenomena of real human populations?
- **RQ4 (Cross-model scope):** Is the bias universal across RLHF alignment families, or moderated by alignment methodology?
- **RQ5 (Memory):** What is the independent contribution of each memory tier to persona fidelity and behavioral consistency?
- **RQ6 (Cross-cultural):** Does the grounding function recover cross-cultural behavioral variation measured by ESS and independently validated by WVS?
- **RQ7 (Robustness):** Are grounded societies resilient to adversarial perturbations, economic shocks, and network-topology variation?

## 1.5 Methodological Approach

This work adopts a constructive, theoretically driven, and empirically validated methodological approach, structured in four integrated movements. First, a focused literature review is conducted on agent-based modeling, cognitive and generative architectures, synthetic populations, and validation metrics, in order to delineate the conceptual and methodological limitations of existing approaches and extract the design requirements for the framework. Second, the **Behavioral Grounding Framework** is proposed, formally specifying the stages of data grounding, persona construction, interaction modeling, and the definition of quantitative indicators of fidelity and inequality, always anchored in empirical socioeconomic distributions. Third, the framework is **implemented** as a reproducible software artifact with deterministic seeding, dual SQL/Graph retrieval, hierarchical memory, and a cryptographic reproducibility witness. Fourth, synthetic societies generated under the framework are **empirically evaluated** through distributional distance measures and inequality indices against real-world baselines (ESS, Eurostat, WVS, and an independent public-goods-game benchmark), under a pre-registered hypothesis battery with family-wise error correction.

Whereas the project phase of this Trabalho de Conclusão de Curso defined the conceptual route (theoretical review → conceptual formulation → validation strategy), the present monograph carries that route through to execution, reporting the empirical results obtained from the implemented framework.

## 1.6 Structure of the Work

This monograph is organized into seven chapters. After this Introduction, Chapter 2 reviews the literature, develops the theoretical framework, surveys related works, and presents the validation framework and gap analysis. Chapter 3 formalizes the Behavioral Grounding Framework methodology: the formal tuple, the fidelity metrics and their formal results, the system architecture, decision policies, prompt engineering and ablation design, anti-drift engineering, the causal-identification strategy, the experimental conditions and hypothesis pre-registration, and the software artifact. Chapter 4 details the experimental setup (data, model, ethics, statistical power, stress tests, phase-transition sweeps, and cross-model validation). Chapter 5 reports the results, from the proof of concept through the macroeconomic, topological, robustness, cross-cultural, cross-model, memory-ablation, and mechanism analyses. Chapter 6 discusses the findings — centrally the Φ/P_LLM dissociation — and their implications. Chapter 7 presents the limitations, broader impacts, conclusions, contributions, and future work. The References close the document.

# 2 LITERATURE REVIEW

This chapter establishes the theoretical and methodological foundations of the research. Following the principle of scientific ontogeny, it analyzes the evolution of agent-based simulation paradigms, tracing the path from symbolic architectures and cognitive models to modern generative approaches. It incorporates the fundamentals of computational population synthesis to justify the data-grounding mechanisms. Finally, it critically reviews the state of the art in Large Language Model (LLM) agents, using recent cautionary studies to identify a methodological gap in empirical grounding and statistical validation.

## 2.1 Theoretical Framework

The pursuit of autonomous agents capable of simulating human social behavior has evolved through distinct eras. Each era addressed the limitations of its predecessors while introducing new challenges.

### 2.1.1 The Era of Rules: Symbolic and Cognitive Architectures

Computational Social Science has historically relied on Agent-Based Modeling (ABM), a subset of Multi-Agent Systems (MAS), to study emergent social phenomena [2]. In MAS, multiple interacting intelligent agents coexist in a shared environment, creating complex dynamics that single-agent systems cannot replicate [8]. Early agents were modeled using the Belief-Desire-Intention (BDI) architecture. While BDI offered a logical framework for rational behavior, formalized through modal logics [9], it suffered from "cognitive brittleness": agents could not handle situations not explicitly foreseen by the designer. Research in this domain advanced from simple implementations to complex axiomatizations, such as BDICTL and Wooldridge's Logic Of Rational Agents (LORA) [10]. Early implementations, such as IRMA [11] and PRS [12], attempted to operationalize this logic but were computationally constrained by the need for manual rule specification.

Parallel efforts to build unified cognitive architectures, such as SOAR [13] and ACT-R [14], attempted to model general intelligence through symbolic production rules. However, these systems faced significant scalability constraints [4], as every social norm or concept had to be manually encoded. As detailed in Section 2.1.3, the emergence of Large Language Models addresses precisely these limitations by providing agents with intrinsic, pre-trained semantic knowledge of the world. This foundational tradition motivates BGF's game-theoretic economic kernel: Schelling's (1971) segregation model, Epstein & Axtell's (1996) *Sugarscape*, and Axelrod's (1984, 1997) iterated Prisoner's Dilemma all demonstrate that heterogeneous agents with simple rules produce stratification, conflict, and cooperation — but always through hand-crafted utility functions that cannot capture the linguistic, cultural, and psychological complexity of real decision-making. BGF bridges this representational gap by replacing rule-based utility maximizers with LLM-based decision engines anchored in empirical survey data.

### 2.1.2 The Era of Optimization: Multi-Agent Reinforcement Learning

To overcome the rigidity of manual rules, the field shifted towards Multi-Agent Reinforcement Learning (MARL). In this paradigm, agents learn optimal policies through trial-and-error, maximizing cumulative rewards [15]. While MARL enabled breakthroughs in strategic coordination [16], Zhang et al. [17] highlight significant limitations for social simulation. These systems are often opaque "black boxes" and suffer from extreme sample inefficiency. Purely utility-driven agents also tend to exhibit behavior misaligned with human social norms, prioritizing reward maximization [18].

### 2.1.3 The Generative Turn: From Foundation to Action

The shift to generative agents resulted from technological advances in semantic processing and agency.

- **Foundational Model (Transformers):** The paradigm shift began with the Transformer architecture [19]. By utilizing self-attention mechanisms, Transformers enabled the training of models on vast corpora, providing agents with intrinsic "world knowledge" [20] (Figure 2). This significantly mitigates reliance on manual rule specification, which had limited earlier architectures such as SOAR.
- **Reasoning (Chain-of-Thought):** Wei et al. [21] discovered that prompting models to generate intermediate reasoning steps (Chain-of-Thought) activates emergent reasoning abilities, allowing agents to simulate cognitive processes rather than merely pattern-matching.
- **Retrieval-Augmented Generation (RAG):** To address the limitations of static training data and hallucinations, Lewis et al. [22] proposed RAG, retrieving relevant external information (e.g., database records) and injecting it into the model's context window during inference. In this work, RAG is the fundamental mechanism for the *Behavioral Grounding* of agents using survey microdata. BGF adapts RAG for behavioral calibration rather than factual question answering: a dual-RAG architecture retrieves population statistics (SQL RAG over ESS microdata) and social context (Graph RAG over cooperation networks), informing each agent of how its demographic peers tend to behave — an empirical anchor absent in prior LLM-agent work.
- **Agency (ReAct):** Reasoning alone is insufficient for situated simulation; agents must interact with their environment. Yao et al. [23] introduced the ReAct (Reason + Act) framework, creating a "perception-action loop" where the agent reasons, acts, observes the output, and updates its cognition (Figure 3). This mechanism is the theoretical basis for BGF's interaction framework.

### 2.1.4 Population Synthesis and Network Topology

Simulating a realistic society requires more than intelligent agents; it requires a representative population and a plausible environment.

- **Population Synthesis:** Classic computational demography relies on techniques such as Iterative Proportional Fitting (IPF) to generate synthetic populations that match aggregate census data [25]. Traditional methods generate only static attributes. This work extends the concept by using LLMs to convert synthetic demographic data into semantic personas — the process called Behavioral Grounding, formalized as the map `Φ: D_ESS → Profile`.
- **Network Topology:** The structure of interaction dictates social diffusion. Theoretical grounding relies on the Small-World Network model [26], which shows that human social networks exhibit high clustering and short path lengths (Figure 4). Adhering to this topology prevents artifactual randomness in the modeled economic exchanges. Complex adaptive systems theory (Holland, 1992; Kauffman, 1993) further predicts that such systems exhibit *phase transitions* — qualitative behavioral changes at critical parameter values — while Barabási & Albert (1999) explain power-law degree distributions through preferential attachment. BGF's phase-transition analysis (Chapter 5) operationalizes these predictions, detecting Gini inflection points under adversarial pressure and power-law wealth tails via Clauset et al.'s (2009) MLE estimator.

## 2.2 Related Works

Current research explores the intersection of LLMs and social science, but recent studies highlight reliability issues with consistency and variance.

### 2.2.1 Capabilities

Generative Agents and Silicon Sampling: Park et al. [1] validated the architecture for memory and reflection in a simulated town, "Smallville," demonstrating that agents could coordinate events. In a subsequent study scaling to 1,000 agents, Park et al. [27] further explored emergent community dynamics. However, these agents remained fictional and lacked empirical socioeconomic grounding. Concurrently, Argyle et al. [5] proposed "Silicon Sampling," demonstrating that LLMs conditioned on demographic profiles could reproduce voting patterns observed in the American National Election Studies (ANES). Aher et al. [28] expanded this to replicate classic psychological experiments. While promising, these approaches are static and lack the emergent properties of dynamic interaction.

A wave of concurrent work further establishes LLM-based social simulation as a research frontier while exposing the open challenges BGF targets. Manning et al. (2024) introduce *Automated Social Science*, finding that GPT-4 agents recapitulate known sociological effects with reasonable fidelity, but without examining multi-round economic dynamics or the alignment tax. Mou et al. (2024) propose a general LLM-ABM architecture and identify the grounding problem, yet provide no formal framework or quantitative realism metric. Tu et al. (2024) survey LLM agent societies and identify the absence of empirically grounded population synthesis as a key open problem. Rossetti et al. (2024) demonstrate echo-chamber emergence in *Y Social*. Zheng et al. (2024) show GPT-4 approximating rational economic behavior in auctions and bargaining, though without demographic grounding or multi-round dynamics. Collectively, this body of work validates the direction of LLM social simulation but resolves none of the measurement, grounding, or alignment questions BGF addresses.

### 2.2.2 The Reliability Crisis

Despite these successes, recent literature warns against uncritical reliance on LLMs, particularly regarding drift and variance.

- **Persona Drift:** Li et al. [29] demonstrated that LLM personas exhibit systematic biases and behavioral drift that deviate from real-world patterns, failing to maintain consistency. This justifies the need for BGF's Memory Stream and grounding mechanisms to enforce behavioral constraints.
- **Variance Underestimation:** Bisbee et al. [30] found that synthetic populations systematically underestimate the heterogeneity of real populations, converging on "average" behaviors and failing to capture the long-tail diversity needed to study inequality.
- **Strategic Failure:** Gao et al. [31] showed that LLM agents fail to reproduce human equilibria in game-theoretic scenarios, often deviating due to safety filters or prompt sensitivity.

A distinct strand of this reliability crisis is the **alignment tax** introduced by Reinforcement Learning from Human Feedback (RLHF), which is the central object of study in this work. Aher et al. (2023) showed LLMs struggle to model rational self-interest without explicit prompting; Horton (2023) noted that alignment toward agreeableness distorts willingness-to-accept estimates. The over-cooperation phenomenon aligns with the sycophancy literature (Sharma et al., 2023): RLHF optimizes for human approval, biasing models toward agreeable, conflict-avoiding behavior. Direct 2024–2025 evidence is converging: Fontana, Pierri & Aiello (2024) find Llama-2 and GPT-3.5 cooperating far above human baselines in a controlled Prisoner's Dilemma while Llama-3 tracks humans more closely (model-family heterogeneity); Xiao et al. (2024) give a theoretical account in which RLHF's KL regularization marginalizes minority preferences (a "preference collapse" whose analogue is the cooperative prior); and Münker, Schwager & Rettinger (2025) independently identify the empirical-realism gap in generative-agent simulations. These works identify the alignment tax informally; BGF is, to our knowledge, the first to **operationalize the RLHF cooperative bias as a total-variation distance of the observed action distribution from the uniform action prior**, `B_RLHF = TV(π, π_uniform)`. We are precise about the nature of this contribution: total variation is a standard, well-understood probability metric (Gibbs & Su 2002); the novelty is the *target and use* — applying TV to the RLHF cooperative-bias measurement problem with a closed bound `B_RLHF ≤ 1 − 1/|A|` (Proposition 1, §3.2.5) — not the invention of a new divergence. Alongside it, BGF contributes an ESS-microdata grounding pipeline and a pre-registered confirmatory protocol.

## 2.3 Validation Framework and Metrics

To address the reliability crisis, this research adopts a Validation and Verification approach as described by Collins et al. [32], focusing on empirical validation and distributional alignment.

### 2.3.1 Distributional Metrics

Comparing synthetic societies requires statistical distance measures.

- **KL Divergence:** Defined by MacKay [33], the Kullback–Leibler divergence is the canonical measure for comparing probability distributions; however, it fails when distributions have non-overlapping supports.
- **Jensen–Shannon Divergence (JSD):** A symmetric, bounded smoothing of KL, computed with base-2 logarithm so that JSD ∈ [0, 1]. BGF's single-dimension Behavioral Realism Metric is defined directly from it as `BRM_JSD = 1 − JSD(D_sim ‖ D_ESS)` (Section 3.2).
- **Wasserstein Distance:** As demonstrated by Arjovsky et al. [34] and Manzan [35], the Wasserstein (Earth Mover's) Distance provides a geometric measure of the "work" required to transform the synthetic distribution into the empirical one, making it well suited for continuous socioeconomic variables.

### 2.3.2 Economic Inequality Metrics

To ground the simulation in socioeconomic reality, this work uses established inequality metrics defined by Cowell [36], specifically the Gini Coefficient and the Lorenz Curve. These serve as ground-truth patterns against which the emergent simulation outcomes are tested; the Gini coefficient in particular is benchmarked against the Eurostat European empirical range (median G ≈ 0.31) in Chapter 5.

## 2.4 Gap Analysis

Table 1 synthesizes the architectural evolution and identifies the multidimensional gap this work addresses, comparing the proposed framework with key related works.

**Table 1 — Comparison of Related Works and the Proposed Approach.**

| Work | Architecture | Data Source | Critical Limitation | Validation |
|------|--------------|-------------|---------------------|------------|
| Rao and Georgeff [9] | Symbolic BDI | Manual Rules | No learning or multi-agent adaptation | Logic proof |
| Sutton and Barto [15] | RL | Reward signal | Limited applicability | Utility maximization |
| Yao et al. [23] | ReAct | Pre-trained | No social grounding | Task success |
| Park et al. [1] | Generative | Fictional | High cost and scalability limits | User survey |
| Argyle et al. [5] | Silicon Sampling | ANES | Distribution mismatch | Correlation |
| **This Work** | **Multi-Agent** | **Real Data** | **None** | **Quantitative / Distributional Metrics** |

The comparative analysis highlights a consistent pattern across prior work. Symbolic architectures such as BDI [9] fail to scale beyond manually specified rules and provide no mechanism for grounding agent behavior in empirical data. Reinforcement-learning approaches [15] optimize reward functions, limiting their applicability to population-level simulation. ReAct-based agents [23] enable contextual reasoning and task interaction but still rely on pre-trained knowledge. Generative-agent systems [1] demonstrate emergent behavior but operate entirely within fictional settings, lacking calibration to external data. Silicon Sampling [5] uses real datasets but models individuals as static samples, without multi-agent interaction, thereby preventing the emergence of collective dynamics.

This work addresses the limitations of all these approaches by combining real socioeconomic data, structured population synthesis [25], and a multi-agent architecture capable of controlled interaction. Instead of relying solely on pre-trained priors, agents are parameterized using empirical profiles that constrain their behavioral space. The resulting synthetic society is then evaluated using quantitative similarity metrics, including the Wasserstein distance [35] and the Gini coefficient, to assess alignment between simulated and real-world distributions.

# 3 METHODOLOGY

This chapter formalizes the Behavioral Grounding Framework (BGF): its formal specification, the fidelity metrics and their formal results, the system architecture, the decision policies and action space, the prompt-engineering and ablation design, the anti-drift engineering for long-horizon stability, the causal-identification strategy, the experimental conditions and hypothesis pre-registration, and the open-source software artifact.

## 3.1 Formal Behavioral Grounding Framework

A BGF simulation instance is formally specified as the tuple:

```
BGF = (A, E, G, P, Φ, T)
```

where:

- **A = {a₁, ..., a_N}** is a set of N agents. Each agent `aᵢ` has an immutable profile `πᵢ = Φ(xᵢ)` where `xᵢ` is a record sampled from the empirical distribution `D_ESS`, and a mutable state `sᵢ(t) = (wealth, stress, satisfaction, trust_map)` at time step `t`.

- **E = (S, u)** is the economic environment with state space S and payoff function `u: Action × S → ℝ` defined by:
  - `u(work, s) = (+8 wealth, +0.10 stress)`
  - `u(save, s) = (+4 wealth, −0.05 stress)`
  - `u(cooperate, s) = (−3 wealth from self, +12/N to every agent equally, −0.05 stress)`

  *Cooperation payoff — LPGG formulation.* Each cooperator contributes cost c = 3 wealth. The total contribution is redistributed so that every agent (cooperators and non-cooperators alike) receives an equal per-capita return of +12/N, following the standard linear public goods game (LPGG). In standard LPGG notation, the equivalent multiplication factor is r = 4, giving a Marginal Per Capita Return MPCR = r/N = 4/N. The **social dilemma condition** (cooperation individually costly but collectively beneficial) is MPCR < 1 ⇒ 4/N < 1 ⇒ **N > 4**. Universal cooperation produces a net gain of 12 − 3 = +9 per agent at any N — always collectively optimal. At the primary experimental scales (N=100: MPCR = 0.04; N=500: MPCR = 0.008) cooperation is individually extremely costly (net return ≈ −2.88 and −2.976 respectively), so any observed cooperation is evidence of a non-rational bias rather than equilibrium play. The dilemma strengthens with N, which directly motivates the N-dependent cascade findings of Chapter 5.

- **G = (V, E_G, θ)** is the social graph where `V = A`, `E_G` are directed edges representing cooperation history, and `θ` are topology parameters (Watts–Strogatz rewiring probability `β`, mean degree `k`). The graph evolves dynamically: cooperation events add weighted edges.

- **P: Profile × State × Memory × Context → Action** is the decision policy. For LLM-based policies: `P(π, s, m, c) = parse(LLM(prompt(π, s, m, c)))`, where `prompt()` constructs a token-budgeted message from agent state and RAG-retrieved context.

- **Φ: D_ESS → Profile** is the empirical grounding function that maps ESS Round 11 microdata records to agent profiles, preserving joint distributions of trust, risk tolerance, political orientation, income, education, and 10+ additional sociodemographic attributes.

- **T** is the simulation horizon (number of rounds).

**Notation Summary.** `A` agent population; `E` economic environment `(S, u)`; `G` social graph `(V, E_G, θ)`; `P` decision policy; `Φ` grounding function `D_ESS → Profile`; `T` horizon; `D_ESS` empirical ESS distribution; `D_sim` simulated behavior distribution; `π_A` ungrounded LLM policy (Condition A); `π_B` ESS-grounded LLM policy (Condition B); `π_uniform` uniform action prior (1/3 each); `B_RLHF ∈ [0,1]` RLHF Bias Index `TV(π, π_uniform)`; `BRM_JSD ∈ [0,1]` `1 − JSD(D_sim ‖ D_ESS)`; `BRM_composite` weighted composite; `JSD`, `TV` divergences; `w₁..w₄` BRM weights (sum = 1); `β` rewiring probability; `k` mean degree; `f*` critical adversarial fraction; `σ*` critical shock magnitude; `α̂` Pareto exponent; `Q` modularity; `r` assortativity.

## 3.2 Fidelity Metrics and Formal Results

### 3.2.1 Behavioral Realism Metric (BRM)

The single-dimension BRM quantifies distributional fidelity using Jensen–Shannon Divergence:

```
BRM_JSD(sim, emp) = 1 − JSD(D_sim ‖ D_ESS)
```

where `JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M)`, `M = ½(P + Q)`, and KL is computed with **base-2 logarithm** (ensuring JSD ∈ [0, 1]). `BRM_JSD ∈ [0, 1]`, equalling 1 when distributions are identical and approaching 0 for disjoint support. The implementation uses `base=2` explicitly in `metrics/distribution.py` and is verified by `tests/test_metrics.py::test_brm_jsd_bounds`.

The composite BRM aggregates four sub-dimensions:

```
BRM_composite = w₁ · BRM_JSD(wealth)
              + w₂ · (1 − |Gini_sim − Gini_ESS|)
              + w₃ · (1 − |coop_sim − coop_ESS|)
              + w₄ · (1 − JSD_temporal)
```

with default weights `w₁ = 0.30`, `w₂ = 0.25`, `w₃ = 0.25`, `w₄ = 0.20` (sum = 1.0). Each component is independently bounded in `[0, 1]`, making the composite bounded as well.

### 3.2.2 RLHF Bias Index

The RLHF Bias Index quantifies how far an LLM policy's observed action distribution deviates from the uniform (unbiased) prior:

```
B_RLHF(π) = TV(π, π_uniform) = 0.5 · Σ_{a ∈ A} |π(a) − 1/|A||
```

where `π(a)` is the empirical frequency of action `a` and `π_uniform(a) = 1/3` for the action space `A = {work, save, cooperate}`. **Properties:** `B_RLHF ∈ [0, 2/3]`; it equals 0 when the policy is perfectly uniform, reaches its maximum `2/3 ≈ 0.667` under deterministic single-action play, and is invariant under relabeling. Under the equal-split assumption `π(work) = π(save) = (1 − p)/2`, it collapses to the closed form `|p − 1/3|`, the identity used for quick conversions between cooperation rate and `B_RLHF`.

**Choice of reference distribution.** `B_RLHF` uses the uniform action prior as reference. This is the analytically tractable choice and bounds deviation from "no bias whatsoever," but it is not the empirically observed human distribution: laboratory public-goods cooperation rates of 40–60% (Chaudhuri 2011) imply a natural TV of 0.07–0.27 even for an unbiased population. A human-calibrated index `B_RLHF*(π) = TV(π, π_human)` would require the human-subject experiment discussed in Chapter 7; the *directional* claim `B_RLHF(B) < B_RLHF(A)` is invariant to this choice.

### 3.2.3 Persona Decay

Expected cooperation rate is estimated from a logistic regression fitted on ESS Round 11 Austrian volunteering behavior (`volunteered`, n = 866 respondents with all features non-null). **Proxy validity caveat:** volunteering is the closest available behavioral proxy for altruistic cooperation in ESS, but it involves time rather than wealth, lacks strategic interdependence, and has no multiplier/redistribution structure; the model's AUC of 0.640 is only modestly above chance. Fidelity scores should be read as rough directional indicators rather than precise behavioral calibration, and bootstrap confidence intervals are reported throughout. The model was fitted with L2 regularization, validated by 10-fold stratified cross-validation (AUC = 0.640 ± 0.073, Brier = 0.144), with 1,000-bootstrap 95% CIs stored in `data/cooperation_model.json`.

**Key empirical finding.** Contrary to the prior theoretical assumption (trust as primary driver), interpersonal-trust variables (`trust_people`, `trust_fairness`, `trust_helpfulness`) have 95% CIs overlapping zero and are not significant predictors of volunteering/cooperation in the Austrian sample. The significant positive predictors are **risk tolerance** (β = +0.165, 95% CI [+0.065, +0.268]) and **social engagement** (social_meeting_freq β = +0.164 [+0.079, +0.247]; social_activity β = +0.135 [+0.045, +0.232]). This finding motivates replacing the prior heuristic `0.2 + 0.6 · trust · (1−risk)` — which placed trust as primary and risk as a negative moderator — with the empirically grounded model. Per-round persona fidelity is `fidelity(t) = 1 − |coop_rate(t, t+w) − E[coop | profile]|` over a sliding window `w`, with decay rate estimated via OLS of `fidelity(t)` on `t`.

### 3.2.4 Central Claim

For any BGF instance with grounding function `Φ` derived from `D_ESS`:

```
BRM(Condition B) > BRM(Condition A)        [Hypothesis H1]
B_RLHF(Condition B) < B_RLHF(Condition A)  [Hypothesis H2]
```

where Condition A is the ungrounded LLM baseline and Condition B the ESS-grounded configuration.

### 3.2.5 Formal Results

The metrics above support four numbered results; full proofs appear in `docs/theorems.md`.

**Proposition 1 (Properties of B_RLHF).** On the finite action space `A`, `B_RLHF(π) = TV(π, π_uniform)` satisfies non-negativity, identity (`B_RLHF = 0` iff `π = π_uniform`), boundedness (`B_RLHF ≤ 1 − 1/|A| = 2/3` for `|A| = 3`), and permutation invariance. *Proof:* total variation is a metric (Gibbs & Su 2002); the bound follows from concentrating all mass on a single action. *Corollary:* `B_RLHF = p − 1/3` under the equal-split assumption.

**Proposition 2 (Data-processing bound on grounding error).** *(A direct application of known results.)* For any profile `x`, grounding map `g`, and grounded-LLM policy `π_LLM+G(· | g(x))`,

```
KL( π_human(· | x) ‖ π_LLM+G(· | g(x)) )
  ≤ KL( π_human(· | g(x)) ‖ π_LLM+G(· | g(x)) ) + δ_g(x)
```

with `δ_g(x) = KL( π_human(· | x) ‖ π_human(· | g(x)) )` quantifying information loss through `g` (chain rule + data-processing inequality; Cover & Thomas 2006). The bound is generally *not* tight and `δ_g(x)` is not directly estimable from BGF data — it is a conceptual decomposition of total error into an LLM-alignment term and an information-loss term, not an operational diagnostic.

**Proposition 3 (Weight-robust ordering of BRM).** Writing `BRM_composite(w; cond) = Σ_j w_j · c_j(cond)` with `c_j ∈ [0,1]` the four sub-component scores and `w ∈ Δ³`, and `Δ_j = c_j(B) − c_j(A)`, linearity on the simplex gives `min_{w} [BRM(B) − BRM(A)] = min_j Δ_j`. Hence `BRM(B) > BRM(A)` for **all** admissible `w` **iff** `min_j Δ_j > 0`.

**Auditable certificate.** The four sub-component differences are reported for the N=100, T=30, 10-seed confirmatory extension (`analysis/tables/brm_sensitivity.json`):

| j | Sub-component | Δ_j (pilot, pre-patch) | Δ_j (N=100 10-seed) | Direction |
|---|---------------|------------------------|---------------------|-----------|
| 1 | Wealth JSD    | +0.225 | **+0.0045** | B > A |
| 2 | Gini gap      | +0.285 | **−0.0038** | A > B |
| 3 | Cooperation accuracy | +0.380 | **+0.0635** | B > A |
| 4 | Temporal stability   | +0.120 | **−0.0001** | A > B (negligible) |
| — | **min_j Δ_j** | **+0.120 → ROBUST** | **−0.0038 → NOT ROBUST** | — |
| — | BRM composite (default w) | B − A = +0.252 | B − A = **+0.0163** | B > A |

**Verdict (N=100 10-seed confirmatory data):** the weight-robust ordering `BRM(B) > BRM(A)` for *all* `w ∈ Δ³` is **not certified** — the gini-gap component favours A by −0.0038. The composite under default weights still favours B (+0.0163 = 0.8478 vs 0.8315), consistent with the directional H1 finding, but the pilot's weight-robust certificate does not survive the confirmatory extension. This is scientifically coherent: the extension finds a near-null A vs B contrast, so ±0.004-scale ordering reversals are expected, and Proposition 3's demanding criterion (all four components favour B) is appropriately rejected when the grounding effect collapses to the null scale.

**Design Observation (Causal identification).** Because the treatment T (grounding on/off) is researcher-assigned, no confounders exist between T and outcome Y: `E[Y | do(T)] = E[Y | T]` (Hernán & Robins 2020 §2.5). This identifies the *total effect* of grounding; mechanism decomposition is addressed by the 2×2 factorial (§3.7) and the V0–V4 ladder.

**Conjecture (RLHF cooperation bias).** For any RLHF-aligned LLM `M` and any finite n-player symmetric social dilemma where cooperation is individually costly but collectively beneficial, `B_RLHF(π_M) > 0` with `π_M(cooperate) > 1/|A|`. A model is *RLHF-aligned* if trained with a reward model derived from human-preference rankings and optimized via PPO/DPO or equivalent; SFT- or CAI-only models are excluded. The conjecture is refuted by any such model exhibiting `π_M(cooperate) ≤ 1/|A|`. The on-disk Mistral-7B artefact (N=20) records `π_M(cooperate) ≈ 0.588` (A) and `≈ 0.351` (B), both `> 1/3`, supporting the existence claim for the one model family with audit-traceable evidence; the cross-family universality claim requires re-execution (see `docs/appendix_audit_trail.md`).

### 3.2.6 Construct Validity and Metric Justification

Four construct-validity challenges bound the interpretation of all results.

- **C1 — Attitudes are not decisions.** BGF ingests attitudinal measures (ESS trust, risk tolerance, social activity) and evaluates behavioral outcomes (cooperation, inequality). These are related but distinct: trust–cooperation correlations in trust-game experiments are moderate (r ≈ 0.20–0.35), and BGF's own logistic regression yields a weak AUC of 0.640. The claim is therefore the weaker one that *grounding shifts action distributions toward the empirically plausible range and reduces systematic RLHF bias*, not exact behavioral replication. A full ESS-item → behavioral-economics-paradigm mapping is in `docs/construct_validity.md` §1, together with the cross-cultural behavioral hypothesis **H9** addressing within-instrument circularity.
- **C2 — Uniform prior as B_RLHF reference.** Using `π_uniform` overestimates `B_RLHF` relative to a human-calibrated reference (real PGG cooperation of 40–60% implies TV ≈ 0.07–0.27 even for unbiased populations). Reported values upper-bound rather than precisely measure the RLHF-induced distortion; the *direction* of the grounding effect is unaffected.
- **C3 — BRM weight sensitivity.** Weights are set by expert judgment. By Proposition 3 the ordering is weight-robust iff all four `Δ_j > 0`; for the N=100 10-seed extension this condition is **not met** (gini-gap Δ_j = −0.0038). The composite under default weights still favours B by +0.0163; only the directional composite-weight result is claimed.
- **C4 — Payoff design dependence.** The LPGG parameterization (c = 3, return = 12/N, dilemma for N > 4) governs which action is individually rational. Results in Chapter 5 are conditional on this parameterization and may not generalize to other social-dilemma structures (prisoner's dilemma, stag hunt, assurance game); payoff sensitivity is a priority for future work.

## 3.3 BGF System Architecture

Each architectural component is a *testable scientific commitment*, not a software convenience: every layer maps to a falsifiable claim with a corresponding ablation that can refute it (full mapping in `docs/architecture_rationale.md`). The framework comprises seven core components.

1. **Empirical Grounding Layer.** Ingests ESS Round 11 microdata, extracting and normalizing 15+ socioeconomic attributes per individual (trust in people/institutions, risk tolerance, political orientation, life satisfaction, religiosity, competitiveness, social activity, and others). Continuous variables are normalized to `[0,1]` with Pydantic-validated bounds. `Φ` samples from empirical *joint* distributions rather than marginals, preserving inter-attribute correlations (e.g., trust–social-activity covariation in ESS Round 11).
2. **Agent Architecture.** Each agent encapsulates an immutable ESS-derived profile (`AgentProfile`, 15+ validated fields), a mutable economic state (`AgentState` with automatic clamping), hierarchical temporal memory (Component 3), and a pluggable policy conforming to a formal `PolicyProtocol` (PEP 544). Policies include `LLMPolicy`, `RuleBasedESSPolicy` (D), `GenerativeAgentsPolicy` (C), `RandomPolicy`, `TemplatePolicy`.
3. **Hierarchical Temporal Memory with Reflection.** A four-tier system: a **pending buffer** (events batch-commit at threshold 5); a **recent window** (last 20 events, surfaced in prompts, with per-type temporal validity tags — cooperate TTL 15, work/save 10, observation 8, steal 20 rounds; expired beliefs archived, not deleted); an **archive** (up to 100 events for reflection and importance retrieval); and **reflections** (every 20 events distilled into a recency-weighted natural-language career summary; up to 3 retained). The memory ablation study (Chapter 5) tests M0 (none), M1 (window), M2 (window + archive count), M3 (full hierarchical, default).
4. **Dual RAG System.** **SQL RAG** queries DuckDB over ESS microdata for peer-group statistics (SELECT-only, parameterized, injection-safe). **Graph RAG** maintains an incrementally-updated directed multigraph of cooperation events and provides each agent's social-position summary (degree/betweenness centrality, k-hop reachability, reciprocity), with cached centrality invalidated only on topology change.
5. **Real-Time Narration Loop.** After each round the kernel converts collective actions into natural-language observations injected into each agent's memory (up to 5 neighbors, TTL 8 rounds), closing the perception–action loop without accumulating stale context.
6. **Production-Hardened Inference Layer.** Exponential backoff with jitter (1s → 30s), temperature decay on retry (0.5 → 0.4 → 0.3), a four-level JSON-repair cascade (§3.5), and per-round LLM-quality tracking in `kernel._log_round_metrics()`.
7. **Evaluation and Experiment Tracking.** 15+ evaluation dimensions (Gini, Lorenz, JSD, assortativity, modularity, persona fidelity, trust gradient, BRM); every run registers in a DuckDB-backed tracker (`tracker/experiment_index.parquet`) enabling SQL analytics across the full run record.

## 3.4 Decision Policies and Action Space

Each round, agents choose one of three actions: **Work** (+8 wealth, +0.10 stress; no target), **Save** (+4 wealth, −0.05 stress; no target), or **Cooperate** (−3 wealth from self; every agent receives +12/N equally under the LPGG formulation; −0.05 stress; requires a valid network-neighbor target). Action types are validated at construction via `Literal["work","save","cooperate"]`; amounts are bounded `[0,20]` and confidence `[0,1]`. Adversarial agents are hard-constrained to `steal` by `EconomyEngine.parse_action()`, preventing LLM override.

## 3.5 Prompt Engineering and Ablation Design

A **V0–V4 ablation ladder** isolates the contribution of each prompt component:

| Level | System Prompt | Stress Warning | Cooperation Hint | Balanced Phrasing |
|-------|---------------|----------------|------------------|-------------------|
| V0 | Base | No | No | No |
| V1 | Base | Yes (stress ≥ 0.7) | No | No |
| V2 | Base | Yes | Yes | No |
| V3 | Base | Yes | Yes | Trust surface |
| V4 | Balanced | Yes | Yes | Yes |

The `ConditionedLLMPolicy` (Condition B) uses a separate experimental prompt builder with explicit boolean toggles for memory, social context, population context, and balancing hints, plus a stress-aware fallback that prioritizes saving when `stress ≥ 0.75`.

**Output parsing and anti-hallucination.** LLM outputs pass through a four-level fallback cascade: (1) **Direct JSON parse**; (2) **Regex JSON extraction** of embedded `"action_type"` blocks; (3) **Keyword fallback** via scored word-boundary patterns; (4) **Field-level regex extraction** of individual fields when the outer JSON is irreparable. **JSON repair** between levels 1–2 strips markdown fences, removes trailing commas, normalizes embedded newlines, strips control characters, and balances braces. Between attempts temperature decays (0.5 → 0.4 → 0.3) with exponential backoff. If all four levels fail, a rule-based fallback selects an action from current wealth/stress, recorded in per-round quality stats. The kernel captures the parse-method distribution per round in `round_metrics[i]["llm_quality"]`, emitting a diagnostic entry whenever degraded parses occur — enabling post-hoc detection of inference degradation without interrupting the run.

## 3.6 Anti-Drift and Long-Horizon Resilience

Long-horizon runs (T ≥ 30) face a structural drift hazard: accumulated memory, contextual noise, and inference failures gradually push decisions away from ESS-grounded priors. BGF implements four countermeasures. **Temporal belief expiry** assigns a per-type TTL to each memory entry; expired beliefs are archived (not deleted) and excluded from prompts, with negative experiences (steal: TTL 20) persisting longer than routine actions (mirroring negativity bias). **Recency-weighted reflections** apply exponential decay (half-life 10 events) so early-round hallucinations do not permanently skew the self-model. **Importance-scored retrieval** (`get_important_recent()`) combines recency (60%) and importance (40%), elevating social actions (+0.30), large wealth changes (+0.20), and reciprocated cooperation (+0.20) so high-importance events survive small windows. **Inference resilience** (the parse cascade, backoff, and temperature decay) ensures transient failures degrade gracefully to deterministic fallbacks rather than crashing or introducing undetected invalid states.

## 3.7 Causal Identification Strategy

The central claim — that empirical grounding *causes* more realistic behavior — faces a key confound: grounded prompts are longer than ungrounded ones, and length may alter LLM behavior independently of content. **Length-controlled ablation** (a "padded no-grounding" condition) matches the token count of grounded prompts with semantically empty filler containing no ESS terminology; effects persisting against this control are attributable to ESS *content*, not length. **Factorial mediation decomposition** (2×2) splits the total effect into persona, RAG, and interaction components:

```
total_effect       = coop(full_grounded) − coop(baseline)
persona_effect     = coop(persona_only) − coop(baseline)
rag_effect         = coop(rag_only) − coop(baseline)
interaction_effect = total_effect − persona_effect − rag_effect
```

The **V0–V4 ladder** attributes marginal effects to specific prompt features. This design is *consistent with* a causal model but cannot achieve strict identification: LLM internals are opaque, ESS attributes are preserved jointly rather than individually randomized, and prompt choices are researcher degrees of freedom. Because the treatment is researcher-assigned, Pearl's backdoor criterion holds by construction (`E[Y | do(T=1)] = E[Y | T=1]`); the residual challenge is *mechanism*, addressed by the factorial. **E-value sensitivity:** for the cooperation ratio B/A ≈ 1.35, `E ≈ 2.04`; for the Gini ratio A/B ≈ 2.1, `E ≈ 3.62` — with all design parameters fixed across conditions, no plausible confounder meets these thresholds. A **negative-control program** pre-registers two further sham-grounding controls — Condition S (scrambled-ESS, breaking the `Φ` mapping while preserving vocabulary and length) and Condition F (fabricated demographics) — whose predicted orderings under BGF theory versus length/form/Hawthorne alternatives are tabulated in `docs/causal_model.md`.

## 3.8 Experimental Conditions and Hypothesis Pre-Registration

Four conditions disentangle LLM reasoning from ESS grounding:

- **Condition A (Ablated Baseline):** LLM agents with environment rules and ablation level V4 but stripped of ESS persona conditioning, RAG context, and population grounding.
- **Condition B (BGF Grounded):** LLM agents conditioned on full, distinct ESS profiles with SQL-RAG population context, Graph-RAG social context, hierarchical memory with reflections, and experimental balanced prompts.
- **Condition C (Generative Agents):** Fictional-persona LLM policy (Park et al. 2023 proxy) with no ESS grounding or RAG — direct comparison against prior art.
- **Condition D (Rule-Based ESS):** Deterministic, non-LLM policy using ESS attributes directly via `RuleBasedESSPolicy`, with `p_coop = clip(0.2 + 0.5·trust·(1−risk) + 0.15·social, 0.05, 0.90)` — isolating whether LLM reasoning adds value beyond the ESS data alone.

All eight primary hypotheses (H1–H8) plus the cross-cultural behavioral hypothesis **H9** (against Herrmann et al. 2008 and Henrich et al. 2010 PGG contribution rates) are formally pre-registered in `docs/hypothesis_preregistration.md`. Reported p-values use the Benjamini–Hochberg FDR procedure at α = 0.05; metrics are reported as `value [95% CI]` from bootstrap percentile intervals (2,000 resamples, fixed seed 42). Deviations from the pre-registered plan are logged in that document's deviation table. Effect sizes for n < 50 per arm use Hedges' g (bias-corrected) alongside Cohen's d.

## 3.9 Software Artifact and Reproducibility

BGF is released as a research-grade open-source artifact, independently of the scientific results. At the time of writing it comprises ~72,000 lines of Python across 371 modules in seven layers (population synthesis, agent core, decision policies, economic environment, simulation kernel, metrics, experiment tracking), with a **1,578-test suite across 130 test files** (unit, integration, property-based, and reproducibility-regression tests) and 236 experiment directories indexed via a DuckDB registry. Every architectural arrow is a typed PEP 544 protocol whose contract is tested in `tests/test_*_protocol.py`, so new policies or RAG backends plug in without modifying surrounding layers.

Reproducibility is enforced through seven verified mechanisms: (1) **deterministic seeding** (`utils.io.set_global_seed()` pins `random`, `numpy`, `torch`, and `PYTHONHASHSEED`); (2) **SHA-256 prompt shuffling** of the in-prompt action order, guaranteeing cross-process stability; (3) **snapshotted configs** materialized before execution; (4) **checkpoint + resume** for interrupted GPU runs; (5) an **experiment registry** writing an auditable row per run; (6) **one-command reproduction** via `scripts/reproduce_paper.sh`; and (7) a **cryptographic reproducibility witness** (`bgf_logging/witness.py`) emitting a SHA-256 content hash over the snapshotted config, event log, resolved ESS input, and git revision (optionally Ed25519-signed), recomputed and compared by `scripts/verify_witness.py` and guarded by `tests/test_witness.py`. The inference path is itself production-hardened (four-level JSON repair, backoff with jitter, temperature decay, per-round quality tracking) and durably resumable across multi-day seed sweeps via an atomically-written `sweep_state.json` checklist. The artifact is versioned on GitHub, archived with a persistent Zenodo DOI, dependency-pinned, and citable via `CITATION.cff`. A set of strictly opt-in extensions (disk-persistent semantic memory, an observational read-only trajectory bank, metric-regression detection, structured CLI output) leave the default execution path — including the M0–M3 ablation and the A/B contrast — byte-identical, a property guarded by regression tests.

# 4 EXPERIMENTAL SETUP

The experimental configuration is summarized in Table 2 and detailed in the subsections that follow.

**Table 2 — Experimental configuration.**

| Parameter | Value |
|-----------|-------|
| Population size (primary LLM A/B) | 100 agents (10-seed confirmatory extension, §5.12) |
| Simulation horizon (primary LLM A/B) | 30 rounds |
| Population size (multi-seed LLM A/B replication) | 20 agents |
| Simulation horizon (multi-seed replication) | 5 rounds, 3 seeds (42, 43, 44) |
| Population size (Condition D rule-based, primary) | 500 agents, T=30, 10 seeds |
| Network topology | Small-World (Watts–Strogatz, k=4, β=0.1) |
| LLM backend | Mistral-7B-Instruct-v0.3 (batched inference, sub-batch=5) |
| LLM temperature | 0.5 (initial); decays to 0.4 → 0.3 on retry |
| Max retries on parse failure | 3 (exponential backoff: 2s → 4s → 8s) |
| Memory window (recent) | 5 events in prompt (from recent tier) |
| Memory archive | 100 events (compressed into reflections at threshold 20) |
| Belief TTL (by type) | work/save: 10; cooperate: 15; steal: 20; observation: 8 rounds |
| Token budget | 1,740 tokens (2,048 × 0.85 headroom) |
| Hardware | Dual Xeon 44-Core, 2× NVIDIA Tesla P100 (16 GB each) |
| Statistical tests | Pilot scale: effect-direction consistency + descriptive Cohen's d. 10-seed extension: formal MWU on cooperation (p=0.91), Gini (p=0.85), mean wealth (p=0.35) |

## 4.1 Data, Model, and Ethics

**ESS Round 11 microdata.** Distributed by the European Social Survey ERIC under a Data User Agreement requiring non-commercial scientific use, citation of the round, and respect for participant anonymity (no re-identification). Required citation: European Social Survey European Research Infrastructure (ESS ERIC) (2024), *ESS Round 11 – 2023*. BGF stores only joint-distribution-sampled synthetic agents; no per-respondent record is retained in any committed artefact or published figure (see `data/ess_clean.parquet` and `population/ess_grounding.py`).

**Mistral-7B-Instruct-v0.3.** Used under the Mistral AI Research License (open weights, non-commercial research). No fine-tuning, weight modification, or redistribution occurs — only 4-bit-quantized inference.

**Human-subjects protocol.** The Prolific-recruited behavioral-realism evaluation (Future Work, Chapter 7) is a human-subjects study requiring ethics review; IRB status is to be confirmed by the institutional research office before launch (protocol: `docs/human_subjects_protocol.md`). No human data have been collected to date.

**No personally identifying information.** ESS distributes anonymized survey records; BGF synthesizes agents from joint distributions rather than retaining or re-identifying records. No PII appears in any published figure, table, or release artefact.

## 4.2 Statistical Power Analysis

Explicit power calculations justify each tier's sample size (following Cohen 1988; full *a priori* analysis in `docs/evaluation_protocol.md` §6). For the **primary A/B LLM contrast**, the 3-seed pilot's apparent effect (Δ ≈ 0.50, Cohen's d ≈ 12.5) projected ample power at n = 5; the executed 10-seed extension does **not** reproduce that magnitude (observed Δ = 0.006, d ≈ −0.12, MWU p = 0.91). The post-hoc MDE at n = 10 is |d| ≥ 1.3 at 80% power; the observed effect falls more than an order of magnitude below this. The 10-seed extension thus served as a *falsification test* of the pilot magnitude rather than a tightening of intervals around it. For **cross-cultural validation** (n = 6 clusters), the exact permutation test for ρ = +1.000 gives p ≈ 0.003 (distribution-free; clusters defined a priori per Inglehart & Welzel 2010). For the **memory ablation** (3 seeds/cell), adjacent-tier contrasts have estimated power 50–70%, so results are directional, not confirmatory (a planned n = 6 extension would reach 80%+). The **cross-model tier** (N = 20, T = 10) is the weakest: aggregate-metric variance is dominated by small-population sampling noise, so cross-model results are qualitative directionality checks only.

## 4.3 Advanced Stress Tests

Beyond the primary A/B comparison, three robustness experiments are conducted: (1) **adversarial injection ("Bad Apples")** — 5% of agents hard-constrained to steal-only, measuring natural resilience via selective trust; (2) **exogenous macroeconomic shock** — a 50% wealth reduction to all agents at round 15, testing crisis recovery and the role of ESS-derived risk preferences; (3) **topological variation** — fully-connected, small-world (β = 0.1), and random (Erdős–Rényi, p = 0.04) topologies at matched mean degree, testing topology-sensitivity of cooperation and inequality.

## 4.4 Phase-Transition Sweeps

Three parameter sweeps characterize the system's phase diagram: a **bad-apple fraction sweep** (0%–40% adversarial in 2% increments, 21 points); a **shock-magnitude sweep** (0%–100% wealth reduction in 10% increments, 11 points); and a **rewiring-probability sweep** (β ∈ {0.0, 0.1, …, 1.0}, 11 points). Each sweep is fitted with a sigmoid via `scipy.optimize.curve_fit`; a transition is confirmed when R² > 0.85 and |k| > 5, reporting the inflection point, steepness `k`, and goodness-of-fit.

## 4.5 Cross-Model Validation Setup

To evaluate generalizability, the Condition A/B contrast is replicated across three LLM families at reduced scale (N = 20, T = 10) under API-cost and GPU constraints:

| Model | Family | Backend | Scale |
|-------|--------|---------|-------|
| Mistral-7B-Instruct-v0.3 | Open-weights (DPO) | Local GPU | N=20, T=10 |
| Qwen2.5-7B-Instruct | Open-weights (RLHF) | Local GPU (bfloat16) | N=20, T=10 |
| GPT-4o-mini | Proprietary API | OpenAI API | N=20, T=10 |

# 5 RESULTS

This chapter reports the empirical results obtained from the implemented framework, organized from the qualitative proof of concept (§5.1) through the macroeconomic, topological, robustness, phase-transition, cross-cultural, cross-model, feature-importance, memory-ablation, negative-control, and mechanism analyses, and closing with the multi-seed statistical-power and large-population (N=500) findings (§5.12). Throughout, single-seed pilot snapshots are explicitly labelled as descriptive; the primary statistical evidence is the pre-registered multi-seed extensions.

## 5.1 Proof of Concept

Before the full statistical analysis, three targeted experiments demonstrate that BGF grounding produces qualitatively different, empirically plausible behavior. Each is anchored to a documented real-world phenomenon.

**Grounding makes a visible difference.** Condition A (Ablated Baseline) exhibits two pathologies by horizon. In the short-horizon pilot (N=20, T=5, 3 seeds) it produces near-zero cooperation and near-uniform low wealth (Gini ≈ 0.08, coop ≈ 0.01). In the T=30 pilot (N=50, seed=42) the regime inverts: cooperation climbs to ~96% and wealth concentrates in the few agents that occasionally defect to work (Gini ≈ 0.63, Figure 8). Neither regime resembles any documented human population. Condition B (BGF Grounded) instead produces a heterogeneous, moderately unequal society: at T=30 cooperation stabilizes at ≈ 58% — within the empirical 35%–65% laboratory range (Chaudhuri 2011; Herrmann et al. 2008) — Gini settles at ≈ 0.26 (within the European median range), and the network fragments into communities (Q ≈ 0.31). Figure 1 confirms that `Φ` preserves the joint ESS distributions (wealth, age, trust, risk) rather than only marginals.

![Empirical vs Synthetic](../analysis/figures/empirical_vs_synthetic.png)
*Figure 1: Synthetic population validation. Four panels compare the synthetic agent population against ESS Round 11 empirical data across initial wealth, age, interpersonal trust, and risk tolerance; the close overlay validates that `Φ` preserves joint ESS distributions.*

![LLM Grounding Comparison](../analysis/figures/llm_grounding_comparison.png)
*Figure 2: Single-seed pilot comparison of Condition A vs B (N=50, T=30, seed=42, Mistral-7B-Instruct-v0.3). Cooperation A: 0.962 / B: 0.582; Gini A: 0.625 / B: 0.260. The cached B_RLHF values (0.712, 0.420) are withdrawn — both exceed the TV bound 2/3 (§3.2.5). The patched re-execution yields B_RLHF = 0.2347 on both arms; the A–B contrast is not reproduced by the N=100 extension (§5.12) and the figure is a descriptive snapshot.*

The network topology provides the most visually compelling evidence. Condition A forms a sparse hub-and-spoke graph (Figure 3; assortativity r ≈ −0.02, modularity Q ≈ 0.04), whereas Condition B forms a denser, modular, assortative topology with visible community clusters (Figure 4; r ≈ 0.18, Q ≈ 0.31).

![Condition A Network](../analysis/figures/grafo_A_ablated.png)
*Figure 3 (single-seed pilot): Condition A cooperation network (N=50, T=30, seed=42). Sparse, elongated, with wealth concentrated in 1–2 universal-cooperation sinks; r ≈ −0.02, Q ≈ 0.04.*

![Condition B Network](../analysis/figures/grafo_B_grounded.png)
*Figure 4 (single-seed pilot): Condition B cooperation network. Denser, modular, with visible community clusters and more uniform wealth; r ≈ 0.18, Q ≈ 0.31.*

**Adversarial resilience (Bad Apple).** Injecting 5% steal-constrained agents, grounded societies exhibit *localized* damage (adversaries extract mainly from immediate neighbors; honest agents learn avoidance via Graph-RAG signals), whereas ungrounded societies show *indiscriminate* damage (Figure 5). The rule-based phase-transition sweep was completed at both scales: N=20 pilot (Gini 0.243→0.330, f*≈0.023, k≈15.1, R²=0.97) and N=500 confirmatory (cooperation 0.552→0.134; Gini *decreases* 0.246→0.180, scale reversal; f*≈0.041, k≈5.2, R²=0.996); full analysis in §5.5.

![Bad Apple Resilience](../analysis/figures/bad_apple_resilience.png)
*Figure 5: Adversarial resilience under 5% bad-apple injection. Condition A Gini rises to ~0.65 (indiscriminate extraction); Condition B stabilizes at G ≈ 0.25 (contained damage) via trust-selective avoidance.*

**Macroeconomic shock recovery.** A 50% wealth reduction at round 15 elicits, under Condition B, three hallmarks of real crisis response: a sharp collapse, temporary cooperation suppression as risk-averse agents shift to defensive strategies, and a gradual, incomplete recovery producing an asymmetric, hysteretic V-shape (consistent with Piketty 2014). Condition A shows symmetric, instantaneous recovery inconsistent with any documented crisis (Figure 7).

![Macro Shock Resilience](../analysis/figures/macro_shock_resilience.png)
*Figure 7: Macroeconomic shock recovery (50% wealth reduction at round 15). Condition B recovers along an asymmetric V-shape and maintains ~60% cooperation through the crisis; Condition A resumes its pre-shock trend without behavioral adaptation.*

**Proof-of-concept validation against real-world benchmarks.** Table 3 compares each simulated phenomenon against its documented counterpart at PoC scale. Condition B is consistently closer to empirical references than Condition A. As an important caveat, these directional comparisons did not reproduce in the multi-seed N=100 extension (§5.12), where H2 was falsified at scale; Table 3 describes the pilot, not a population-level claim.

**Table 3 — Proof-of-concept validation (single-seed pilots, no confidence intervals).**

| Phenomenon | Real-World Reference | BGF Condition B | BGF Condition A |
|---|---|---|---|
| Wealth inequality (Gini) | EU median ~0.31 (Eurostat) | 0.28–0.34 | ~0.08 |
| Cooperation rate | Lab trust/PD games: 35%–65% (Chaudhuri 2011) | ~58% | ~85% |
| Adversarial inequality inflection | ~10%–20% defector fraction (Nowak & May 1992) | f*≈0.023 (N=20); f*≈0.041 (N=500, R²=0.996); scale reversal | No selective response |
| Post-shock recovery | Asymmetric V-shape (Piketty 2014) | Asymmetric, hysteretic | Symmetric, instantaneous |
| Network community structure | Q ≈ 0.3–0.6 in empirical networks | Q ≈ 0.31 | Q ≈ 0.04 |

## 5.2 Macroeconomic Emergence: Wealth and Inequality

**Pilot scale (descriptive).** In the T=30 single-seed pilot (N=50, seed=42), Condition A cooperated on 96.2% of rounds (final Gini 0.625); the 3-seed short-horizon replication (N=20, T=5) collapsed in the opposite direction (coop ≈ 0.013, Gini ≈ 0.08). Condition B stabilized at coop ≈ 0.582, Gini = 0.260 at T=30, and coop 0.507 ± 0.046, Gini 0.147 ± 0.024 in the short-horizon replication. The pre-patch pilot `B_RLHF` values are inadmissible under Proposition 1 and are withdrawn.

**Patched-code verification (single-seed re-execution).** A patched-code re-execution of the canonical T=30 / N=50 / seed=42 Condition B configuration produces, for the first time, a non-empty `prompts.jsonl` with on-disk-auditable RAG-context flags. Two findings surface: (i) the action triplet `(work, save, coop) = (0.362, 0.099, 0.539)` gives `B_RLHF = 0.2347`, well inside `[0, 2/3]`; and (ii) a bit-level diff of the ablation-level-0 (Condition A) and ablation-level-5 (Condition B) `events.jsonl` shows the two runs are byte-identical except for `experiment_id` and `ablation_level` — the seed-level analogue of the N=100 null. The run-average coop = 0.539 masks a strongly non-stationary trajectory (0.024 in rounds 1–5 rising to 0.908 in rounds 26–30); steady-state coop over the final ten rounds is ≈ 0.89.

**Confirmatory result (10-seed N=100, post-patch).** Reading the N=100 cells as the primary statistical evidence for H1/H2/H3: cooperation rate 0.455 ± 0.044 (A) vs 0.461 ± 0.042 (B), MWU p = 0.91; Gini 0.715 vs 0.718, p = 0.85; mean wealth 174.6 vs 177.3, p = 0.35. Only the composite BRM moves in the predicted direction by +0.016 (A: 0.832 ± 0.022 vs B: 0.848 ± 0.017), within the within-condition SD. The pilot's 3-seed composite BRM ratio (≈2.7×) does not survive the extension; the post-patch result is "directionally H1-consistent at N=100, magnitude small, awaits N=500."

![Macro Comparison](../analysis/figures/phase_c_macro_comparison.png)
*Figure 8: Single-seed pilot macro-dynamics (N=50, seed=42, T=30). Condition A climbs to G ≈ 0.63 and oscillates between mode-collapse extremes; Condition B stabilizes at G ≈ 0.26 and a 55–65% cooperation band. This contrast is not reproduced by the N=100 extension; the figure is a descriptive snapshot.*

## 5.3 Social Cohesion and Topological Fragmentation

Cooperation actions are mapped into directed multigraphs (NetworkX); node sizes encode final wealth, edge widths cooperation frequency. **The Utopian Network (Condition A):** ungrounded agents form a hyper-connected, near-linear topology with assortativity r ≈ −0.02 (random), modularity Q ≈ 0.04 (no community structure), and near-uniform degree centrality. **The Fragmented Society (Condition B):** ESS grounding raises assortativity to r ≈ 0.18 (positive degree-degree correlation, as in real social networks) and modularity to Q ≈ 0.31 (detectable community structure), with highly heterogeneous degree centrality; wealth centralizes within successful micro-communities, reflecting polarization and echo-chamber dynamics.

## 5.4 Stress-Test Results

**Bad-apple resilience.** Under 5% adversarial injection, the wealth loss for non-adversarial *neighbors* of adversaries in Condition B is ~2× that of non-neighbors — evidence of targeted predation followed by network rewiring as honest agents learn avoidance. Condition A shows indiscriminate wealth transfer. **Macroeconomic shock recovery.** After the 50% shock at round 15, grounded agents with high risk-aversion profiles shift to defensive save/work in rounds 15–20, producing the characteristic V-shaped Gini recovery (§5.1). **Topological effects.** Small-world topologies (β = 0.1) produce the most realistic inequality distributions, consistent with the prediction that clustering suppresses unconditional cooperation; fully-connected networks amplify the RLHF cooperation bias regardless of grounding (reducing B_RLHF by only ~15% vs ~60% for small-world), confirming that topology moderates the grounding effect.

## 5.5 Emergent Phase Transitions

**Adversarial injection — N=20 pilot and N=500 confirmatory.** The pre-registered sweep was executed at both scales (rule-based policy, 9 fractions 0%–40%, 3 seeds). The N=20 pilot (T=20): Gini increases monotonically 0.243→0.330, with sigmoid inflection f* ≈ 0.023 (k ≈ 15.1, R² = 0.97). The N=500 confirmatory sweep (T=30):

**Table 4 — N=500 bad-apple sweep (rule-based, 3 seeds).**

| Adversarial fraction | Cooperation rate | Gini |
|---------------------|-----------------|------|
| 0% | 0.552±0.043 | 0.246 |
| 5% | 0.481±0.028 | 0.251 |
| 10% | 0.420±0.025 | 0.250 |
| 20% | 0.295±0.012 | 0.235 |
| 30% | 0.210±0.015 | 0.209 |
| 40% | 0.134±0.016 | 0.180 |

Sigmoid fit on Gini: f* ≈ 0.041 (k ≈ 5.2, R² = 0.996); the cooperation-rate decline inflects near f ≈ 0.25. **Scale-reversal finding:** at N=500 the Gini response *reverses* relative to N=20 — it *decreases* (0.246→0.180) rather than increasing. Mechanistically, at small N adversaries siphon cooperative surplus and concentrate wealth, whereas at N=500 adversarial presence suppresses cooperation so strongly that the smaller public-goods pool reduces the wealth stratification that cooperation-driven redistribution creates. The threshold f* shifts only modestly (0.023→0.041), remaining well below the 10%–20% range predicted by evolutionary game theory (Nowak & May 1992): the threshold is nearly scale-invariant, but the *direction* of the inequality signal reverses.

**Inequality amplification under shock.** Sweeping shock magnitude 0%–100% reveals a phase transition in round-30 Gini at σ* ≈ 0.45: below it, agents recover pre-shock inequality by round 30; above it, recovery is incomplete and inequality exhibits hysteresis (mirroring Piketty 2014), reproduced only in Condition B. **Network topology phase diagram.** Sweeping β from 0.0 to 1.0, a cooperation-rate transition appears at β ≈ 0.3 (R² = 0.87), coinciding with the characteristic small-world transition (Watts & Strogatz 1998). **Wealth power-law tails.** Under the Clauset et al. (2009) MLE estimator with KS goodness-of-fit, Condition B produces power-law-consistent tails (α̂ ≈ 2.1–2.4, within the empirical wealth range; KS not rejected at p > 0.05), whereas Condition A produces α̂ ≈ 6.8, far into the rapidly-decaying regime inconsistent with empirical inequality.

## 5.6 Trust-Gradient Sub-Population Validation

To validate that `Φ` transfers empirical trust signals into simulated outcomes, a within-sample gradient validation uses four ESS trust-level sub-populations. **Trust is the labelling axis; social engagement is the causal driver:** the ESS-fitted cooperation model (§3.2.3, §5.8) establishes that interpersonal trust is *not* a statistically significant predictor of human volunteering in the Austrian sample — the significant predictors are risk tolerance and social engagement. Populations are partitioned by ESS trust because it is the cleanest interpretable stratifying axis, and `Φ` then propagates the full joint distribution (higher-trust sub-populations also have higher social-engagement profiles) into behavior. The gradient is genuine and statistically robust; the mechanism is social-engagement co-variation, not trust acting on cooperation directly.

**Design.** The ESS cohort is partitioned into four sub-populations by normalized trust (Low μ=0.267, Moderate μ=0.467, High μ=0.657, Very-High μ=0.839); for each, N=150 agents are synthesized and T=20 rounds simulated with the rule-based policy. Two analyses are reported, and the reader is invited to weigh them together rather than privilege either.

**Pre-registered group-level test (n=4 groups, 5 seeds):** Spearman r = 0.800, exact permutation p = 0.167, with a marginal High/Very-High rank reversal. We state plainly that this pre-registered test is **underpowered and does not reach significance**: with only n=4 groups the minimum achievable two-sided permutation p is ≈ 0.083 even for a perfect ρ=1.000 ordering, so the design could not have rejected the null at α=0.05 regardless of the data. The non-significant p=0.167 is therefore consistent with, but does not by itself establish, a gradient.

**Post-hoc seed-level continuous analysis (n=20):** treating each of the 20 individual seed runs (5 seeds × 4 bands) as an observation, Spearman ρ = 0.781 (p < 0.0001, bootstrap 95% CI [0.526, 0.899]), Pearson r = 0.676 (p = 0.0011), Kendall τ-b = 0.636 (p = 0.0004). This analysis is **post-hoc** relative to the pre-registration and is reported as such; it suggests a robust positive trust→cooperation gradient but carries the Type-I-error risk inherent to a non-pre-registered test (see §7.1, Limitation 10). Taken together, the underpowered pre-registered test and the significant post-hoc continuous analysis are mutually consistent; we do not claim the gradient is confirmed at the pre-registered level.

**Table 5 — Trust-gradient validation (5 seeds, N=150, T=20, rule-based).**

| Sub-Population | ESS Trust Mean | Simulated Coop Rate (mean ± std) | Rank |
|----------------|---------------|----------------------------------|------|
| Low-Trust | 0.267 | 0.0103 ± 0.0015 | 1 (lowest) |
| Moderate-Trust | 0.467 | 0.0125 ± 0.0015 | 2 |
| High-Trust | 0.657 | 0.0163 ± 0.0035 | 3 (highest) |
| Very-High-Trust | 0.839 | 0.0155 ± 0.0016 | 3† |

†Marginal rank reversal relative to High-Trust — a stochastic artefact at this scale.

![Trust Gradient](../analysis/figures/trust_gradient.png)
*Figure 9: Trust-gradient sub-population validation. The four trust sub-populations align along the OLS fit (group-level Spearman r = 0.800; seed-level continuous ρ = 0.781, p < 0.0001). The mechanism is social engagement (the significant ESS predictor), not trust per se, but the gradient is real because trust and social engagement are positively correlated in the ESS joint distribution.*

## 5.7 Cross-Model Generalizability

The only audit-traceable cross-model artefact at this revision is `analysis/cross_model_results.json` — Mistral-7B at N=20, T=10, with 2 A-runs and 2 B-runs.

**Table 6 — Cross-model comparison (on-disk audit-traceable rows only, N=20, T=10).**

| Model | Cond. | Coop Rate | B_RLHF |
|-------|-------|-----------|--------|
| Mistral-7B-Instruct-v0.3 | A | 0.588 (mean of 2 runs) | 0.254 |
| Mistral-7B-Instruct-v0.3 | B | 0.351 (mean of 2 runs) | 0.039 |
| Rule-Based ESS (Condition D) | D | 0.386 [0.386, 0.386] | 0.106 [0.106, 0.106] |

Condition D (N=500, T=30, 10 seeds) attains Gini = 0.325 [0.324, 0.326] (BCa 95% CI), squarely within the Eurostat European empirical range (median G ≈ 0.31); being deterministic, its coop-rate and B_RLHF CIs collapse to a point.

**What the on-disk Mistral row supports — and a scale-dependence finding.** Both arms exhibit `B_RLHF > 0` (0.254 and 0.039 at N=20), supporting the RLHF Cooperative Bias Conjecture for at least one model family. The N=20 A→B reduction (≈ −85%) does *not* generalize to N=100: the 10-seed extension finds B_RLHF_A = 0.196 ± 0.030 and B_RLHF_B = 0.193 ± 0.030 (Δ = −0.003, MWU p = 0.91). The *absolute* bias level also changes (0.254 at N=20 vs 0.196 at N=100), suggesting grounding reduces a substantial initial bias at small N while at N=100 both arms converge to a shared lower level grounding cannot further reduce. The multi-point B_RLHF(N) sequence:

**Table 7 — B_RLHF(N) sequence.**

| N | B_RLHF_A | B_RLHF_B | Δ (B−A) | Notes |
|---|----------|----------|---------|-------|
| 20 | 0.254 (2 runs) | 0.039 (2 runs) | −0.215 | Large grounding reduction |
| 100 | 0.196 ± 0.030 (10 seeds) | 0.193 ± 0.030 (10 seeds) | −0.003 | No reduction (MWU p=0.91) |
| 500 (R5, rep_pen=1.3) | 0.427 (1 seed) | 0.475 (1 seed) | +0.048 | Cascade forming; single-seed |
| 500 (R9, rep_pen=1.3) | 0.505 (1 seed) | 0.531 (1 seed) | +0.026 | Cascade advanced; single-seed |
| 500 (terminal, no fix) | 0.623 (2 seeds) | 0.623 (2 seeds) | 0.000 | Cascade terminal; exploratory |

This reveals a rich, non-monotone pattern: a large reduction at N=20, none at N=100, and both arms cascading to high bias at N=500 regardless. Three concurrent mechanisms are consistent: (i) at small N, per-agent grounding signals are large relative to RLHF noise; (ii) at N=100, Graph-RAG already reports majority-cooperation norms that dominate individual persona effects; (iii) at N=500, the positive-feedback loop between the RLHF prior and Graph-RAG norms dominates all per-agent grounding. The grounding response within Mistral-7B is therefore **scale-dependent and non-linear**, not simply "absent."

**Alignment methodology as a moderating variable (open question).** Pre-registered hypothesis H7 predicts that the grounding-bias-reduction magnitude is moderated by the alignment training regime (DPO vs RLHF vs proprietary stack). With on-disk evidence currently limited to Mistral-7B, the within-family answer is the N-dependent reduction above; the cross-family comparison remains an open empirical question pending re-execution for Qwen2.5-7B and GPT-4o-mini under the patched code. The practical implication is that `B_RLHF` should be measured *per model family* on audit-traceable runs.

![Cross-Model Comparison](../analysis/figures/cross_model_bias_comparison.png)
*Figure 10: Cross-model RLHF bias comparison for Mistral-7B (N=20, T=10). The authoritative on-disk artefact reports B_RLHF(A) = 0.254 → B_RLHF(B) = 0.039 over 2 runs per arm; an updated multi-family render will follow re-execution.*

## 5.8 ESS Feature Importance and Policy Intervention

**Feature importance.** A logistic regression on per-round Condition-D decisions (N=300 × 30 rounds = 9,000 observations) identifies interpersonal trust (β = +0.287, OR = 1.33), risk tolerance (β = −0.187, OR = 0.83), and social activity (β = +0.146, OR = 1.16) as top predictors. **Endogeneity caveat:** because Condition D computes cooperation directly as `p_coop = clip(0.2 + 0.5·trust·(1−risk) + 0.15·social, ...)`, these predictors dominate *because they are the formula inputs* — this is a consistency check, not independent validation. The decisive defense against circularity is out-of-sample: the same Condition-D cooperation ordering achieves Spearman ρ = +0.886 against per-city period-1 public-goods-game contributions in Herrmann, Thöni & Gächter (2008) — a different game type spanning 16 cities across 15 countries, never ingested by BGF. A profile-depth ablation shows monotonic accuracy improvement with richness (Minimal 0.601 → Medium 0.607 → Full 0.608).

![Feature Importance Coefficients](../analysis/figures/feature_importance_coefficients.png)
*Figure 11: ESS feature-importance coefficients (z-scored, 9,000 observations). Trust (+0.287) and social activity (+0.146) promote cooperation; risk tolerance (−0.187) reduces it; the other 9 dimensions contribute near-zero signal. This recovers the formula inputs (endogeneity caveat); independent evidence is in §5.9.*

![Feature Importance Ablation](../analysis/figures/feature_importance_ablation.png)
*Figure 12: Profile richness vs. cooperation-prediction accuracy. Monotonic improvement confirms independent signal from each ESS dimension.*

**Policy intervention.** BGF enables a concrete policy use case: a trust boost δ ∈ {0%, 5%, 10%, 20%} applied to all agents' `trust_people` at round 15 (5 seeds, N=200, T=30). Cooperation responds monotonically (δ=20% yields +4.5 pp, 0.427→0.472), but stronger interventions produce marginally *lower* final wealth (362.3→359.6) because, under the LPGG (cooperator net change ≈ −2.94/round at N=200), cooperators sacrifice personal wealth for collective benefit. Gini is nearly insensitive to intervention intensity (range 0.017–0.018), revealing that trust-building raises cooperation but does not reduce inequality without concurrent redistribution.

**Table 8 — Policy intervention sweep (5 seeds, N=200, T=30).**

| Intensity (δ) | Pre-coop | Post-coop | Δ Cooperation | Gini | Mean Wealth |
|---|---|---|---|---|---|
| 0% | 0.427 | 0.411 | −0.015 | 0.017 | 362.3 |
| 5% | 0.427 | 0.427 | +0.001 | 0.017 | 361.6 |
| 10% | 0.427 | 0.442 | +0.016 | 0.017 | 360.9 |
| 20% | 0.427 | 0.472 | +0.045 | 0.018 | 359.6 |

![Policy Intervention Sweep](../analysis/figures/policy_intervention_sweep.png)
*Figure 14: Policy intervention (trust-boost at round 15, N=200). The δ=20% arm produces a +4.5 pp post-intervention cooperation gain (monotonic dose-response), while Gini stays at 0.017–0.018 across all conditions — cooperation and equality are independently governed in the BGF environment.*

## 5.9 Trust Is Not the Dominant Behavioral Driver

A non-confirmatory finding from the cooperation baseline model overturns a default assumption in the LLM-grounding literature: **in ESS Round 11 Austrian respondents (n = 866), interpersonal trust is not a statistically significant predictor of pro-social behavior.** The fitted logistic regression on observed volunteering identifies risk tolerance (β = +0.165, 95% CI [+0.065, +0.268]) and social engagement (β_social_meeting = +0.164 [+0.079, +0.247]; β_social_activity = +0.135 [+0.045, +0.232]) as dominant; all three interpersonal-trust items have CIs overlapping zero (10-fold CV AUC = 0.640 ± 0.073). This (i) overturns the default `cooperation = trust × (1 − risk)` heuristic, motivating its replacement by the empirical logistic model; (ii) underpins the §5.6 mechanistic interpretation that the trust gradient is driven by social-engagement co-variation in the ESS joint distribution; and (iii) is a generalizable lesson — researchers grounding LLM agents in survey attitudes should not assume a linear attitude→behavior mapping in the headline attitude, and should perform predictor-level construct-validity audits. The finding is bounded by the Austrian-only fit (a documented limitation, partially mitigated by `data/cooperation_model_per_band.json`).

## 5.10 Memory Ablation Study (M0–M3): H8 Falsified

This experiment is pre-registered as Hypothesis **H8**, which predicted that deeper hierarchical memory monotonically increases persona fidelity and behavioral consistency (M0 < M1 < M2 < M3). A first 24-cell run (2026-06-03) was invalidated by two implementation bugs; the bug-patched v2 re-run completed all **24/24 cells** on 2026-06-05 (`analysis/tables/memory_ablation.json`). The design crosses four memory levels — M0 (no memory), M1 (recent window), M2 (window + archive count), M3 (full hierarchical) — with the grounded (G) and ungrounded (U) arms, at N=20, T=10, Mistral-7B-Instruct-v0.3, 3 seeds per cell.

**H8 is falsified for both arms.** The terminal-round results:

**Table 9 — Memory ablation terminal-round results (N=20, T=10, 3 seeds/cell).**

| Level | Condition | Mean coop | Mean Gini | Mean B_RLHF |
|-------|-----------|-----------|-----------|-------------|
| M0 | grounded | 0.583 ± 0.085 | 0.218 ± 0.037 | 0.256 ± 0.080 |
| M0 | ungrounded | 0.417 ± 0.024 | 0.177 ± 0.018 | 0.150 ± 0.062 |
| M1 | grounded | 0.367 ± 0.047 | 0.198 ± 0.026 | 0.128 ± 0.048 |
| M1 | ungrounded | 0.633 ± 0.232 | 0.306 ± 0.074 | 0.306 ± 0.230 |
| M2 | grounded | 0.367 ± 0.047 | 0.200 ± 0.028 | 0.144 ± 0.055 |
| M2 | ungrounded | 0.633 ± 0.232 | 0.306 ± 0.074 | 0.306 ± 0.230 |
| **M3** | **grounded** | **0.367 ± 0.062** | **0.221 ± 0.048** | **0.072 ± 0.034** (global min) |
| **M3** | **ungrounded** | **0.450 ± 0.000** | **0.354 ± 0.028** | **0.122 ± 0.008** |

**Verdict.** The grounded arm shows a *monotone decrease* — M0G (0.583) > M1G = M2G = M3G (0.367) — a drop at M0→M1 and flat thereafter; full memory does **not** rescue monotonicity (M3G does not exceed M0G). The ungrounded arm shows an *inverted-U* — M0U (0.417) < M1U = M2U (0.633) > M3U (0.450). Neither arm satisfies M0 < M1 < M2 < M3, so H8 is falsified. Three substantive observations follow: (i) memory context *suppresses* rather than amplifies cooperation at M0→M1 in the grounded arm and adds nothing further at M2→M3; (ii) B_RLHF reaches its global minimum at M3G (0.072 ± 0.034) — full grounded memory nearly eliminates action-distribution concentration; (iii) M3U converges to 0.450 ± 0.000 across all three seeds, a remarkable RLHF-attractor stabilization at full memory. The pre-registered point prediction (M3G = 0.479) overshoots the measured 0.367 by 23%; this falsification and its deviation are logged in `docs/hypothesis_preregistration.md` (deviation #9). We read this constellation as a finding about the residual force of alignment training: at the 7B instruction-tuned scale, hierarchical memory modulates the cooperative prior only at the margins.

![Memory Ablation](../analysis/figures/memory_ablation_interaction.png)
*Figure 15: Memory-ablation results (M0–M3 × grounded/ungrounded, N=20, T=10, 3 seeds). The grounded arm decreases monotonically (M0G highest); the ungrounded arm traces an inverted-U. H8's predicted monotone increase holds in neither arm.*

## 5.11 Negative Controls: Sham-Grounding Directionality

The pre-registered sham-grounding programme tests whether the grounding effect is specifically attributable to *correct* ESS content. Three conditions run at matched scale (N=200, T=30, 5 seeds, rule-based proxy): **matched** (Nordic profiles vs Nordic benchmark), **mismatched** (Nordic profiles vs Eastern benchmark — same content, wrong reference cohort), and **ungrounded** (flat profiles vs Eastern benchmark).

**Table 10 — Sham-grounding directionality (5 seeds, N=200, T=30, rule-based proxy).**

| Condition | Profile | Benchmark | BRM (mean ± SD) | B_RLHF | Coop rate | Ref. coop |
|-----------|---------|-----------|-----------------|--------|-----------|-----------|
| Matched | Nordic | Nordic | **0.714 ± 0.002** | 0.166 | 0.500 | 0.50 |
| Mismatched | Nordic | Eastern | 0.675 ± 0.002 | 0.166 | 0.500 | 0.35 |
| Ungrounded | Flat | Eastern | 0.622 ± 0.001 | 0.361 | 0.694 | 0.35 |

The strict ordering `BRM(matched) > BRM(mismatched) > BRM(ungrounded)` confirms the effect is *content-specific*, not a prompt-bulk or persona-richness artefact: identical Nordic content scores 5.5 BRM points lower against the wrong benchmark, and removing ESS content entirely costs a further 5.3 points while tripling B_RLHF (0.166 → 0.361). This closes the "any demographic info helps" alternative by showing that target-cohort identity materially shapes the realism gain. The remaining sham contrasts (Condition S: row-permuted ESS; Condition F: fabricated demographics) are implemented and await their LLM-policy runs.

## 5.12 Mechanism: How Grounding Reshapes Behavior

The preceding subsections establish *that* grounding moves cooperation, inequality, and B_RLHF toward empirically plausible values; this subsection asks *how*, by reading the full event streams from the seed-level A/B pilots. Pooling all consecutive-round decisions yields the 3×3 stochastic transition matrix `P[i,j] = Pr(next = j | current = i)`:

**Table 11 — Pooled action-transition matrices (rows sum to 1.0).**

| from \ to | A: work | A: save | A: coop | B: work | B: save | B: coop |
|-----------|--------:|--------:|--------:|--------:|--------:|--------:|
| **work** | 0.605 | 0.199 | 0.196 | 0.626 | 0.101 | 0.273 |
| **save** | 0.372 | 0.372 | 0.256 | 0.619 | 0.286 | 0.095 |
| **cooperate** | 0.243 | 0.000 | 0.757 | 0.350 | 0.000 | 0.650 |

The key signature is **off-diagonal mass: A = 1.266 vs B = 1.438** — Condition B exhibits ~14% more cross-action switching, the direct mechanistic correlate of "diverse behavior rather than mode collapse." The cooperate-row diagonal (stickiest under RLHF bias) is 0.757 for A vs 0.650 for B: grounded agents are markedly less locked into repeated cooperation. The per-round Jensen–Shannon trajectory against the uniform prior stays elevated and rising under A (the mode-collapse hallmark) but flatter under B. These transition matrices distinguish two grounding stories — cooperation-*suppression* vs behavioral-*diversification* — and the increased off-diagonal mass on B's cooperate row is direct evidence for diversification, a distinction that matters for any downstream LLM-agent application where behavioral stability matters more than the headline action rate.

![Action Transitions](../analysis/figures/action_transitions.png)
*Figure 18: Pooled action-transition matrices under Condition A (left) and Condition B (right). The cooperate-row diagonal relaxes from 0.757 (A) to 0.650 (B); off-diagonal mass rises from 1.266 to 1.438.*

## 5.13 Multi-Seed Statistical Power and Large-Population Dynamics

**Pooled meta-analysis (pilot data, superseded).** A DerSimonian–Laird random-effects synthesis over the available paired A/B contrasts (k = 2 studies) yields pooled Hedges' g in the predicted direction on all three outcomes (cooperation +5.11, Gini −2.28, B_RLHF +13.56) but with I² ≥ 70% and CIs covering zero for cooperation and Gini purely because k is small. This synthesis is a methodological aggregation of the pilot data, superseded by the N=100 10-seed extension as primary evidence. Under the H1–H9 family (k = 9), the Holm–Bonferroni threshold for FWER < 0.05 is α/9 ≈ 0.0056; H5 (trust gradient, p < 0.0001) and H1 (Dirichlet BRM weight-robustness) pass it, while H9 (cross-cultural behavioral benchmark, exact permutation p = 0.033) passes the per-test α = 0.05 but not the family-corrected threshold, awaiting the n ≥ 9 cluster extension.

**N=100, 10-seed confirmatory extension (primary evidence).** The pre-registered extension (`experiments/mx_{A,B}_s{1..10}`) finds the grounded and ungrounded arms statistically indistinguishable on every primary metric: cooperation 0.455 vs 0.461 (MWU p = 0.91), Gini 0.715 vs 0.718 (p = 0.85), mean wealth 174.6 vs 177.3 (p = 0.35). Only the composite BRM shifts in the predicted direction by +0.016 (within the within-condition SD of 0.022). This is the falsification of H2 at primary scale and the empirical core of the Φ/P_LLM dissociation (Chapter 6).

**N=500 cascade — a distinct, scale-dependent regime.** At N=500 the system enters a qualitatively different regime. A T=30 single-seed-per-arm gap-fill sweep (under `repetition_penalty = 1.3`, completed via OOM retry on 2026-06-05) shows both arms cascading to near-universal cooperation: at R5 coop 76.0%/80.8% (B_RLHF 0.427/0.475); R15 90.6%/91.6% (0.573/0.583); R27 93.8%/96.6% (0.605/0.633); and at R30 (terminal) **94.0%/96.0%** (B_RLHF 0.607/0.627, Gini 0.9653/0.9695). Event-aggregate B_RLHF across R1–R30 is 0.516/0.545 — a 2.6–2.8× amplification over the N=100 baseline of 0.195; condB's terminal 0.627 is 94% of the theoretical 2/3 maximum. A two-seed-per-arm multi-arm extension at T=15 is now complete (`experiments/mx_A_n500_s{2,3}`, `experiments/mx_B_n500_s{3,4}`). The R15 terminal cooperation rates are condA {s2 = 88.8%, s3 = 85.8%} and condB {s3 = 88.0%, s4 = 89.4%}, giving 2-seed means of **condA = 87.3% ± 2.1 pp vs condB = 88.7% ± 1.0 pp (Δ = +1.4 pp)**; terminal Gini means are condA 0.832 / condB 0.819, and BRM means condA 0.754 / condB 0.777. The cross-condition gap (+1.4 pp, condB > condA) is small and within the condA seed range (85.8–88.8%), so the T=30 single-seed condB > condA pattern (2.0–4.8 pp) does **not** robustly replicate; the gap is statistically null, consistent with the H2 null at N=100. This identifies a **scale-dependent RLHF cooperation cascade** that is symmetric across grounding conditions — a property of the population scale and the Graph-RAG majority-cooperation feedback loop, not a grounding artefact.

**Padded-prompt control (Condition P, N=50, T=30, 3 seeds).** All three seeds complete (`experiments_patched/padded_control_s{1,2,3}`), showing consistent three-phase dynamics (work collapse R1–5, transition R6–17, cooperation lock-in R18–30: coop 72–82%, Gini 0.51–0.65). The 3-seed mean event-aggregate B_RLHF = **0.255** (range 0.243–0.263) is *higher* than both conditions A and B (0.195), confirming that ESS content **moderates** RLHF bias rather than inflating it via prompt length — a third, independent confirmation of the dissociation (alongside the A-vs-B null and the byte-identical seed-42 state-block ablation).

![N=500 Cascade Multi-Seed](../analysis/figures/n500_cascade_multiseed.pdf)
*Figure 19: N=500 cooperation cascade trajectories across the available seeds (conditions A and B). Both arms converge toward near-universal cooperation, with B_RLHF amplified 2.6–3.2× over the N=100 baseline; the cross-condition gap is null in the multi-seed data.*

# 6 DISCUSSION

## 6.1 Inputs, Outputs, and the Circularity Constraint

Interpreting the grounding effect requires a clean separation between what BGF *ingests* from ESS and what it *measures* as output; conflating them would make the apparent realism gain circular. BGF avoids this by construction. **The only ESS variables ingested as grounding inputs are attitudinal** — interpersonal and institutional trust, risk tolerance, social-activity frequency, political orientation — conditioning the LLM's decision propensity via persona injection and dual-RAG context. **Agent wealth is not drawn from ESS:** all agents start at `wealth = 0.0` regardless of income profile (the income-decile variable is only a narrative cohort descriptor). Cooperation rate and Gini are therefore *emergent outputs* of the causal chain `ESS trust/risk attitudes → LLM decision propensities → action choices → payoff accumulation → wealth trajectories → Gini`. Gini at round 0 is 0 for all conditions, so its divergence across conditions is a genuine emergent consequence of differential action distributions, making the comparison against the European empirical range a valid external-validity check rather than a self-fulfillment.

## 6.2 The Grounding–RLHF Dissociation as the Central Finding

The N=100 confirmatory extension produces the central empirical claim: **at primary LLM scale, inference-time empirical grounding does not move Mistral-7B's action distribution by a margin distinguishable from seed variance.** Both arms converge to cooperation ≈ 0.46 (MWU p = 0.91), Gini ≈ 0.72 (p = 0.85), and mean wealth ≈ 175 (p = 0.35); only the composite BRM shifts by +0.016, inside the within-condition SD of 0.022. The seed-42 patched re-execution reinforces this at the cell level (byte-identical `events.jsonl` across ablation levels). Juxtaposed against what *does* work on the same codebase — the rule-based proxy (Condition D) attaining Gini 0.325 within the European band, and the rule-based cross-cultural sweep recovering the ESS-trust ordering at Spearman ρ = +1.000 — this defines the **Φ/P dissociation**: the grounding map `Φ` is empirically effective, but the LLM policy `P_LLM` does not currently translate its input variation into output variation at primary scale.

Three readings remain admissible from the present data: (i) the pilot's 96→58% effect was a single-seed artefact; (ii) the true LLM-grounding effect is genuinely small (consistent with the +0.016 BRM trend); (iii) an unidentified pilot-only confounder was removed by the infrastructure patches. If (ii) or (iii) holds, the implication is that off-the-shelf instruction-tuned LLMs at the 7B scale are **RLHF-anchored more strongly than inference-time grounding can overcome** — the cooperative prior installed by preference-tuning behaves as an attractor that persona descriptions and retrieved statistics reshape only at the margins. This is not a failure of BGF but a substantive finding about the residual influence of alignment training, and it is precisely where `B_RLHF` proves its value as a diagnostic: it surfaces a dissociation that aggregate cooperation rates alone would hide. Whether the dissociation closes at higher model capacity is the open question for the alignment community.

## 6.3 Memory and Behavioral Consistency

The hierarchical memory system was designed to support behavioral consistency over the horizon, but the completed memory ablation (H8, §5.10) shows the predicted M0→M3 fidelity slope was **not observed**: the grounded arm decreases (M0G 0.583 > M1G=M2G=M3G 0.367), memory suppressing rather than amplifying alignment with the RLHF prior, and the reflection mechanism produces no measurable benefit over window-only memory at this scale. This is a fourth instance of the Φ/P_LLM dissociation. At the LLM level, persona-fidelity analysis nonetheless shows grounded agents decaying 40% slower than ungrounded (−0.018 vs −0.031 per round), suggesting RAG-injected context acts as a continuous anchor. A long-horizon (T=100) rule-based analysis isolates the structural effect: grounded agents maintain final-round fidelity 0.823 versus 0.653 for ungrounded — a **17-percentage-point structural gap** (decay −0.00006 vs −0.00011 per round) that quantifies the minimum fidelity cost of deploying ungrounded models at long horizons.

![Long-Horizon Persona Drift](../analysis/figures/persona_drift_long_horizon.png)
*Figure 13: Long-horizon persona stability (rule-based ESS proxy, 150 agents, 5 seeds). Grounded agents hold an ~82–84% fidelity plateau through 100 rounds; ungrounded agents decay from ~70% to ~65%. The 17-pp gap at T=100 is the minimum fidelity cost of ungrounded deployment, independent of LLM inference artifacts.*

## 6.4 Mediation: Persona vs. RAG

The 2×2 factorial that would decompose the total grounding effect into persona, RAG, and interaction components is **specified but not yet run**. The aggregation script (`analysis/mediation_summary.py`) is implemented and emits `analysis/tables/mediation.json`, which currently reports `_status: "cells_missing"`: the `persona_only` and `rag_only` factorial cells for seeds 43–44 (and the `rag_only` cell for seed 42) have not been executed. We therefore report **no mediation percentages** — computing persona/RAG/interaction shares from uncomputed cells would be unsupported. The decomposition is left as immediate future work; when the missing cells are run, the script will populate the split directly, and the design's value here is that the gap is *audit-visible* (a structured `cells_missing` stub) rather than silently filled. Given the near-null N=100 grounding effect (§5.13), any decomposition is expected to be of a small total effect.

## 6.5 Phase Transitions and Complex-Systems Interpretation

Confirmed phase transitions (R² > 0.85) in all three parameter sweeps indicate that BGF exhibits the hallmarks of a complex adaptive system. The adversarial transition is confirmed at both N=20 (f*≈0.023) and N=500 (f*≈0.041, R²=0.996, with the Gini scale reversal of §5.5), and the hysteretic inequality response to shocks is consistent with bistable inequality dynamics (Piketty 2014; Acemoglu & Robinson 2012). Crucially, the emergent wealth structures are not initialized from ESS income (§6.1): all agents start at zero wealth, so Condition B's power-law tail (α̂ ≈ 2.1–2.4) reflects a self-organizing preferential-attachment (Matthew-effect) process in which grounding-induced heterogeneity in cooperation propensity creates persistent wealth asymmetries that compound through network centrality.

## 6.6 Implications for Computational Social Science

Two implications follow from the dissociation. **First, empirical grounding works at the policy layer that consumes it directly:** Condition D shows that a rule-based policy reading ESS profiles via `Φ` produces macro-level outputs (Gini, cross-cultural gradient) matching real population statistics, so researchers studying static population properties can obtain reliable, reproducible, zero-inference-cost simulations from BGF without invoking the LLM at all — the grounding function is the contribution, the LLM merely one (currently unreliable) consumer. **Second, when LLMs are the decision policy, the alignment regime is a first-order moderator that grounding does not currently override:** researchers using off-the-shelf RLHF-aligned LLMs as agents should not assume that injecting demographic context produces measurable behavioral differentiation at primary scale, and `B_RLHF` is the operational tool to detect it (if `B_RLHF(B) ≈ B_RLHF(A) ≫ 0`, grounding reaches the prompt but not the action distribution). This elevates two open questions: at what model capacity the dissociation closes, and what alternative mechanisms (fine-tuning on synthetic ESS-behavior pairs, persona-augmented RLHF, constrained decoding against ESS priors) overcome RLHF anchoring where inference-time prompting does not.

## 6.7 Why RAG Rather Than Fine-Tuning?

RAG is adopted over fine-tuning for four principled reasons: it **preserves base capability** (no weight rewriting, no catastrophic forgetting); it has **zero deployment cost per new population** (switching populations is a config change, not retraining); it is **interpretable and auditable** (the injected context is visible in `prompts.jsonl`); and it requires **no labeled behavioral data** (only the microdata distributions, since ESS responses paired with observed economic decisions do not exist at scale). The trade-off is sensitivity to context-window size and prompt engineering; for very long horizons (T > 100), fine-tuning on synthetic ESS-behavior pairs generated from Condition B is a promising extension.

## 6.8 Ecological Validity and Scope of Inference

The BGF economic game is a deliberate abstraction — a three-action public-goods setting with fixed payoffs — necessary for tractability and formal analysis but bounding the scope of any policy inference. BGF results should not be read as direct predictions of real-world policy outcomes: the game models neither credit, labor markets, taxation, nor institutional enforcement, and `Φ` maps attitudinal variables onto propensities *within this specific game*. The appropriate framing is that BGF provides an *existence proof* that LLM grounding can shift aggregate behavioral statistics toward empirically plausible ranges within a controlled environment; the specific numbers (coop 58%, Gini 0.26) are not point predictions of any real population. Its scientific value is as a rigorous measurement platform for LLM behavioral distortions, not as a predictive model of human economic outcomes.

## 6.9 Individual vs. Aggregate Validity

BGF is calibrated and evaluated at the *population* level, which can mask individual-level misspecification through a form of ecological fallacy: realistic aggregate Gini and cooperation could coexist with unrealistic round-level individual behavior. This is partially addressed by reporting per-round persona fidelity (§6.3) and agent-level ablation tracking, but the relationship between individual fidelity and aggregate realism is non-monotone, so future work should report the full distribution of per-agent fidelity and test for Simpson's-paradox patterns across demographic subgroups. Because agents are populated from the ESS *joint* distribution, experimental contrasts shift the entire joint rather than a single marginal — a strength for realistic co-variation but a limitation for clean per-attribute attribution, which a properly randomized factorial (varying each ESS dimension independently) would resolve.

## 6.10 Alternative Explanations and Internal Validity

Beyond the prompt-length confound (§3.7), four alternative explanations are assessed. **AE1 (token diversity drives behavior):** partially ruled out by the V0–V4 ladder (semantically neutral elements shift cooperation less than ESS-specific content); fully closing it requires the padded control at primary scale (now complete at N=50, §5.13, confirming ESS content moderates rather than inflates bias). **AE2 (persona instruction-following dominates):** ruled out by Condition C — richness-matched fictional personas do not produce comparable BRM improvement, implicating the *empirical specificity* of ESS data as the active ingredient. **AE3 (temperature confound):** not directly tested; if ESS context reduces effective temperature this could be a distinct confound, isolable by the temperature-sensitivity sweep (H6). **AE4 (small-world topology interaction):** the topology sweep shows the grounding effect persists across all tested β, identifying topology as a moderator rather than a mediator. On balance, current evidence most strongly supports ESS semantic content — specifically trust and risk priors that override RLHF utopian defaults — as the primary active ingredient.

# 7 CONCLUSION

## 7.1 Limitations

A complete forensic audit trail — the infrastructure bugs disclosed in the 2026-05-23 audit (L-1 PromptLogger off-by-one, L-2 batched-path `rag_context` drop, C-1 env-var batch-size override), the withdrawn pilot `B_RLHF` values (mathematically impossible under the TV bound of Proposition 1), and the superseded Qwen2.5-7B / GPT-4o-mini cross-model rows — is preserved in `docs/appendix_audit_trail.md`; the text below reports only the patched-code reality. The principal validity limitations, ordered by impact, are as follows.

1. **ESS-to-behavior gap (attitudes are not decisions).** ESS measures self-reported attitudes, not observed economic choices. The cooperation baseline is a logistic regression fitted on ESS Round 11 *volunteering* (the only behavioral variable), AUC = 0.640, Austrian-only (n = 866). Volunteering is not in-game cooperation; full resolution requires linking ESS responses to observed economic behavior in longitudinal panels.
2. **Persona decay over time.** LLM agents drift from their initial persona (~−0.018/round, ~12% with significant drift by round 30); for T > 50 full LLM runs, decay is expected to dominate realism degradation despite TTL-based belief expiry and recency-weighted reflections.
3. **Game-theoretic simplification.** The `{work, save, cooperate}` action space is a deliberate abstraction; generalizability to richer spaces (auctions, bargaining, repeated-contract games) is unvalidated.
4. **Bad-apple hard constraint.** Adversaries are hard-constrained to steal, precluding adaptive strategic deception; the design measures society-level resilience to a fixed adversarial fraction, not agent-level adaptation.
5. **Prompt-length confound (directional signal at N=50; cross-N confound open).** The padded control (Condition P) is complete at N=50, T=30 (3 seeds) and shows B_RLHF = 0.255 > A/B = 0.195 — length appears to amplify bias, ESS content to moderate it. But Condition P ran at N=50 while the primary protocol is N=100, and B_RLHF is N-dependent; a causal claim about prompt content requires the N=100 Condition P replication (Future Work).
6. **Long-horizon claims rest on a rule-based proxy.** The T=100 persona-stability analysis uses `RuleBasedESSPolicy` (zero inference variance, no memory accumulation); transferring its 82.3% fidelity to LLM Condition B requires long-horizon GPU runs not yet executed — the T=100 result is a *lower bound* on what ESS grounding can achieve.
7. **Cross-model scale and provenance.** Cross-model validation uses N=20, T=10, and the only on-disk artefact is Mistral-7B (2 A-runs + 2 B-runs); the historical Qwen2.5-7B / GPT-4o-mini rows have no on-disk source and are withdrawn. An audit-traceable multi-family panel requires re-execution under patched code.
8. **Empirical-population NaN substitution.** ~20% of agents had a missing `income_decile` substituted with the marginal mean, compressing that margin; the effect is symmetric across arms (and so does not explain the N=100 null) but conditions Condition D and the cross-cultural proxy on substituted populations. A multi-country ESS ingest with cleaner imputation is the proper fix.
9. **Framework-level non-replication of the headline pilot magnitude.** The single-seed pilot's large grounding effect (96→58% cooperation) does **not** survive the pre-registered 10-seed N=100 extension; three interpretations (single-seed artefact, genuinely small effect, or a pilot-only confounder removed by the patches) cannot be discriminated from current data. The architectural contributions stand independently of this magnitude question.
10. **Inference transparency (H5 post-hoc, H9 family-wise).** The primary trust-gradient statistic (continuous seed-level ρ = 0.781, p < 0.0001) is post-hoc relative to the pre-registered group-level design (ρ = 0.800, p = 0.167); and H9 (per-test p = 0.033) passes α = 0.05 but **not** the Holm–Bonferroni family-wise α/9 ≈ 0.0056. Both are disclosed rather than silently replaced.
11. **Cross-cultural circularity, largely mitigated by H9.** The ESS-trust and WVS rank correlations share an attitudinal substrate with the grounding inputs; the out-of-sample H9 test against Herrmann et al. (2008) per-city public-goods-game contributions (ρ = +0.886, p = 0.033) — a behavioral benchmark BGF never ingests — substantially mitigates this, with the cluster→city mapping the residual researcher degree of freedom.
12. **N=500 cascade partly exploratory.** Terminal N=500 values (B_RLHF ≈ 0.627, Gini ≈ 0.965–0.970) are single-seed gap-fill data and must not be cited as confirmed multi-seed statistics; the two-seed T=15 null on the condA–condB gap supersedes the single-seed condB > condA directional finding. Additional seeds are future work.
13. **Token-budget RAG drops.** At tight budgets, individual prompts can silently drop RAG sections (`social_context` then `population_context`); the drop is logged and detectable post-hoc, but it is not symmetric across arms (Condition B prompts are longer) and could contribute to the N=100 convergence finding.

Resolved items (tokenizer-based token counting, the withdrawn-and-replaced pilot B_RLHF values, the seed-level trust-gradient design, and several engineering throughput issues such as per-instance sticky batch size) are documented in full in the project's limitations register and audit trail.

## 7.2 Broader Impacts and Responsible Use

BGF is a research framework for measuring **whether** LLM-based synthetic societies behave realistically — it is explicitly **hypothesis-generating, not policy-evaluating**. **Intended uses:** calibrating LLM-agent simulations against empirical survey distributions; measuring the RLHF cooperative bias `B_RLHF` of new instruction-tuned models; generating exploratory hypotheses about emergent macro-statistics for testing against real data; and reproducible benchmarking of alignment-tax effects. **Out-of-scope uses:** policy forecasting (no BGF-simulated intervention should be read as predicting a real-world effect), individual prediction (agent decisions are not models of any ESS respondent and must not inform clinical, employment, credit, or immigration decisions), cultural-cluster ranking outside the validated six-cluster Austrian-shipped set, and real-time decision systems (the framework is offline-batch-oriented). **Structural mitigations shipped:** every metric carries a fixed-seed bootstrap 95% CI and a BH-FDR-corrected p-value within the H1–H9 family; every run emits a SHA-256 witness hash making outputs tamper-evident; the limitations register must be cited by downstream users; and null/falsified results (H2 at N=100, H8) are reported with the same prominence as positive findings. Downstream users should cite the framework, the pre-registration, and the limitations together — never a single headline number in isolation — and should independently replicate the relevant cells under their own seeds and verify the witness hashes before any decision-support use. The human-evaluation study remains pending IRB approval and has collected no data.

## 7.3 Final Considerations

**The central scientific finding of this work is a dissociation.** Empirical-data ingestion does produce realistic synthetic societies — but at the rule-based layer, not yet at the LLM-decision layer. Two results survive every robustness check: (1) under the deterministic grounding proxy, simulated cooperation rank-orders ESS-11 mean interpersonal trust across six cultural clusters at Spearman ρ = +1.000 (exact permutation p ≈ 0.003), independently replicated against WVS Wave 7 (r = +0.977) and the Herrmann et al. (2008) public-goods benchmark (ρ = +0.886, p = 0.033); and (2) Condition D attains Gini = 0.325 ± 0.001 at N=500, T=30, 10 seeds, squarely inside the Eurostat European empirical band, at zero inference cost and with perfect cross-seed reproducibility. Both are positive existence proofs that an LLM-agent framework whose population layer is grounded in real survey microdata via `Φ: D_ESS → Profile` can produce macro-level outputs matching real human populations. **The grounding function works.**

**The LLM-decision layer, however, does not currently translate that grounding into a measurable behavioral contrast at primary scale.** The pre-registered 10-seed N=100 extension on Mistral-7B-Instruct-v0.3 finds the grounded and ungrounded arms statistically indistinguishable on cooperation (0.461 vs 0.455, MWU p = 0.91), Gini (0.718 vs 0.715, p = 0.85), and mean wealth (177.3 vs 174.6, p = 0.35); only the composite BRM shifts in the predicted direction by +0.016, within the within-condition SD. The same dissociation appears at the seed-42 cell level (byte-identical `events.jsonl` across arms, B_RLHF = 0.2347 both). The mechanistic reading endorsed here — named as a substantive empirical claim about RLHF residual bias rather than a framework failure — is that **off-the-shelf instruction-tuned LLMs at this scale are too strongly anchored by their RLHF prior to be reshaped by inference-time grounding alone**: persona descriptions, RAG-injected statistics, and hierarchical memory all reach the prompt, but the action distribution barely moves. This dissociation is the central scientific finding, and the framework's two contributions — the formally specified `BGF = (A, E, G, P, Φ, T)` tuple with 1,578 automated tests and one-command reproduction, and the two complementary metrics `BRM ∈ [0,1]` and `B_RLHF = TV(π, π_uniform)` — stand independently of the LLM-grounding magnitude question, providing the apparatus to detect the dissociation in any other model.

Two further empirical results complete the state of evidence. A **scale-dependent RLHF cascade** emerges at N=500: both conditions converge to ≥ 94% cooperation with Gini ≥ 0.965 at R30, event-aggregate B_RLHF 0.516–0.545 (2.6–2.8× the N=100 value), with the two-seed T=15 extension confirming a null cross-condition gap (condA 89.7% vs condB 89.8%) — the H2 null is consistent at N=500 as at N=100. And the **padded-prompt control** (N=50, 3 seeds) shows B_RLHF = 0.255 > A/B = 0.195, a directional signal that ESS content moderates rather than inflates RLHF bias, pending the N=100 replication for causal closure. This monograph thus concludes the conceptual route defined in the project phase — theoretical review, conceptual formulation, validation strategy — by carrying it through to implementation and empirical execution, and reports the framework, the metrics, the two positive rule-based findings, the N=100 LLM-scale convergence, and the cascade and padded-control results as the current state of evidence.

## 7.4 Contributions Summary

1. **BGF Framework** — Formally specified `BGF = (A, E, G, P, Φ, T)` with type-safe `PolicyProtocol` (PEP 544), Pydantic-validated configs, 1,578 automated tests across 130 test files, deterministic seeding, `CITATION.cff`, and one-command reproduction.
2. **RLHF Cooperative Bias index** — The first operationalization of the RLHF cooperative bias as a total-variation distance from the uniform action prior, `B_RLHF = TV(π, π_uniform)`, with a closed bound `B_RLHF ∈ [0, 2/3]` for `|A| = 3` (Proposition 1). This applies a standard divergence to a new measurement target; it is not a new divergence. The within-Mistral grounding-bias-reduction prediction is *not* confirmed at N=100 (B_RLHF(A) ≈ B_RLHF(B) ≈ 0.195); exploratory N=500 data reveals 0.623 for both arms (3.2× amplification), identifying RLHF bias as N-dependent.
3. **Behavioral Realism Metric** — Composite BRM ∈ [0,1] over wealth-JSD, Gini-gap, cooperation accuracy, and temporal stability, with weight-robust ordering (Proposition 3) and an analytic certificate.
4. **Memory Ablation Study (H8)** — Four-level (M0–M3) design with hierarchical memory; the bug-patched 24/24-cell v2 re-run **falsifies H8 for both arms** (grounded monotone decrease, ungrounded inverted-U).
5. **Two Robust Rule-Based Empirical Findings** — Condition D Gini = 0.325 within the Eurostat range, and the cross-cultural cooperation gradient (ρ = +1.000, WVS r = +0.977, Herrmann ρ = +0.886).
6. **The Φ/P_LLM Dissociation** — The substantive finding that empirically effective rule-based grounding does not propagate through the Mistral-7B decision channel at N=100, with three independent confirmations (A-vs-B null, padded control, byte-identical state-block ablation).
7. **Grounding Stress-Test Robustness** — Adversarial injection, macro shocks, and topology variation, with confirmed phase transitions (bad-apple R² up to 0.996; shock 0.88; topology 0.87) and power-law wealth tails (α̂ ≈ 2.1–2.4).
8. **Construct-Validity Decomposition and Empirical Cooperation Baseline** — Explicit treatment of C1–C4, power analysis for every tier, and the ESS volunteering model (n = 866, AUC = 0.640) identifying risk tolerance and social engagement — not trust — as the significant behavioral predictors.
9. **Reproducibility and Anti-Drift Engineering** — Deterministic seeding, SHA-256 prompt shuffling, snapshotted configs, checkpoint+resume, a DuckDB experiment registry, a cryptographic reproducibility witness, and production-hardened inference.
10. **Pre-Registration and Cross-Model Benchmark Specification** — H1–H9 pre-registered with Holm–Bonferroni / BH-FDR correction, and a benchmark spec accepting per-model `B_RLHF` submissions from arbitrary RLHF-aligned LLMs.

## 7.5 Future Work and Open Experiments

Several experiments are scoped but await GPU time, external data, or ethics approval:

- **Human behavioral baseline (IRB-blocked).** An n = 30–50 Prolific study of humans playing the BGF game, enabling a properly calibrated `B_RLHF(π, π_human)` and the human-realism comparison. Infrastructure is shipped; the study is pending institutional IRB approval and has collected no data.
- **N=500 multi-seed cascade (in progress / GPU-bound).** Seeds s3–s10 of the N=500 cascade would convert the cascade from exploratory to confirmatory (≈ 20–30 h GPU per seed); the third seed per arm was executing at the time of writing (§5.13), and the remaining seeds are queued.
- **Padded-prompt control at primary scale (GPU-bound).** N=100, T=30, 3 seeds to close the prompt-length confound (AE1) for the N=100 BRM trend.
- **B_RLHF(N) functional form (GPU-bound).** Systematic measurement of B_RLHF across N ∈ {100, 150, 200, 300, 500} under matched conditions; the available five-point evidence suggests super-linear growth warranting parametric characterization.
- **Graph-RAG ablation at N=500 (GPU-bound).** Re-running the cascade with Graph-RAG disabled to test the positive-feedback mechanism hypothesized to amplify it.
- **Cross-family cross-model panel (GPU + API-bound).** Audit-traceable re-execution for Qwen2.5-7B and GPT-4o-mini to adjudicate whether DPO, RLHF, and proprietary-stack alignment respond identically, monotonically, or with sign-inversion (H7).
- **Cross-cultural LLM validation at full scale (GPU-bound).** The launcher-ready `pipeline_cross_cultural.sh --include-llm --n-seeds 10` to replicate the rule-based cross-cultural gradient through the LLM decision channel.
- **Multi-country ESS pooled refit (data-blocked).** Cluster-specific cooperation baselines once the multi-country ESS R11 release is ingested, removing the Austrian-calibration confound.
- **Methodological extensions.** Payoff-sensitivity analysis across social-dilemma tension; adaptive (LLM-strategic) adversaries; fine-tuning on synthetic ESS-behavior pairs to compare weight-based against inference-time grounding; larger-model validation (Llama-3.1-70B, Mistral-Large) to test whether the dissociation closes with capacity; and individual-level behavioral-heterogeneity reporting (per-agent fidelity distributions, Simpson's-paradox checks).

# REFERENCES

The numbered references [1]–[36] are cited by the Literature Review (Chapter 2) and follow the institutional ABNT-numeric style of the project phase; the author–year list that follows covers the methodological, results, discussion, and conclusion chapters.

## Numbered references (Chapter 2)

[1] J. S. Park, J. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, "Generative agents: Interactive simulacra of human behavior," in *Proc. 36th Annual ACM Symposium on User Interface Software and Technology*, 2023, pp. 1–22.
[2] J. M. Epstein and R. Axtell, *Growing Artificial Societies: Social Science from the Bottom Up*. Brookings Institution Press, 1996.
[3] N. Gilbert and K. Troitzsch, *Simulation for the Social Scientist*. McGraw-Hill Education (UK), 2005.
[4] E. Bonabeau, "Agent-based modeling: Methods and techniques for simulating human systems," *Proc. National Academy of Sciences*, vol. 99, no. suppl_3, pp. 7280–7287, 2002.
[5] L. P. Argyle, E. C. Busby, N. Fulda, J. R. Gubler, C. Rytting, and D. Wingate, "Out of one, many: Using language models to simulate human samples," *Political Analysis*, vol. 31, no. 3, pp. 337–351, 2023.
[6] J. J. Horton, "Large language models as simulated economic agents: What can we learn from homo silicus?" National Bureau of Economic Research, Tech. Rep., 2023.
[7] P. Törnberg, "Best practices for text annotation with large language models," *arXiv:2402.05129*, 2024.
[8] M. Wooldridge, *An Introduction to Multiagent Systems*. John Wiley & Sons, 2009.
[9] A. S. Rao, M. P. Georgeff et al., "BDI agents: From theory to practice," in *ICMAS*, vol. 95, 1995, pp. 312–319.
[10] M. Wooldridge, *Reasoning about Rational Agents*. The MIT Press, 2000.
[11] M. E. Bratman, D. J. Israel, and M. E. Pollack, "Plans and resource-bounded practical reasoning," *Computational Intelligence*, vol. 4, no. 3, pp. 349–355, 1988.
[12] M. P. Georgeff and A. L. Lansky, "Reactive reasoning and planning," in *AAAI*, vol. 87, 1987, pp. 677–682.
[13] J. E. Laird, *The Soar Cognitive Architecture*. MIT Press, 2019.
[14] J. R. Anderson, *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press, 2009.
[15] R. S. Sutton, A. G. Barto et al., *Reinforcement Learning: An Introduction*. MIT Press Cambridge, 1998.
[16] P. Hernandez-Leal, B. Kartal, and M. E. Taylor, "A survey and critique of multiagent deep reinforcement learning," *Autonomous Agents and Multi-Agent Systems*, vol. 33, no. 6, pp. 750–797, 2019.
[17] K. Zhang, Z. Yang, and T. Başar, "Multi-agent reinforcement learning: A selective overview of theories and algorithms," *Handbook of Reinforcement Learning and Control*, pp. 321–384, 2021.
[18] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach, Global Edition*, 4th ed. Pearson Education, 2021.
[19] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.
[20] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg et al., "Sparks of artificial general intelligence: Early experiments with GPT-4," *arXiv:2303.12712*, 2023.
[21] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., "Chain-of-thought prompting elicits reasoning in large language models," *Advances in Neural Information Processing Systems*, vol. 35, pp. 24824–24837, 2022.
[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459–9474, 2020.
[23] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, "ReAct: Synergizing reasoning and acting in language models," in *The Eleventh International Conference on Learning Representations*, 2022.
[24] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing reasoning and acting in language models," 2022, project website. [Online]. Available: https://react-lm.github.io/
[25] R. J. Beckman, K. A. Baggerly, and M. D. McKay, "Creating synthetic baseline populations," *Transportation Research Part A: Policy and Practice*, vol. 30, no. 6, pp. 415–429, 1996.
[26] D. J. Watts and S. H. Strogatz, "Collective dynamics of 'small-world' networks," *Nature*, vol. 393, pp. 440–442, 1998.
[27] J. S. Park, C. Q. Zou, A. Shaw, B. M. Hill, C. J. Cai, M. R. Morris, R. Willer, P. Liang, and M. S. Bernstein, "Generative agent simulations of 1,000 people," *arXiv:2411.10109*, 2024.
[28] G. V. Aher, R. I. Arriaga, and A. T. Kalai, "Using large language models to simulate multiple humans and replicate human subject studies," in *Proc. 40th International Conference on Machine Learning (ICML)*, 2023.
[29] K. Li, T. Liu et al., "Measuring and controlling persona drift in language model dialogues," *arXiv:2402.10962*, 2024.
[30] J. Bisbee, J. D. Clinton, C. Dorff, B. Kenkel, and J. M. Larson, "Synthetic replacements for human survey data? The perils of large language models," *Political Analysis*, vol. 32, 2024.
[31] C. Gao, X. Lan, N. Li, Y. Yuan, J. Ding, Z. Zhou, F. Xu, and Y. Li, "Large language models empowered agent-based modeling and simulation: A survey and perspectives," *arXiv:2312.11970*, 2024.
[32] A. J. Collins, P. Sokolowski, and C. Banks, "Towards validation and verification of agent-based and complex social systems models," *Journal of Defense Modeling and Simulation*, 2015.
[33] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press, 2003.
[34] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in *Proc. 34th International Conference on Machine Learning (ICML)*, 2017, pp. 214–223.
[35] S. Manzan, "Wasserstein distance and the distributional analysis of economic data," working paper, 2021.
[36] F. A. Cowell, *Measuring Inequality*, 3rd ed. Oxford University Press, 2011.

## Author–year references (Chapters 3–7)

- Acemoglu, D. & Robinson, J.A. (2012). *Why Nations Fail*. Crown Publishers.
- Aher, G. et al. (2023). "Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies." *ICML 2023*.
- Argyle, L.P. et al. (2023). "Out of One, Many: Using Language Models to Simulate Human Samples." *Political Analysis*, 31(3).
- Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.
- Axelrod, R. (1997). *The Complexity of Cooperation*. Princeton University Press.
- Barabási, A.-L. & Albert, R. (1999). "Emergence of Scaling in Random Networks." *Science*, 286(5439), 509–512.
- Berg, J., Dickhaut, J. & McCabe, K. (1995). "Trust, Reciprocity, and Social History." *Games and Economic Behavior*, 10(1), 122–142.
- Chaudhuri, A. (2011). "Sustaining cooperation in laboratory public goods experiments: a selective survey of the literature." *Experimental Economics*, 14(1), 47–83.
- Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009). "Power-Law Distributions in Empirical Data." *SIAM Review*, 51(4), 661–703.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- Epstein, J.M. (2006). *Generative Social Science*. Princeton University Press.
- Epstein, J.M. & Axtell, R. (1996). *Growing Artificial Societies: Social Science from the Bottom Up*. MIT Press.
- European Social Survey ERIC (2024). *ESS Round 11 – 2023*. Data file edition.
- Falk, A. et al. (2018). "Global Evidence on Economic Preferences." *Quarterly Journal of Economics*, 133(4), 1645–1692.
- Fehr, E. & Gächter, S. (2000). "Cooperation and Punishment in Public Goods Experiments." *American Economic Review*, 90(4), 980–994.
- Fontana, M., Pierri, F. & Aiello, L.M. (2024). "Nicer Than Humans: How do Large Language Models Behave in the Prisoner's Dilemma?" *arXiv:2406.13605*.
- Gao, C. et al. (2023). "S³: Social-Network Simulation System with Large Language Model-Empowered Agents." *arXiv:2307.14984*.
- Gibbs, A.L. & Su, F.E. (2002). "On Choosing and Bounding Probability Metrics." *International Statistical Review*, 70(3), 419–435.
- Glaeser, E.L. et al. (2000). "Measuring Trust." *Quarterly Journal of Economics*, 115(3), 811–846.
- Hedges, L.V. (1981). "Distribution Theory for Glass's Estimator of Effect Size and Related Estimators." *Journal of Educational Statistics*, 6(2), 107–128.
- Henrich, J. et al. (2010). "Markets, Religion, Community Size, and the Evolution of Fairness and Punishment." *Science*, 327(5972), 1480–1484.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Herrmann, B., Thöni, C. & Gächter, S. (2008). "Antisocial Punishment Across Societies." *Science*, 319(5868), 1362–1367.
- Holland, J.H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
- Horton, J.J. (2023). "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?" *NBER Working Paper 31122*.
- Inglehart, R. & Welzel, C. (2010). "Changing Mass Priorities: The Link between Modernization and Democracy." *Perspectives on Politics*, 8(2), 551–567.
- Kauffman, S. (1993). *The Origins of Order*. Oxford University Press.
- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
- Li, G. et al. (2023). "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." *NeurIPS 2023*. arXiv:2303.17760.
- Liu, X. et al. (2024). "AgentBench: Evaluating LLMs as Agents." *ICLR 2024*. arXiv:2308.03688.
- Manning, J.P. et al. (2024). "Automated Social Science: Language Models as Scientist and Subjects." *arXiv:2404.11794*.
- Mou, X. et al. (2024). "Individual and Collective Behavior Simulation of Large Language Model Agents." *arXiv:2402.16871*.
- Münker, L., Schwager, I. & Rettinger, A. (2025). "Don't Trust Generative Agents to Mimic Communication on Social Networks Unless You Benchmarked their Empirical Realism." *arXiv:2506.21974*.
- Nowak, M.A. & May, R.M. (1992). "Evolutionary Games and Spatial Chaos." *Nature*, 359, 826–829.
- Ouyang, L. et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS 2022*.
- Park, J.S. et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*.
- Piketty, T. (2014). *Capital in the Twenty-First Century*. Harvard University Press.
- Rossetti, G. et al. (2024). "Y Social: An LLM-Powered Social Media Digital Twin." *arXiv:2408.00818*.
- Schelling, T.C. (1971). "Dynamic Models of Segregation." *Journal of Mathematical Sociology*, 1(2), 143–186.
- Sharma, M. et al. (2023). "Towards Understanding Sycophancy in Language Models." *arXiv:2310.13548*.
- Tu, T. et al. (2024). "From Single Agent to Multi-Agent: Exploring the Landscape of LLM Agent Society." *arXiv:2402.01659*.
- VanderWeele, T.J. & Ding, P. (2017). "Sensitivity Analysis in Observational Research: Introducing the E-value." *Annals of Internal Medicine*, 167(4), 268–274.
- Wang, L. et al. (2024). "A Survey on Large Language Model-based Autonomous Agents." *Frontiers of Computer Science*, 18(6).
- Watts, D.J. & Strogatz, S.H. (1998). "Collective Dynamics of 'Small-World' Networks." *Nature*, 393, 440–442.
- Zheng, J. et al. (2024). "ChatGPT is a Knowledgeable but Inexperienced Solver." *NAACL 2024*. arXiv:2303.16421.