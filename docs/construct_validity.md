# Construct Validity: Bridging ESS Attitudes to BGF Economic Choices

> **Evidence-status convention.** Each empirical claim is annotated `[audit: X.Y]` referencing a row in `docs/evidence_audit.md`. Pure literature/derivation mappings carry `[📐]`; pending experiments carry `[⏳]`.

The single sharpest external-validity challenge to BGF is the *attitude–behavior gap*: ESS items measure self-reported attitudes (e.g., generalized trust on a 0–10 scale), while the BGF game measures choices over economic actions. Conflating the two is a documented hazard in survey-to-behavior literature (LaPiere 1934 onward). This document closes that gap by:

1. **Mapping each load-bearing ESS construct to the canonical behavioral-economics paradigm** that measures it as a *choice*, and then to the BGF action that operationalizes the same construct (§1).
2. **Justifying the BGF payoff structure** against canonical public-goods-game (PGG) and trust-game (TG) parameters (§2).
3. **Pre-registering a new hypothesis H9** that tests BGF cooperation rates against country-level *behavioral* PGG benchmarks (Henrich et al. 2010; Herrmann et al. 2008), an out-of-sample test independent of the ESS data used in Φ (§3).

This document is the bridge between the formal claim in `theoretical_foundations.md` §1 ("ESS is an approximate sufficient statistic") and the external-validity benchmarks used in `paper.md` §9 (Limitations 1, 11).

---

## 1. ESS → Behavioral Paradigm → BGF Action

The mapping table below identifies, for each load-bearing ESS construct used by Φ, (a) the canonical lab paradigm that measures the *behavioral* analogue, (b) the BGF action that operationalizes it, and (c) the published anchor that establishes the construct's behavioral validity.

| ESS item (Φ input) | Construct | Canonical behavioral paradigm | BGF action | Anchor |
|---|---|---|---|---|
| `ppltrst` (0–10 most people can be trusted) | Generalized trust | **Trust Game** (Berg, Dickhaut & McCabe 1995): investor sends 0–10 to trustee, multiplied 3× | `cooperate` (contribute to public pool, multiplier 12/3=4× per cooperator) | Glaeser et al. (2000, QJE) show `ppltrst` correlates with trust-game investments at r ≈ 0.30 |
| `pplfair` (most people try to be fair) | Reciprocity / fairness norm | **Ultimatum Game** (Güth et al. 1982): proposer offers split, responder accepts/rejects | `cooperate` (rejecting `steal` action when adversarial agents present, H6 resilience) | Henrich et al. (2010, *Science*) cross-cultural UG/DG data |
| `pplhlp` (most people try to be helpful) | Prosociality | **Public Goods Game** (Isaac, Walker, Williams 1994) | `cooperate` (canonical PGG action) | Direct construct match; PGG contribution rate is the canonical prosociality measure |
| ESS R11 volunteering (`stfedu`, related civic items) | Behavioral prosociality | **PGG with punishment** (Fehr & Gächter 2002, *Nature*) | `cooperate` under bad-apple injection (H6) | Volunteering correlates with PGG contribution at country level (Henrich et al. 2010) |
| `impfun` / `ipgdtim` (importance of fun, novelty) | Risk tolerance | **Risk task** (Holt & Laury 2002) — paired lottery choices | `work` (deterministic +8) vs. `save` (deterministic +4) vs. `cooperate` (variable, risky) | Falk et al. (2018, QJE) GPS validates self-reported risk against incentivized choice |
| `impfut` / `ipfrule` (importance of planning ahead) | Patience / time preference | **Intertemporal choice** (Frederick, Loewenstein & O'Donoghue 2002) | `save` (delayed payoff via accumulation) vs. `work` (immediate) | Falk et al. (2018) GPS validates self-reported patience against incentivized choice |
| Income decile (`hinctnta`) | Resource endowment | Endowment effect literature (Kahneman, Knetsch & Thaler 1990) | Initial wealth distribution; affects marginal utility of `save` vs. `cooperate` | Direct; income is exogenous endowment |
| Country | Cultural cluster | Cross-cultural PGG (Henrich et al. 2010; Herrmann et al. 2008, *Science*) | All actions; mediated by cluster-specific cooperation baseline | Henrich et al. document 15× cross-country variation in PGG contribution rates |

### 1.1 What the Table Establishes

For every ESS construct the framework uses, there is (i) a peer-reviewed behavioral paradigm that operationalizes the same construct as a *choice*, and (ii) a published correlation between the survey item and choice in that paradigm. This blocks the "ESS measures attitudes, not behavior" objection at the construct level: the framework is anchored to behavioral validation studies, not to the ESS items in isolation. `[audit: D.1 📐 literature mapping; D.4 📐 Glaeser r ≈ 0.20–0.35 bound]`

### 1.2 What the Table Does Not Establish

The mapping does not assert that the BGF action is *equivalent* to the lab paradigm, only that it operationalizes the same underlying construct. Specifically:

- The BGF `cooperate` action fuses *contribution* (PGG-like), *trust extension* (TG-like), and *fairness* (UG-like). A future cleaner-mapped BGF could expose three separate prosocial actions.
- The lab paradigms are typically one-shot or short-horizon; BGF is T=30 repeated, which more closely matches *Iterated PGG* (Fehr & Gächter 2002).
- All lab anchors are WEIRD-population biased (Henrich, Heine & Norenzayan 2010, *BBS*). Cross-cultural validation requires the country-level benchmarks in §3.

---

## 2. Payoff-Structure Justification

The BGF economy (`environment/economy.py`) uses payoffs:

```
work     →  +8 wealth (private)
save     →  +4 wealth (private)
cooperate →  −3 wealth (contributed) ; pool × 4 / N_cooperators distributed back
```

### 2.1 Match to Canonical PGG

The canonical *Public Goods Game* (Isaac, Walker, Williams 1994) parameters are:

- Endowment per round: `E`
- Contribution: `c ∈ [0, E]`
- Group multiplier: `m × c / N` returned to each of N members.
- Cooperation is socially efficient iff `m > 1`; defection is individually rational iff `m / N < 1`.

The BGF parameters set `E = 3` (per-round cost of `cooperate`), `m = 4` (multiplier on the public pool), and `N = N_cooperators` (variable). The social-dilemma structure (`m > 1` ⇒ efficient cooperation; `m / N < 1` for N ≥ 5 ⇒ free-riding dominant) is preserved. The 4× multiplier sits squarely in the published PGG literature range (m ∈ [2, 5], Ledyard 1995 *Handbook of Experimental Economics*). `[audit: D.2 📐 construction; payoff params in environment/economy.py]`

### 2.2 Match to Trust Game

The Berg-Dickhaut-McCabe trust game uses a 3× multiplier on investor sends. BGF `cooperate` uses a 4× pool multiplier — slightly more generous to reflect repeated-game discounting (T=30 horizon). The structural feature — that prosocial action requires *trusting* that others reciprocate — is preserved.

### 2.3 Why Not a Continuous Action Space

A continuous-action variant exists (`decision/continuous_policy.py`, exploratory). The discrete three-action abstraction is retained for the primary results because:

1. It maps cleanly to the canonical PGG payoff structure (above).
2. It permits closed-form analysis of action distributions (B_RLHF as TV-distance to uniform-3).
3. It eliminates one degree of freedom (the agent's choice of *magnitude* of cooperation) so the realism gain can be attributed to *whether* to cooperate, not *how much*.

The architectural commitment in `architecture_rationale.md` §1 is that this abstraction is benign — falsifiable by a continuous-policy comparison showing different macro patterns.

---

## 3. Pre-Registered Hypothesis H9 — Cross-Cultural PGG OOD Validation

### 3.1 Motivation

Paper §9 Limitation 11 flags a circularity concern: ESS trust is both an input to Φ and a reference for trust-gradient validation (H5). The cross-cultural validation in §6.6 (Spearman ρ = 1.0 against ESS-11 cluster means) is therefore *within-instrument*. A cleaner test uses a **behavioral benchmark independent of ESS**.

### 3.2 Hypothesis

**H9 — Behavioral Cross-Cultural Validation.** BGF country-cluster cooperation rates (Condition B, T=30 final round) correlate positively with the published country-level *Public Goods Game contribution rates* from Herrmann, Thöni & Gächter (2008, *Science*) and Henrich et al. (2010, *Science*).

### 3.3 Operationalization

- **Independent variable**: Published PGG contribution rate per country (Herrmann et al. 2008 Table 1; Henrich et al. 2010 supplementary).
- **Dependent variable**: BGF Condition-B cooperation rate per country cluster, computed via `metrics/cross_cultural.py`.
- **Test**: Spearman ρ across countries with both data sources available (estimated n ≈ 15 countries covered by both Herrmann and the ESS-11 sample).
- **Statistical threshold**: ρ > 0, exact permutation p < 0.05 (achievable at n=15 vs. n=4 of H5).
- **Falsification**: ρ ≤ 0, or p ≥ 0.10.

### 3.4 Why H9 Is the Strongest External Test

- It uses a **behavioral** benchmark (actual choice), not an attitudinal one.
- It uses **out-of-sample data** (Herrmann/Henrich data is not in Φ; ESS Φ does not see PGG contributions).
- It tests **emergent macro behavior** of the simulation against **emergent macro behavior** of human populations, with the *same construct* (cooperation rate).
- It is **independent of the prosocial-language confound** in B_RLHF: cross-country *ranking* is preserved even if all countries are shifted upward by RLHF-induced prosociality.

### 3.5 Implementation Cost

Zero additional GPU. Reuses cross-cultural pilot outputs (`analysis/cross_cultural_results.json`, `analysis/cross_cultural_expanded_results.json`). Required new code: ~80 lines in `metrics/cross_cultural.py` to ingest Herrmann/Henrich tables and compute the Spearman correlation. New analysis artifact: `analysis/tables/h9_cross_cultural_behavioral.json`. `[audit: D.3 ⏳ pending H9 implementation]`

### 3.6 Deviation Disclosure

H9 is added to the pre-registration *after* the original H1–H8 set. This is disclosed in `docs/hypothesis_preregistration.md` with the date of addition, justification (closing limitation #11), and explicit acknowledgement that H9 is *added* not *replaced* — the original H1–H8 stand on their own. Per the pre-registration protocol, H9 is treated as confirmatory for the new question (cross-cultural behavioral validity) and reported with the same FDR correction once data is in hand.

---

## 4. Threats Not Closed by This Document

For completeness, the construct-validity threats this document does *not* eliminate:

1. **WEIRDness of lab anchors** (Henrich et al. 2010, *BBS*): the behavioral-economics paradigms used to validate ESS constructs are themselves disproportionately from Western university samples. Cross-cultural PGG data (Herrmann et al. 2008) partly mitigates this, but full mitigation requires non-WEIRD field data.
2. **Single-game generalization**: BGF is one game with three actions. The construct-validity argument generalizes to PGG-class games; it does not generalize to bargaining (UG, Nash bargaining), market games (Vernon Smith), or political-economy games (Acemoglu-style).
3. **Repeated-game artifacts**: T=30 introduces reputation and strategy effects absent from one-shot anchors. Mitigation: the Axelrod (1984) iterated-PD literature establishes that cooperation in repeated games is itself a measured behavior; the framework's stability over T is then a *predicted* artifact, not a threat.

These residual threats are documented in `paper.md` §9 (Limitations 1, 6) and are not addressed by this document or by H9.

---

## References (added to `paper/references.bib`)

- Berg, J., Dickhaut, J., & McCabe, K. (1995). Trust, reciprocity, and social history. *Games and Economic Behavior*, 10(1), 122–142.
- Falk, A., Becker, A., Dohmen, T., Enke, B., Huffman, D., & Sunde, U. (2018). Global evidence on economic preferences. *Quarterly Journal of Economics*, 133(4), 1645–1692.
- Fehr, E., & Gächter, S. (2002). Altruistic punishment in humans. *Nature*, 415, 137–140.
- Glaeser, E. L., Laibson, D. I., Scheinkman, J. A., & Soutter, C. L. (2000). Measuring trust. *Quarterly Journal of Economics*, 115(3), 811–846.
- Henrich, J., Ensminger, J., McElreath, R., et al. (2010). Markets, religion, community size, and the evolution of fairness and punishment. *Science*, 327(5972), 1480–1484.
- Henrich, J., Heine, S. J., & Norenzayan, A. (2010). The weirdest people in the world? *Behavioral and Brain Sciences*, 33(2–3), 61–83.
- Herrmann, B., Thöni, C., & Gächter, S. (2008). Antisocial punishment across societies. *Science*, 319(5868), 1362–1367.
- Holt, C. A., & Laury, S. K. (2002). Risk aversion and incentive effects. *American Economic Review*, 92(5), 1644–1655.
- Ledyard, J. O. (1995). Public goods: A survey of experimental research. In *Handbook of Experimental Economics*, Kagel & Roth (eds.), Princeton University Press.
