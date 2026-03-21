from __future__ import annotations

import re

from population.society_spec import SocietySpec


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def _extract_population_size(text: str) -> int | None:
    m = re.search(r"(\d+)\s+(agentes|agents|personas|pessoas|people)", text)
    if m:
        return int(m.group(1))
    return None


def _extract_countries(text: str) -> list[str] | None:
    country_map = {
        "austria": "AT",
        "austria ": "AT",
        "at ": "AT",
        "brasil": "BR",
        "brazil": "BR",
        "portugal": "PT",
        "italia": "IT",
        "italy": "IT",
        "franca": "FR",
        "france": "FR",
        "alemanha": "DE",
        "germany": "DE",
        "espanha": "ES",
        "spain": "ES",
    }
    found = []
    padded = f" {text} "
    for key, value in country_map.items():
        if key in padded:
            found.append(value)
    return sorted(set(found)) or None


def parse_society_prompt(prompt: str, target_population_size: int | None = None) -> SocietySpec:
    text = prompt.strip().lower()
    size = target_population_size or _extract_population_size(text) or 50
    notes: list[str] = []

    age_profile = None
    if _contains_any(text, ["envelhecida", "idosa", "older", "elderly", "aging", "ageing"]):
        age_profile = "elderly"
    elif _contains_any(text, ["jovem", "young", "younger"]):
        age_profile = "young"
    elif _contains_any(text, ["mista", "mixed age", "mixed-age"]):
        age_profile = "mixed"

    urbanization = None
    if _contains_any(text, ["urbana", "urban", "metropolitana", "city", "big city"]):
        urbanization = "urban"
    elif _contains_any(text, ["rural", "campo", "interior", "countryside", "village"]):
        urbanization = "rural"
    elif _contains_any(text, ["suburb", "subúrbio", "suburbana"]):
        urbanization = "suburban"

    trust_people_band = None
    if _contains_any(text, ["desconfiada", "baixa confiança interpessoal", "low trust in people", "distrustful"]):
        trust_people_band = "low"
    elif _contains_any(text, ["alta confiança interpessoal", "high trust in people", "confiante"]):
        trust_people_band = "high"

    trust_institutions_band = None
    if _contains_any(text, ["baixa confiança institucional", "low institutional trust", "anti-institucional"]):
        trust_institutions_band = "low"
    elif _contains_any(text, ["alta confiança institucional", "high institutional trust", "pró-institucional"]):
        trust_institutions_band = "high"

    social_activity_band = None
    if _contains_any(text, ["pouca atividade social", "isolada", "isolated", "low social activity"]):
        social_activity_band = "low"
    elif _contains_any(text, ["muito sociável", "alta atividade social", "high social activity", "socially active"]):
        social_activity_band = "high"

    religiosity_band = None
    if _contains_any(text, ["religiosa", "religious", "religiosidade alta"]):
        religiosity_band = "high"
    elif _contains_any(text, ["secular", "não religiosa", "not religious", "low religiosity"]):
        religiosity_band = "low"

    political_orientation_band = None
    if _contains_any(text, ["esquerda", "left-wing", "progressista", "progressive"]):
        political_orientation_band = "left"
    elif _contains_any(text, ["centro-esquerda", "center-left"]):
        political_orientation_band = "center_left"
    elif _contains_any(text, ["centro-direita", "center-right"]):
        political_orientation_band = "center_right"
    elif _contains_any(text, ["direita", "right-wing", "conservadora", "conservative"]):
        political_orientation_band = "right"
    elif _contains_any(text, ["centrista", "moderada", "center", "centrist"]):
        political_orientation_band = "center"

    risk_tolerance_band = None
    if _contains_any(text, ["aversa ao risco", "risk averse", "risk-averse", "baixa tolerância ao risco"]):
        risk_tolerance_band = "low"
    elif _contains_any(text, ["propensa ao risco", "risk seeking", "high risk tolerance"]):
        risk_tolerance_band = "high"

    competitiveness_band = None
    if _contains_any(text, ["competitiva", "competitive", "alta competitividade"]):
        competitiveness_band = "high"
    elif _contains_any(text, ["baixa competitividade", "cooperativa", "low competitiveness"]):
        competitiveness_band = "low"

    countries = _extract_countries(text)
    if countries:
        notes.append(f"Country filter inferred from prompt: {countries}")

    return SocietySpec(
        narrative=prompt,
        countries=countries,
        age_profile=age_profile,
        urbanization=urbanization,
        trust_people_band=trust_people_band,
        trust_institutions_band=trust_institutions_band,
        social_activity_band=social_activity_band,
        religiosity_band=religiosity_band,
        political_orientation_band=political_orientation_band,
        risk_tolerance_band=risk_tolerance_band,
        competitiveness_band=competitiveness_band,
        target_population_size=size,
        notes=notes,
    )
