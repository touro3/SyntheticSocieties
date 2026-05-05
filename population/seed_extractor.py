from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from population.persona_synthesizer import PersonaRecord


@dataclass
class SeedDocument:
    title: str
    text: str
    domain: str = "general"


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str = "entity"
    stance: str = "neutral"
    attributes: dict[str, Any] = field(default_factory=dict)


class SeedExtractor:
    """Extract population seeds from arbitrary domain text.

    The LLM path asks for a compact JSON schema; the deterministic regex path
    keeps this module usable in CI and in no-backend runs.
    """

    _JSON_OBJECT_PATTERNS = (
        r"\{.*\}",
        r"```(?:json)?\s*(\{.*?\})\s*```",
    )
    _ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b")
    _SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
    _STOP_ENTITIES = {
        "A",
        "An",
        "And",
        "As",
        "At",
        "But",
        "For",
        "From",
        "If",
        "In",
        "It",
        "New",
        "No",
        "Not",
        "Of",
        "On",
        "Or",
        "The",
        "This",
        "To",
        "With",
    }

    def __init__(self, backend: Any | None = None, temperature: float = 0.0):
        self.backend = backend
        self.temperature = temperature

    def extract(self, doc: SeedDocument) -> list[ExtractedEntity]:
        """Extract entities from a seed document.

        Falls back to deterministic regex extraction when no backend is
        configured or when the backend returns unparseable text.
        """
        if self.backend is None:
            return self._fallback_extract(doc)

        try:
            raw = self._call_backend(doc)
            entities = self._parse_llm_entities(raw)
            if entities:
                return entities
        except Exception:
            pass

        return self._fallback_extract(doc)

    def build_graph(self, entities: list[ExtractedEntity]) -> nx.DiGraph:
        """Build a directed knowledge graph from extracted entities."""
        graph = nx.DiGraph()

        for entity in entities:
            attrs = dict(entity.attributes)
            relationships = attrs.pop("relationships", [])
            graph.add_node(
                entity.name,
                name=entity.name,
                entity_type=entity.entity_type,
                stance=entity.stance,
                attributes=attrs,
            )
            for rel in self._normalize_relationships(relationships):
                target = rel["target"]
                if target not in graph:
                    graph.add_node(target, name=target, entity_type="entity", stance="neutral", attributes={})
                graph.add_edge(
                    entity.name,
                    target,
                    relationship=rel.get("relationship", "related_to"),
                    weight=float(rel.get("weight", 1.0)),
                )

        return graph

    def to_persona_records(
        self,
        entities: list[ExtractedEntity],
        n_agents: int,
        rng: random.Random | None = None,
    ) -> list[PersonaRecord]:
        """Convert entities into PersonaRecord objects with ESS-like defaults."""
        rng = rng or random.Random(42)
        source_entities = entities or [
            ExtractedEntity(
                name="Seed participant",
                entity_type="person",
                stance="neutral",
                attributes={},
            )
        ]

        records: list[PersonaRecord] = []
        for i in range(n_agents):
            entity = source_entities[i % len(source_entities)]
            attrs = entity.attributes or {}
            profile = self._sample_profile_defaults(rng, entity, attrs)
            records.append(
                PersonaRecord(
                    agent_id=f"agent_{i}",
                    age=int(attrs.get("age", profile["age"])),
                    income=float(attrs.get("income", profile["income"])),
                    education=str(attrs.get("education", profile["education"])),
                    occupation=str(attrs.get("occupation", profile["occupation"])),
                    location=str(attrs.get("location", profile["location"])),
                    political_preference=str(attrs.get("political_preference", profile["political_preference"])),
                    risk_tolerance=float(attrs.get("risk_tolerance", profile["risk_tolerance"])),
                    social_class=str(attrs.get("social_class", profile["social_class"])),
                    initial_wealth=float(attrs.get("initial_wealth", profile["initial_wealth"])),
                    gender=self._optional_int(attrs.get("gender", profile["gender"])),
                    country=attrs.get("country", profile["country"]),
                    education_level=self._optional_int(attrs.get("education_level", profile["education_level"])),
                    income_decile=self._optional_int(attrs.get("income_decile", profile["income_decile"])),
                    trust_people=self._optional_float(attrs.get("trust_people", profile["trust_people"])),
                    trust_institutions=self._optional_float(
                        attrs.get("trust_institutions", profile["trust_institutions"])
                    ),
                    political_orientation=self._optional_float(
                        attrs.get("political_orientation", profile["political_orientation"])
                    ),
                    life_satisfaction=self._optional_float(
                        attrs.get("life_satisfaction", profile["life_satisfaction"])
                    ),
                    happiness=self._optional_float(attrs.get("happiness", profile["happiness"])),
                    immigration_attitude=self._optional_float(
                        attrs.get("immigration_attitude", profile["immigration_attitude"])
                    ),
                    social_activity=self._optional_float(attrs.get("social_activity", profile["social_activity"])),
                    competitiveness=self._optional_float(attrs.get("competitiveness", profile["competitiveness"])),
                    leadership_preference=self._optional_float(
                        attrs.get("leadership_preference", profile["leadership_preference"])
                    ),
                    health_status=self._optional_float(attrs.get("health_status", profile["health_status"])),
                    religiosity=self._optional_float(attrs.get("religiosity", profile["religiosity"])),
                )
            )
        return records

    def _call_backend(self, doc: SeedDocument) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract social-simulation seed entities from the user document. "
                    "Respond with JSON only: {\"entities\":[{\"name\":str,"
                    "\"entity_type\":str,\"stance\":str,\"attributes\":dict,"
                    "\"relationships\":[{\"target\":str,\"relationship\":str,\"weight\":float}]}]}."
                ),
            },
            {
                "role": "user",
                "content": f"Title: {doc.title}\nDomain: {doc.domain}\n\n{doc.text}",
            },
        ]
        try:
            result = self.backend.generate(messages=messages, temperature=self.temperature)
        except TypeError:
            result = self.backend.generate(messages, self.temperature)
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)

    def _parse_llm_entities(self, raw_text: str) -> list[ExtractedEntity]:
        if not raw_text or not raw_text.strip():
            return []

        candidates = [raw_text.strip()]
        for pattern in self._JSON_OBJECT_PATTERNS:
            for match in re.findall(pattern, raw_text, flags=re.DOTALL):
                candidates.append(match if isinstance(match, str) else match[0])

        for candidate in candidates:
            try:
                data = json.loads(self._repair_json(candidate))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(data, list):
                entities_raw = data
            elif isinstance(data, dict):
                entities_raw = data.get("entities", [])
            else:
                entities_raw = []
            if not isinstance(entities_raw, list):
                continue
            entities = [self._entity_from_raw(item) for item in entities_raw if isinstance(item, dict)]
            entities = [e for e in entities if e is not None]
            if entities:
                return entities
        return []

    def _entity_from_raw(self, item: dict[str, Any]) -> ExtractedEntity | None:
        name = str(item.get("name", "")).strip()
        if not name:
            return None
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        relationships = item.get("relationships")
        if relationships:
            attributes = {**attributes, "relationships": relationships}
        return ExtractedEntity(
            name=name,
            entity_type=str(item.get("entity_type", item.get("type", "entity")) or "entity"),
            stance=str(item.get("stance", "neutral") or "neutral"),
            attributes=attributes,
        )

    def _fallback_extract(self, doc: SeedDocument) -> list[ExtractedEntity]:
        text = f"{doc.title}. {doc.text}"
        names = self._candidate_entity_names(text)
        if not names:
            names = [doc.title.strip() or f"{doc.domain.title()} seed"]

        entities: list[ExtractedEntity] = []
        for name in names[:50]:
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=self._infer_entity_type(name),
                    stance=self._infer_stance_for_name(name, text),
                    attributes={
                        "domain": doc.domain,
                        "source_title": doc.title,
                        "relationships": self._cooccurrence_relationships(name, names, text),
                    },
                )
            )
        return entities

    def _candidate_entity_names(self, text: str) -> list[str]:
        counts: dict[str, int] = {}
        for match in self._ENTITY_RE.findall(text):
            name = re.sub(r"\s+", " ", match).strip(" ,;:()[]")
            name = re.sub(r"^(The|A|An)\s+", "", name)
            name = re.split(
                r"\s+(Announces|Says|Reports|Warns|Supports|Backs|Opposes|Approves|Rejects)\s+",
                name,
                maxsplit=1,
            )[0]
            if not name or name in self._STOP_ENTITIES or len(name) < 3:
                continue
            if name.split()[0] in self._STOP_ENTITIES and len(name.split()) == 1:
                continue
            counts[name] = counts.get(name, 0) + 1
        return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

    def _cooccurrence_relationships(self, name: str, names: list[str], text: str) -> list[dict[str, Any]]:
        rels: list[dict[str, Any]] = []
        for sentence in self._SENTENCE_RE.findall(text):
            if name not in sentence:
                continue
            for other in names:
                if other != name and other in sentence:
                    rels.append({"target": other, "relationship": "co_mentions", "weight": 1.0})
        return rels[:10]

    @staticmethod
    def _normalize_relationships(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        rels: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                rels.append({"target": item, "relationship": "related_to", "weight": 1.0})
            elif isinstance(item, dict) and item.get("target"):
                rels.append(item)
        return rels

    @staticmethod
    def _repair_json(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        text += "}" * max(open_braces, 0)
        text += "]" * max(open_brackets, 0)
        return text

    @staticmethod
    def _infer_entity_type(name: str) -> str:
        lower = name.lower()
        if any(word in lower for word in ("ministry", "government", "agency", "commission", "parliament")):
            return "institution"
        if any(word in lower for word in ("inc", "corp", "company", "bank", "fund", "firm")):
            return "organization"
        if any(word in lower for word in ("policy", "plan", "bill", "program", "proposal")):
            return "policy"
        if len(name.split()) >= 2:
            return "person_or_group"
        return "entity"

    @staticmethod
    def _infer_stance_for_name(name: str, text: str) -> str:
        support = re.compile(r"\b(supports?|backs?|endorses?|favors?|welcomes?)\b", re.I)
        oppose = re.compile(r"\b(opposes?|criticizes?|rejects?|warns?|condemns?)\b", re.I)
        concern = re.compile(r"\b(concerned|worries|risk|uncertain|cautious)\b", re.I)
        for sentence in SeedExtractor._SENTENCE_RE.findall(text):
            if name not in sentence:
                continue
            if support.search(sentence):
                return "support"
            if oppose.search(sentence):
                return "oppose"
            if concern.search(sentence):
                return "concern"
        return "neutral"

    @staticmethod
    def _sample_profile_defaults(
        rng: random.Random,
        entity: ExtractedEntity,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        income_decile = int(attrs.get("income_decile") or rng.randint(3, 8))
        stance = (entity.stance or "neutral").lower()
        political_orientation = {
            "support": rng.uniform(0.35, 0.65),
            "oppose": rng.choice([rng.uniform(0.15, 0.35), rng.uniform(0.65, 0.85)]),
            "concern": rng.uniform(0.25, 0.55),
        }.get(stance, rng.uniform(0.35, 0.65))
        trust_people = rng.uniform(0.35, 0.75)
        risk_tolerance = rng.uniform(0.25, 0.75)
        return {
            "age": rng.randint(25, 65),
            "income": float(income_decile * rng.randint(350, 550)),
            "education": rng.choice(["upper_secondary", "bachelor", "vocational", "college"]),
            "occupation": "worker" if entity.entity_type != "institution" else "public sector worker",
            "location": rng.choice(["big_city", "suburbs", "town", "village"]),
            "political_preference": SeedExtractor._political_label(political_orientation),
            "risk_tolerance": risk_tolerance,
            "social_class": SeedExtractor._social_class(income_decile),
            "initial_wealth": 50.0 + income_decile * 5.0,
            "gender": rng.choice([1, 2]),
            "country": "AT",
            "education_level": rng.choice([2, 3, 4, 6]),
            "income_decile": income_decile,
            "trust_people": trust_people,
            "trust_institutions": rng.uniform(0.30, 0.80),
            "political_orientation": political_orientation,
            "life_satisfaction": rng.uniform(0.35, 0.85),
            "happiness": rng.uniform(0.35, 0.85),
            "immigration_attitude": rng.uniform(0.25, 0.75),
            "social_activity": rng.uniform(0.30, 0.80),
            "competitiveness": rng.uniform(0.25, 0.75),
            "leadership_preference": rng.uniform(0.25, 0.75),
            "health_status": rng.uniform(0.35, 0.85),
            "religiosity": rng.uniform(0.10, 0.75),
        }

    @staticmethod
    def _political_label(value: float) -> str:
        if value < 0.3:
            return "left"
        if value < 0.45:
            return "center_left"
        if value < 0.55:
            return "center"
        if value < 0.7:
            return "center_right"
        return "right"

    @staticmethod
    def _social_class(decile: int) -> str:
        if decile <= 3:
            return "working"
        if decile >= 8:
            return "upper_middle"
        return "middle"

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
