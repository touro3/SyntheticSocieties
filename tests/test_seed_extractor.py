from population.seed_extractor import ExtractedEntity, SeedDocument, SeedExtractor


def test_seed_extractor_empty_entities_make_persona_records():
    extractor = SeedExtractor(backend=None)
    records = extractor.to_persona_records([], 3)

    assert len(records) == 3
    assert [record.agent_id for record in records] == ["agent_0", "agent_1", "agent_2"]
    assert all(record.age > 0 for record in records)
    assert all(record.initial_wealth > 0 for record in records)


def test_seed_extractor_build_graph_uses_relationships():
    extractor = SeedExtractor(backend=None)
    entities = [
        ExtractedEntity(
            name="Central Bank",
            entity_type="institution",
            stance="concern",
            attributes={"relationships": [{"target": "Households", "relationship": "affects", "weight": 0.8}]},
        )
    ]

    graph = extractor.build_graph(entities)

    assert graph.has_node("Central Bank")
    assert graph.has_edge("Central Bank", "Households")
    assert graph["Central Bank"]["Households"]["relationship"] == "affects"


def test_seed_extractor_regex_fallback_extracts_document_entities():
    doc = SeedDocument(
        title="Energy Ministry Announces Relief Plan",
        text="The Energy Ministry supports Households after a price shock. Consumer Groups warn about delays.",
        domain="policy",
    )

    entities = SeedExtractor(backend=None).extract(doc)
    names = {entity.name for entity in entities}

    assert "Energy Ministry" in names
    assert any(entity.stance in {"support", "concern", "neutral"} for entity in entities)


def test_seed_extractor_round_trips_to_json(tmp_path):
    """save() followed by load() must reproduce the same entities."""
    extractor = SeedExtractor(backend=None)
    original = [
        ExtractedEntity(
            name="Central Bank",
            entity_type="institution",
            stance="concern",
            attributes={"income_decile": 7},
        ),
        ExtractedEntity(name="Households", entity_type="person_or_group", stance="neutral", attributes={}),
    ]

    cache_path = str(tmp_path / "entities.json")
    extractor.save(cache_path, original)
    loaded = SeedExtractor.load(cache_path)

    assert len(loaded) == len(original)
    assert loaded[0].name == "Central Bank"
    assert loaded[0].entity_type == "institution"
    assert loaded[0].stance == "concern"
    assert loaded[0].attributes.get("income_decile") == 7
    assert loaded[1].name == "Households"
