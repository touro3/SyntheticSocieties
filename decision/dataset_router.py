from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "das", "dos", "um", "uma", "para", "por",
    "com", "sem", "que", "quer", "sociedade", "dataset", "dados", "data",
    "the", "and", "for", "with", "from", "this", "that", "social", "research",
}


@dataclass
class RoutedDataset:
    dataset_id: str
    score: float
    matched_terms: list[str]
    title: str
    trust_level: str
    auditable: bool
    local_path: str


class DatasetRegistry:
    def __init__(self, registry_path: str | Path = "data/dataset_registry.json") -> None:
        self.registry_path = Path(registry_path)
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.registry_version = payload.get("registry_version", "unknown")
        self.datasets = payload["datasets"]

    def _registry_hash(self) -> str:
        return hashlib.sha256(self.registry_path.read_bytes()).hexdigest()

    def get(self, dataset_id: str) -> dict[str, Any]:
        for dataset in self.datasets:
            if dataset["id"] == dataset_id:
                return dataset
        raise KeyError(f"Dataset not found: {dataset_id}")

    def list(self) -> list[dict[str, Any]]:
        return list(self.datasets)

    def route(self, query: str, top_k: int = 3) -> list[RoutedDataset]:
        query_terms = self._tokenize(query)
        routed: list[RoutedDataset] = []

        for dataset in self.datasets:
            field_terms = self._dataset_terms(dataset)
            matched = sorted(query_terms.intersection(field_terms))
            if not matched:
                score = 0.0
            else:
                keyword_terms = set(dataset.get("semantic_keywords", []))
                weighted = 0.0
                for term in matched:
                    if term in keyword_terms:
                        weighted += 2.0
                    else:
                        weighted += 1.0
                score = weighted / max(1.0, len(query_terms))

            routed.append(
                RoutedDataset(
                    dataset_id=dataset["id"],
                    score=score,
                    matched_terms=matched,
                    title=dataset["title"],
                    trust_level=dataset.get("trust_level", "unknown"),
                    auditable=bool(dataset.get("auditable", False)),
                    local_path=dataset["local_path"],
                )
            )

        routed.sort(key=lambda x: x.score, reverse=True)
        return routed[:top_k]

    def audit_selection(self, query: str, routed: list[RoutedDataset], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "registry_path": str(self.registry_path),
            "registry_sha256": self._registry_hash(),
            "registry_version": self.registry_version,
            "selection_method": "auditable_semantic_registry_rag",
            "ranked_results": [
                {
                    "dataset_id": item.dataset_id,
                    "title": item.title,
                    "score": item.score,
                    "matched_terms": item.matched_terms,
                    "trust_level": item.trust_level,
                    "auditable": item.auditable,
                    "local_path": item.local_path,
                }
                for item in routed
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def build_rag_context(self, dataset_ids: list[str]) -> str:
        lines = []
        for dataset_id in dataset_ids:
            ds = self.get(dataset_id)
            lines.append(
                f"{ds['title']}: {ds['description']} "
                f"Domains={', '.join(ds.get('domains', []))}. "
                f"Trust={ds.get('trust_level', 'unknown')}. "
                f"Provider={ds.get('provider', 'unknown')}."
            )
        return "\n".join(lines)

    def required_columns(self, dataset_id: str, names: list[str]) -> list[str]:
        dataset = self.get(dataset_id)
        derived = set(dataset.get("derived_columns", {}).keys())
        required = set()
        for name in names:
            if name in derived:
                required.update(self._referenced_columns(dataset["derived_columns"][name]))
            else:
                required.add(name)
        return sorted(required)

    def load_frame(
        self,
        dataset_id: str,
        selected_names: list[str],
        where_sql: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        dataset = self.get(dataset_id)
        local_path = Path(dataset["local_path"])
        if not local_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {local_path}")

        select_exprs = []
        derived = dataset.get("derived_columns", {})
        for name in selected_names:
            if name in derived:
                select_exprs.append(f"{derived[name]} AS {name}")
            else:
                select_exprs.append(name)

        source_expr = self._source_expr(local_path, dataset["format"])
        sql = f"SELECT {', '.join(select_exprs)} FROM {source_expr}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        conn = duckdb.connect(database=":memory:")
        try:
            return conn.execute(sql).fetch_df()
        finally:
            conn.close()

    def _source_expr(self, local_path: Path, fmt: str) -> str:
        path_str = str(local_path).replace("'", "''")
        if fmt == "parquet":
            return f"read_parquet('{path_str}')"
        if fmt == "csv":
            return f"read_csv_auto('{path_str}', HEADER=TRUE)"
        raise ValueError(f"Unsupported dataset format: {fmt}")

    def _referenced_columns(self, expr: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        reserved = {"AS", "and", "or"}
        return [t for t in tokens if t.lower() not in reserved]

    def _tokenize(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.lower()))
        return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}

    def _dataset_terms(self, dataset: dict[str, Any]) -> set[str]:
        fields = []
        for key in ["title", "description", "provider", "citation"]:
            value = dataset.get(key)
            if value:
                fields.append(value)
        fields.extend(dataset.get("domains", []))
        fields.extend(dataset.get("semantic_keywords", []))
        for feature in dataset.get("profile_features", []):
            fields.append(feature["name"])
            fields.append(feature.get("label", ""))
        for item in dataset.get("target_items", []):
            fields.append(item["name"])
            fields.append(item.get("description", ""))
        return self._tokenize(" ".join(fields))
