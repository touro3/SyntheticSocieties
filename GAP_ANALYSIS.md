# BGF vs MiroFish — Current Gap Analysis

## Context

The codebase references MiroFish throughout as a reference architecture.
All "Mirrors MiroFish's…" comments indicate patterns **already adopted**.
This document summarises what's done and what's still behind.

---

## Already matched (all implemented)

| MiroFish pattern | BGF implementation |
|---|---|
| `valid_at` / `expired_at` on knowledge-graph nodes | `MemoryItem.valid_at` + `expires_at_round` + per-type TTL dict — `agents/memory.py:55–98` |
| Activity-batching (buffer N events → single graph push) | `HierarchicalMemory.add_batch()` + `_pending_buffer` flush at threshold 5 — `agents/memory.py:155–198` |
| Flask blueprint architecture (auth, rate-limit, CORS) | `api/app.py` — bearer-token auth, `flask_limiter`, blueprints, `/simulate`, `/status`, `/interview`, `/inject`, `/report` |
| OasisProfileGenerator persona enrichment | `build_persona_natural_language()` + `enrich_persona_from_graph()` — `population/persona_synthesizer.py:224–330` |
| `retry_with_backoff` on LLM calls | Jittered exponential backoff in `LLMBackend._generate_with_retries()` — `decision/llm_backend.py:142–469` |
| Staged multi-stage LLM config generation | `configs/llm_config_generator.py` — 4 sequential focused calls |
| `_try_fix_json` field-level JSON repair | `_repair_json()` + `_field_level_extract()` — `decision/output_parser.py:200–360` |
| ReACT agent: panorama, insight_forge, conflict resolution, None-guard | `analysis/react_report_agent.py` — full ReACT loop with all four patterns |
| SIGTERM / SIGINT / SIGHUP graceful shutdown | `simulation/signal_handler.py` — GracefulShutdown context manager |
| `ZepGraphMemoryUpdater` real-time episode-feeding | `SimulationKernel._narrate_and_update_memory()` — narration injected after each round with TTL=8 |

---

## Still behind MiroFish (gaps)

### 1. Persistent knowledge graph — **architectural gap**
MiroFish uses Zep (persistent Neo4j-style graph) via `ZepGraphMemoryUpdater`:
memory persists across runs and enables cross-run entity relationship queries.

BGF uses **in-memory** `HierarchicalMemory`. `enrich_persona_from_graph()` exists
(`population/persona_synthesizer.py:293`) but queries the transient
`NetworkManager`, not a persistent store. The graph is lost when the process exits.

**Impact**: no cross-run agent memory, no persistent social relationship graph.

---

### 2. GPU simulation runs — **critical paper blocker**
| Run | Command | Blocks |
|---|---|---|
| 10-seed A/B LLM comparison | `python scripts/run_experiment_matrix.py --include-llm --conditions A B --seeds 1..10 --rounds 30 --agents 100` | Bootstrap 95% CIs on all primary metrics (Table 2) |
| Padded control (Phase 29.1) | `python scripts/run_padded_control.py --seeds 42,123,7 --agents 500 --rounds 30` | Prompt-length confound control |

Both scripts + analysis infrastructure exist; results are pending GPU time.
Without these, all primary claims carry only "pilot" confidence.

---

### 3. Human evaluation — **not yet deployed**
Infrastructure is complete:
- Frontend: `human_experiment/app/index.html`
- Server: `human_experiment/server/server.py`
- Analysis: `scripts/analyze_human_baseline.py`
- Protocol: `docs/human_subjects_protocol.md`

Still missing: Prolific deployment (~$120), participant data, boxplot figure.
This is the single strongest possible evidence for the central claim.

---

### 4. LaTeX venue formatting — **DONE ✓**
`docs/paper_aamas.tex` stub created (ACM sigconf, 8 pages).
Abstract, contributions, formal spec, cross-cultural result, and bibliography
entries are populated.  TODO sections remain for §§ 4–6 (results tables).
Target: AAMAS 2027.

---

### 5. Reproducibility hardening — **DONE ✓**
- `decision/llm_backend.py` already used `BGF_MODEL_CACHE` env var — confirmed
- `scripts/launch_gpu_ab.sh`, `launch_cond_d.sh`, `run_queue.sh` hardcoded paths
  replaced with `$(cd "$(dirname "$0")/.." && pwd)` — **fixed**
- `requirements.txt`: all `>=` entries pinned to exact installed versions — **fixed**
- `requirements-api.txt`: `gunicorn>=23.0` pinned to `==23.0.0`; `python-dotenv` added — **fixed**
- `python-dotenv==1.2.2` added to both requirements files — **fixed**
- `Dockerfile` already existed; `SECURITY.md` already tracked — confirmed

---

### 6. Scaled cross-model + memory ablation GPU runs (Phase 29.3/29.4)
Infrastructure complete but GPU runs not yet done:
- Phase 29.3: `configs/cross_model/gpt4o_mini.yaml` → N=50, T=20 (inverse effect test)
- Phase 29.4: `configs/memory_ablation/m0_no_memory.yaml … m3_full.yaml` → 2×4 factorial × 3 seeds

---

## Priority order (what to do next)

```
1. GPU runs (#2)          — unblocks Table 2, bootstrap CIs, padded-control row
2. Human eval deploy (#3) — 5-day Prolific clock, starts immediately
3. LaTeX stub (#4)        — 30-min first pass, unblocks venue commitment
4. Repro hardening (#5)   — HF cache env var is 10 min; Docker ~1 hr
5. Persistent graph (#1)  — architectural, no clear deadline
```