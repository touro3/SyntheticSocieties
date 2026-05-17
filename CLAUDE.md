CLAUDE.md
Context Navigation
When you need to understand the codebase, docs, or any files in this project:
ALWAYS query the knowledge graph first: `/graphify query "your question"`
Only read raw files if I explicitly say "read the file" or "look at the raw file"
Use `graphify-out/wiki/index.md` as your navigation entrypoint for browsing structure
---
Project Overview
SyntheticSocieties is an agent-based economic simulation framework (called BGF — Behavioral Grounding Framework) that tests whether LLMs grounded in empirical socio-economic data (from the European Social Survey, ESS) produce more realistic behavior than naive LLMs. Agents make economic decisions (work, save, cooperate, steal) in a game-theoretic setting while evolving across simulation rounds.
Deployment: HuggingFace Spaces (`touro3/synthetic-societies`), with a Flask API backend + Vue SPA frontend.
---
Common Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run baseline pipeline (no GPU needed: 5 rounds, 10 agents, no LLM)
python scripts/run_full_pipeline.py

# Run with LLM (GPU required: Mistral-7B-Instruct-v0.3)
python scripts/run_full_pipeline.py --include-llm --rounds 10 --agents 5

# Multi-seed statistical sweep
python scripts/run_full_pipeline.py --seeds 1,2,3,4,5 --include-llm

# Regenerate plots from existing experiments
python scripts/run_full_pipeline.py --plots-only

# Run tests
pytest tests/ -v
pytest tests/test_rag.py -v          # RAG-specific subset
python scripts/test_rag_features.py  # Interactive RAG showcase

# ESS calibration validation
python metrics/calibration.py

# Start API server (dev)
python api/app.py --debug
python api/app.py --host 0.0.0.0 --port 5050  # expose externally

# Production API
gunicorn 'api.app:create_app()'
```
Experiment Pipelines
```bash
bash scripts/pipeline_bad_apple.sh        # 5% adversarial agent injection
bash scripts/pipeline_macro_shock.sh      # Economic crisis (50% wealth shock at round 15)
bash scripts/pipeline_topology.sh         # Network topology experiments
bash scripts/pipeline_phase_c.sh          # 100 agents, 100 rounds, LLM with RAG
bash scripts/pipeline_phase_d.sh          # Large-scale 500-agent simulation
bash scripts/pipeline_cross_cultural.sh   # Nordic/Southern/Eastern cluster validation
bash scripts/run_all_experiments.sh       # Full paper reproduction
```
---
Architecture
```
ESS Data → Population synthesis → Agent creation → Simulation kernel → Metrics
                                                         ↑
                                              Flask API + Vue SPA
                                          (wizard, interviews, anchor)
```
Core Simulation Layers
`population/` — Synthesizes realistic agent populations from ESS survey data. `ess_grounding.py` samples empirical distributions; `generator.py` creates agent profiles; `persona_synthesizer.py` generates natural-language persona descriptions. `country_clusters.py` defines Nordic/Southern/Eastern ESS-11 cluster benchmarks.
`agents/` — Agent data structures. `agent.py` is the core class; `profile.py` holds demographic/psychological attributes; `state.py` tracks mutable state (wealth, stress); `memory.py` is a hierarchical memory system with a sliding window.
`decision/` — Pluggable policy system. All policies implement the same interface:
`random_policy.py`, `template_policy.py`, `rule_based_policy.py` — Non-LLM baselines
`llm_policy.py` — LLM-based decisions via `llm_backend.py` / `fast_batched_backend.py`
`generative_agents_policy.py` — Fictional-persona LLM policy (Park et al. 2023 proxy), no ESS/RAG
`sql_rag.py`, `graph_rag.py` — RAG backends that inject empirical context into prompts
`prompt_builder.py` — Constructs prompts from agent state, memory, world context
`output_parser.py` — Parses LLM JSON responses with regex fallback (anti-hallucination)
`model_config.py` — `ModelConfig` dataclass + `get_backend()` factory for cross-model support
`openai_backend.py` — OpenAI responses.create API with caching and retry
`environment/` — The economic world:
`economy.py` — `EconomyEngine`: parses agent actions, enforces game-theory rules. Actions: `work` (+8 wealth), `save` (+4), `cooperate` (-3, generates +12/cooperator shared), `steal` (bad apples take 50% of public pool)
`network.py` — `NetworkManager`: small-world (Watts-Strogatz) and random (Erdős–Rényi)
`institutions.py` — Institutional redistribution rules
`payoffs.py` — `DEFAULT_PAYOFFS` dataclass used by economy engine and human game
`world.py` / `world_state.py` — Global state and agent context assembly
`simulation/` — Runtime:
`kernel.py` — Event-driven main loop with batched GPU inference
`ipc.py` — `SimulationIPCServer` / `SimulationIPCClient` for live agent interviews, anchor queries, and exogenous event injection during running simulations
`crash_recovery.py` — `scan_incomplete_runs()` for resumable experiments
`metrics/` — 20+ evaluation dimensions: `inequality.py` (Gini), `calibration.py` (ESS comparison), `distribution.py`, `network_metrics.py`, `persona_fidelity.py`, `trajectories.py`, `macro_metrics.py`, `behavioral_realism.py` (BRM + B_RLHF), `persona_decay.py`, `mediation.py` (2×2 factorial), `complexity.py` (phase transitions + power laws), `trust_gradient.py` (Spearman rank), `cross_model.py`, `cross_cultural.py`.
`tracker/` — DuckDB-based experiment registry (`experiment_index.parquet`). `analytics.py` enables SQL queries and regression detection across experiments.
API + Frontend Layer
`api/app.py` — Flask REST API (the main entry point for the HuggingFace Space). This is a large file (~2000 lines) that contains:
All route handlers in a single `create_app()` factory function
Auth middleware (`_check_auth()`) — bearer-token auth via `BGF_API_TOKEN`
AI Simulation Design (`_call_design_llm()`) — routes to Groq/OpenAI/Ollama for scenario generation
Agent interview system — data-replay + OpenAI LLM for natural-language agent responses
Anchor agent — omniscient macro-view summarizer with opinion mining from interview logs
Simulation wizard (`/simulate-wizard`) — no-YAML simulation launcher
Human evaluation (`/human-eval/*`) — Prolific study vignettes
Human baseline game (`/human-game/*`) — interactive economic game for calibration
B_RLHF Benchmark (`/benchmark/*`) — specification, leaderboard, submissions
Network replay (`/replay/<exp_id>`) — reconstructs per-round interaction graph from events.jsonl
`api/static/` — Built Vue SPA (served by Flask)
`api/templates/` — Fallback Jinja2 templates when Vue build isn't present
Key API Endpoints
Method	Path	Auth	Purpose
GET	`/health`	No	Liveness probe + capability checks
GET	`/api/capabilities`	No	Which LLM providers are configured
POST	`/design-simulation`	Yes	AI scenario design via Groq/OpenAI/Ollama
POST	`/simulate-wizard`	Yes	Launch simulation from wizard params
POST	`/simulate`	Yes	Launch from YAML config path
GET	`/status/<exp_id>`	No	Poll run progress
GET	`/results/<exp_id>`	No	Metrics + summary for completed run
GET	`/replay/<exp_id>`	No	Per-round interaction network
POST	`/interview/<exp_id>/<agent_id>`	Yes	Interview agent (live IPC or data replay)
POST	`/anchor/<exp_id>`	Yes	Omniscient macro-view query
POST	`/inject/<exp_id>`	Yes	Inject live exogenous event
GET	`/experiments`	No	List all experiments from tracker
POST	`/upload-ess-data`	Yes	Upload custom ESS data file
Auth rule: `GET`/`HEAD`/`OPTIONS` are always public. All `POST`/`PUT`/`DELETE` require `Authorization: Bearer <BGF_API_TOKEN>` when the token is configured.
---
Environment Variables
Variable	Required	Purpose
`BGF_API_TOKEN`	For POST endpoints	Bearer token for API auth. Unset = open mode
`GROQ_API_KEY`	For Groq provider	Used by `/design-simulation` and LLM policies
`OPENAI_API_KEY`	For OpenAI provider	Used by design, interviews, anchor, report
`BGF_REPORT_API_KEY`	Optional	Overrides `OPENAI_API_KEY` for `/report` only
`BGF_CORS_ORIGINS`	Optional	Comma-separated CORS origins
`PORT`	Optional	Override default port 5050
The API loads `.env` from repo root via `python-dotenv` if installed.
On HuggingFace Spaces, set these as Secrets in Settings.
---
Configuration
Hydra-based configs in `configs/`. Default: `configs/base_config.yaml`.
Key config values:
`policy.type`: `mock | random | template | rule_based | llm | generative_agents`
`population.source`: `empirical` (ESS) or `synthetic`
`llm.model_id`: defaults to `mistralai/Mistral-7B-Instruct-v0.3`
`llm.backend_type`: `huggingface` (local GPU) | `openai` (API) | `groq` | `ollama`
`network.type`: `random | small_world`
Cross-model configs: `configs/cross_model/mistral.yaml`, `qwen2.5-7b.yaml`, `gpt4o_mini.yaml`
Condition C config: `configs/condition_c.yaml`
The wizard endpoint (`/simulate-wizard`) generates YAML configs at runtime in `configs/wizard/`.
---
Experiment Outputs
Each run produces `experiments/<exp_id>/` containing:
`run_state.json` — Status tracking (running/complete/failed), polled by `/status`
`heartbeat.json` — Live progress counter (round_id) during execution
`metadata.json` — Policy type, population size, seed
`events.jsonl` — Round-level agent decisions and states (used by replay, interview, anchor)
`prompts.jsonl` — LLM prompts (debugging/analysis)
`metrics.json` — Aggregate metrics
`summary.json` — Wealth/action summaries
`config.yaml` — Run configuration snapshot
`scenario.json` — AI-design context (if created via wizard + design)
`interview_responses.jsonl` — Accumulated Q&A pairs from `/interview` calls (used by anchor for opinion tallying)
Plots: `analysis/figures/`. Network exports: `analysis/networks/`.
---
Key Design Patterns
Experimental Conditions
Condition A: "Pure LLM" — no grounding, no RAG
Condition B: "Grounded LLM" — RAG-injected ESS distributions (central experimental contrast)
Condition C: "Generative Agents" — fictional-persona LLM policy (Park et al. 2023 proxy), no ESS/RAG
Agent Mechanics
Bad apple agents: `is_adversarial=True` on profile. `EconomyEngine.parse_action()` hard-constrains them to steal-only
Anti-hallucination: `output_parser.py` uses strict JSON + regex fallback; `EconomyEngine` maps invalid LLM responses to `work`
Memory: hierarchical with sliding window and long-term summary slot
RAG: `sql_rag.py` + `graph_rag.py` provide dual retrieval. Verify both are active in Condition B runs
API Patterns
Auth flow: `_check_auth()` → GETs pass freely, POSTs require bearer token when `BGF_API_TOKEN` is set. The frontend must send `Authorization: Bearer <token>` on every POST
Design LLM caching: `_DESIGN_CACHE` (128 entries, keyed by sha256 of provider+prompt) avoids duplicate API calls
OpenAI client singleton: `_OAI_CLIENTS` dict reuses clients by api_key prefix to avoid TCP reconnect overhead
Events cache: `_read_events_cached()` uses byte-offset caching so repeated reads of `events.jsonl` only parse new lines (O(new) not O(total))
IPC bridge: `SimulationIPCClient/Server` enables live agent interview, anchor queries, and event injection during running simulations. Falls back to data-replay for completed runs
Interview → Anchor pipeline: Each `/interview` call appends to `interview_responses.jsonl`. The `/anchor` endpoint reads these to tally opinions across agents (e.g., "majority preferred X")
Wizard config generation: `/simulate-wizard` writes a YAML to `configs/wizard/`, pre-writes `run_state.json`, then spawns `scripts/run_config_simulation.py` in a daemon thread
Infrastructure
Batched inference: `fast_batched_backend.py` handles GPU batching; `llm_backend.py` is single-agent fallback
Rate limiting: `flask-limiter` with per-endpoint limits (optional dependency)
File uploads: ESS data via `/upload-ess-data` → `uploads/ess_data/<uuid>.parquet` + `<uuid>_analysis.json` sidecar
Population synthesis from design: `_generate_population_parquet()` creates trait distributions from AI-designed parameters
---
Common Debugging Patterns
"Authorization header required" on POST endpoints
The frontend isn't sending the `Authorization: Bearer <BGF_API_TOKEN>` header. Check:
Is `BGF_API_TOKEN` set in the environment/Secrets?
Is the frontend including the header on POST requests?
GETs work without auth — only POSTs are gated
Simulation stuck at 0 rounds
`/status` detects this: if running + 0 completed + updated >90s ago → marks `stale: true`. For LLM policies, also sets `gpu_wait: true`. Check:
LLM model path / API key availability
GPU memory (for HuggingFace backend)
`run_state.json` in the experiment dir
Interview returns "replay_data" instead of "replay_llm"
No `OPENAI_API_KEY` configured → falls back to rule-based keyword matching. Set the key for natural-language responses.
Design LLM provider selection
Auto-selection priority: Ollama (local) → Groq → OpenAI. Override with `provider` field in request body.
Checking available providers
`GET /api/capabilities` returns which providers have keys configured and which is preferred.
Experiment not found after wizard launch
`/simulate-wizard` pre-writes `run_state.json` before spawning the thread, so `/status` should never 404 immediately. If it does, check filesystem permissions on `experiments/`.
---
Directory Structure
```
api/                    Flask API + Vue SPA
  app.py               Main API (all routes in create_app())
  static/              Built Vue frontend assets
  templates/           Fallback Jinja2 templates
agents/                Agent data structures
analysis/              Analysis scripts, figures, network exports
  react_report_agent.py  ReACT agent for /report endpoint
benchmark/             B_RLHF benchmark spec, leaderboard, submissions
bgf_logging/           Event and prompt logging
configs/               Hydra YAML configs
  wizard/              Auto-generated configs from /simulate-wizard
data/                  ESS datasets (Parquet)
  human/               Human eval ratings + game responses
decision/              Policy system + LLM backends + RAG
docs/                  Paper draft, hypotheses, eval protocol
environment/           Economy engine, network, institutions, payoffs
experiments/           Completed experiment runs (~105)
human_experiment/      Standalone human baseline game (Prolific)
  app/                 Static HTML/JS served at /human-game/
metrics/               20+ evaluation modules
population/            Population synthesis from ESS
scripts/               Pipeline scripts, plotting utilities
simulation/            Kernel, IPC bridge, crash recovery
tracker/               DuckDB experiment registry
tests/                 636+ tests (pytest)
uploads/               Runtime file uploads
  ess_data/            Uploaded ESS parquets + analysis sidecars
utils/                 Config and I/O utilities
```
---
Testing
```bash
pytest tests/ -v                     # Full suite (636+ tests)
pytest tests/test_rag.py -v          # RAG-specific
python scripts/test_rag_features.py  # Interactive RAG showcase
```
Shared fixtures in `conftest.py`. Includes unit, integration, cross-model, cross-cultural, and metric validation tests.