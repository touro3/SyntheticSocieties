# BGF Architecture

Visual: open `docs/architecture.excalidraw` at [excalidraw.com](https://excalidraw.com)
(File → Open) or via the Excalidraw VS Code extension. Mermaid fallback below.

```mermaid
flowchart TD
    ESS["ESS Survey Data<br/><i>data/ · Parquet</i>"]
    POP["Population Synthesis<br/><i>population/</i><br/>ess_grounding · generator · persona_synthesizer"]
    AG["Agents<br/><i>agents/</i><br/>agent · profile · state · hierarchical memory"]
    DEC["Decision / Policy Layer<br/><i>decision/</i><br/>Protocol: random | template | rule | LLM | gen-agents"]
    ENV["Environment<br/><i>environment/</i><br/>economy · network (NetworkX) · institutions · world"]
    KER["Simulation Kernel<br/><i>simulation/kernel.py</i><br/>event loop · batched GPU inference"]
    MET["Metrics<br/><i>metrics/</i><br/>Gini · calibration · BRM · mediation · 20+ dims"]
    TRK["Experiment Tracker<br/><i>tracker/ · DuckDB + Parquet</i>"]

    ESS --> POP --> AG --> DEC --> ENV --> KER --> MET --> TRK

    LLM["LLM Backends<br/>HF Mistral-7B (GPU) | OpenAI API"]
    RAG["Dual RAG<br/>sql_rag + graph_rag (ESS injection)"]
    CFG["Hydra Config<br/><i>configs/</i>"]
    LOG["bgf_logging/<br/>event + prompt logs (repro)"]

    LLM --> DEC
    RAG --> DEC
    CFG -.-> POP
    CFG -.-> KER
    LOG -.-> KER

    classDef data fill:#ffd8a8,stroke:#1e1e1e
    classDef policy fill:#a5d8ff,stroke:#1e1e1e
    classDef cross fill:#ffec99,stroke:#1e1e1e
    class ESS data
    class DEC,LLM,RAG policy
    class CFG,LOG cross
```

**Data flow:** ESS empirical distributions ground synthetic agent populations →
agents make decisions through a pluggable policy `Protocol` (LLM backends + dual
RAG inject context) → the economic environment resolves actions → the kernel
drives the event loop with batched inference → metrics evaluate realism → every
run is registered in the DuckDB tracker. Hydra config and `bgf_logging` are
cross-cutting (snapshots + full reproducibility).
