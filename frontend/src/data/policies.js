// Shared policy definitions — single source of truth for RunView and HomeView
export const POLICIES = [
  { value: 'mock',              glyph: '◻', name: 'Mock',        desc: 'Fixed action every round — fastest',        gpu: false },
  { value: 'random',            glyph: '◈', name: 'Random',      desc: 'Uniform random action each round',          gpu: false },
  { value: 'rule_based',        glyph: '◆', name: 'Rule-Based',  desc: 'Heuristic rules on wealth + stress',        gpu: false },
  { value: 'template',          glyph: '▣', name: 'Template',    desc: 'Template prompt, no LLM — deterministic',   gpu: false },
  { value: 'llm',               glyph: '▲', name: 'LLM',         desc: 'Mistral-7B or GPT — grounded reasoning',    gpu: true  },
  { value: 'generative_agents', glyph: '⬡', name: 'Generative',  desc: 'Park et al. 2023 fictional persona',        gpu: true  },
]

export const POLICY_COUNT = POLICIES.length
