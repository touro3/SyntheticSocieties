<template>
  <div>
    <div class="page-header">
      <div>
        <h1 class="page-title">Run Simulation</h1>
        <p class="page-sub">Configure your simulation with the visual builder — no YAML or code needed.</p>
      </div>
    </div>

    <!-- Step progress indicator -->
    <div class="step-progress">
      <div v-for="(s, i) in stepLabels" :key="i" class="sp-item">
        <div class="sp-dot" :class="{ done: i < currentStep }">{{ i + 1 }}</div>
        <span class="sp-label">{{ s }}</span>
        <div v-if="i < stepLabels.length - 1" class="sp-line"></div>
      </div>
    </div>

    <div class="wizard-layout">

      <!-- ── LEFT: Wizard form ─────────────────────────────────────── -->
      <div class="wizard-form">

        <!-- Step 0: AI Design (optional) -->
        <div class="step-card card design-card">
          <div class="step-head">
            <span class="step-num design-num">✦</span>
            <div>
              <div class="step-title">
                AI Simulation Design
                <span class="optional-tag">optional</span>
              </div>
              <div class="step-sub">Describe your scenario — the AI configures the wizard and synthesises a population</div>
            </div>
          </div>

          <textarea
            v-model="designPrompt"
            class="design-textarea"
            placeholder="e.g. Simulate 50 Wall Street traders in a competitive market with 5% corrupt agents and a market crash in round 20…"
            rows="3"
          />

          <!-- AI provider picker for design -->
          <div class="design-provider-row">
            <button v-for="dp in designProviders" :key="dp.value"
              class="dprov-btn"
              :class="{
                selected: designProvider === dp.value,
                unavailable: serverCaps !== null && !dp.available(serverCaps)
              }"
              @click="designProvider = dp.value"
              :title="dp.tooltip(serverCaps)"
            >
              <span class="dprov-glyph">{{ dp.glyph }}</span>
              <span class="dprov-name">{{ dp.name }}</span>
              <span class="dprov-tag" :class="dp.tagClass(serverCaps)">{{ dp.tag(serverCaps) }}</span>
            </button>
          </div>

          <!-- Ollama model selector when Ollama is chosen -->
          <div v-if="designProvider === 'ollama' && serverCaps?.design?.ollama" class="design-model-row">
            <label class="dm-label">Model</label>
            <div class="dm-chips">
              <button v-for="m in ollamaDesignModels" :key="m.id"
                class="dm-chip" :class="{ selected: designOllamaModel === m.id }"
                @click="designOllamaModel = m.id">
                {{ m.name }} <span class="dm-note">{{ m.note }}</span>
              </button>
            </div>
          </div>

          <div class="design-controls">
            <button
              class="btn btn-primary design-btn"
              @click="runDesign"
              :disabled="!designPrompt.trim() || designStatus === 'loading' || !canDesign"
            >
              <span v-if="designStatus === 'loading'" class="spin">⟳</span>
              <span v-else>✦</span>
              {{ designStatus === 'loading' ? 'Designing…' : 'Design with AI' }}
            </button>
            <div v-if="!canDesign && serverCaps" class="design-unavail-hint">
              {{ designUnavailReason }}
            </div>
          </div>

          <div v-if="designStatus === 'error'" class="upload-error-msg">✗ {{ designError }}</div>

          <!-- Result panel -->
          <div v-if="designResult" class="design-result">
            <div class="design-title">{{ designResult.scenario_title }}</div>
            <div class="design-desc">{{ designResult.scenario_description }}</div>

            <div class="design-chips">
              <span class="chip">{{ designResult.config?.agents }} agents</span>
              <span class="chip">{{ designResult.config?.rounds }} rounds</span>
              <span class="chip">{{ designResult.config?.policy }}</span>
              <span class="chip">{{ designResult.config?.network_type }}</span>
              <span v-if="designResult.config?.bad_apple_frac > 0" class="chip chip-warn">
                {{ (designResult.config.bad_apple_frac * 100).toFixed(0) }}% adversarial
              </span>
            </div>

            <div v-if="designResult.population_narrative" class="design-narrative">
              <span class="narrative-label">Population context → injected into agent prompts</span>
              {{ designResult.population_narrative }}
            </div>

            <div v-if="designResult.reasoning" class="design-reasoning">
              <span class="reasoning-label">Reasoning</span>
              {{ designResult.reasoning }}
            </div>

            <div class="design-action-row">
              <button v-if="!designApplied" class="btn btn-primary" @click="applyDesign">
                Apply to Wizard →
              </button>
              <div v-else class="design-applied-badge">✓ Applied — wizard configured below</div>
              <button class="btn btn-ghost btn-sm" @click="clearDesign">Clear</button>
            </div>
          </div>
        </div>

        <!-- Step 1: Policy -->
        <div class="step-card card">
          <div class="step-head">
            <span class="step-num">1</span>
            <div>
              <div class="step-title">Policy Engine</div>
              <div class="step-sub">How agents make decisions each round</div>
            </div>
          </div>
          <div class="policy-grid">
            <button v-for="p in policies" :key="p.value"
              class="policy-btn" :class="{ selected: form.policy === p.value }"
              @click="form.policy = p.value">
              <span class="p-glyph">{{ p.glyph }}</span>
              <span class="p-name">{{ p.name }}</span>
              <span class="p-desc">{{ p.desc }}</span>
              <span class="p-tag" :class="p.gpu ? 'gpu' : 'cpu'">{{ p.gpu ? 'GPU' : 'CPU' }}</span>
            </button>
          </div>

          <!-- LLM provider selector -->
          <div v-if="selectedPolicy?.gpu" class="llm-config">
            <div class="llm-config-head">AI Provider for Agents</div>
            <div class="provider-grid">
              <button v-for="p in providers" :key="p.value"
                class="provider-btn"
                :class="{
                  selected: form.llm_backend === p.value,
                  unavailable: serverCaps !== null && !providerAvailable(p, serverCaps)
                }"
                @click="selectProvider(p)">
                <div class="prov-top">
                  <span class="prov-glyph">{{ p.glyph }}</span>
                  <span v-if="p.free" class="prov-free">Free</span>
                </div>
                <span class="prov-name">{{ p.name }}</span>
                <span class="prov-desc">{{ p.desc }}</span>
                <span class="prov-status" v-if="serverCaps">
                  {{ providerAvailable(p, serverCaps) ? '✓ Ready' : '✗ Not configured' }}
                </span>
              </button>
            </div>

            <!-- Model picker -->
            <div class="field" v-if="selectedProvider?.models?.length">
              <label>Model</label>
              <div class="model-row">
                <button v-for="m in selectedProvider.models" :key="m.id"
                  class="model-btn" :class="{ selected: form.llm_model_id === m.id }"
                  @click="form.llm_model_id = m.id">
                  <span class="mb-name">{{ m.name }}</span>
                  <span class="mb-note">{{ m.note }}</span>
                </button>
              </div>
            </div>

            <!-- No key input — all keys are server-side -->
            <div v-if="form.llm_backend === 'huggingface'" class="warn-box">
              Requires a CUDA-capable GPU on the server.
            </div>
          </div>
        </div>

        <!-- Step 2: Scale -->
        <div class="step-card card">
          <div class="step-head">
            <span class="step-num">2</span>
            <div>
              <div class="step-title">Scale</div>
              <div class="step-sub">Population size and simulation length</div>
            </div>
          </div>

          <div class="slider-group">
            <div class="slider-row">
              <label>Agents <span class="slider-val">{{ form.agents }}</span></label>
              <input type="range" v-model.number="form.agents"
                min="5" max="200" step="5" class="slider" />
              <div class="slider-ticks">
                <span>5</span><span>50</span><span>100</span><span>200</span>
              </div>
            </div>
            <div class="slider-row">
              <label>Rounds <span class="slider-val">{{ form.rounds }}</span></label>
              <input type="range" v-model.number="form.rounds"
                min="5" max="100" step="5" class="slider" />
              <div class="slider-ticks">
                <span>5</span><span>25</span><span>50</span><span>100</span>
              </div>
            </div>
            <div class="slider-row">
              <label>Random seed <span class="slider-val mono">{{ form.seed }}</span></label>
              <input type="number" v-model.number="form.seed" min="0" max="99999" style="max-width:140px" />
            </div>
          </div>

          <div class="est-box">
            <span class="est-label">Estimated runtime:</span>
            <span class="est-val">{{ estimatedTime }}</span>
          </div>
        </div>

        <!-- Step 3: Network -->
        <div class="step-card card">
          <div class="step-head">
            <span class="step-num">3</span>
            <div>
              <div class="step-title">Network Topology</div>
              <div class="step-sub">Social graph structure connecting agents</div>
            </div>
          </div>
          <div class="option-row">
            <button v-for="n in networks" :key="n.value"
              class="option-btn" :class="{ selected: form.network_type === n.value }"
              @click="form.network_type = n.value">
              <span class="o-glyph">{{ n.glyph }}</span>
              <span class="o-name">{{ n.name }}</span>
              <span class="o-desc">{{ n.desc }}</span>
            </button>
          </div>
        </div>

        <!-- Step 4: Population -->
        <div class="step-card card">
          <div class="step-head">
            <span class="step-num">4</span>
            <div>
              <div class="step-title">Population Source</div>
              <div class="step-sub">Agent demographic profiles</div>
            </div>
          </div>
          <div class="option-row">
            <button v-for="s in sources" :key="s.value"
              class="option-btn" :class="{ selected: form.population_source === s.value }"
              @click="form.population_source = s.value">
              <span class="o-glyph">{{ s.glyph }}</span>
              <span class="o-name">{{ s.name }}</span>
              <span class="o-desc">{{ s.desc }}</span>
            </button>
          </div>

          <!-- Custom data file upload (empirical only) -->
          <div v-if="form.population_source === 'empirical'" class="ess-upload">
            <div class="ess-upload-head">
              Custom Data File
              <span class="optional-tag">optional</span>
            </div>
            <div class="ess-upload-desc">
              Upload any <code>.csv</code> or <code>.parquet</code> file. The system will analyze it
              and map whatever columns are present to simulation dimensions — no required columns.
              Leave empty to use the built-in ESS Round 11 dataset.
            </div>

            <label class="upload-label" :class="{ uploading: uploadStatus === 'uploading' }">
              <input type="file" accept=".csv,.parquet" class="upload-input" @change="uploadEssFile" />
              <span class="upload-btn">
                <span v-if="uploadStatus === 'uploading'" class="spin">⟳</span>
                <span v-else>↑</span>
                {{ uploadStatus === 'uploading' ? 'Analyzing…' : (uploadStatus === 'done' ? '↑ Replace file' : 'Choose file (.csv / .parquet)') }}
              </span>
            </label>

            <!-- Error state -->
            <div v-if="uploadStatus === 'error'" class="upload-error-msg">
              ✗ {{ uploadError }}
            </div>

            <!-- Analysis report (NotebookLM-style) -->
            <div v-if="uploadStatus === 'done' && uploadInfo?.analysis" class="analysis-report">

              <!-- Filename chip -->
              <div class="upload-filename-chip">
                <span class="file-icon">◈</span>
                <span class="file-name">{{ uploadFilename }}</span>
                <span class="file-rows">{{ uploadInfo.rows.toLocaleString() }} rows</span>
              </div>

              <!-- Narrative (the key "NotebookLM" output — used in agent prompts) -->
              <div class="analysis-narrative">
                {{ uploadInfo.analysis.narrative }}
              </div>

              <!-- Summary bar -->
              <div class="analysis-summary">
                <span class="analysis-icon">◉</span>
                <span class="analysis-text">{{ uploadInfo.analysis.summary }}</span>
                <span class="coverage-badge"
                  :class="uploadInfo.analysis.coverage.pct >= 70 ? 'cov-high' : uploadInfo.analysis.coverage.pct >= 40 ? 'cov-mid' : 'cov-low'">
                  {{ uploadInfo.analysis.coverage.pct }}%
                </span>
              </div>

              <!-- Coverage breakdown -->
              <div class="dim-coverage">
                <div class="dim-row" v-for="d in uploadInfo.analysis.dimensions" :key="d.name">
                  <span class="dim-status" :class="'dim-' + d.status">
                    {{ d.status === 'direct' ? '✓' : d.status === 'computed' ? '◐' : '—' }}
                  </span>
                  <span class="dim-name">{{ d.name.replace(/_/g, ' ') }}</span>
                  <span class="dim-source">{{ d.status === 'fallback' ? 'config default' : d.source }}</span>
                  <span class="dim-mean" v-if="d.stats?.mean != null">
                    μ {{ d.stats.mean.toFixed(2) }}
                  </span>
                </div>
              </div>

              <!-- Quality row -->
              <div class="quality-row">
                <span class="q-item">
                  <span class="q-label">Completeness</span>
                  <span class="q-val">{{ uploadInfo.analysis.quality.completeness_pct }}%</span>
                </span>
                <span class="q-item">
                  <span class="q-label">Direct dims</span>
                  <span class="q-val">{{ uploadInfo.analysis.coverage.direct }}</span>
                </span>
                <span class="q-item">
                  <span class="q-label">Derived dims</span>
                  <span class="q-val">{{ uploadInfo.analysis.coverage.computed }}</span>
                </span>
                <span class="q-item">
                  <span class="q-label">Fallback dims</span>
                  <span class="q-val">{{ uploadInfo.analysis.coverage.fallback }}</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Step 5: Adversarial agents -->
        <div class="step-card card">
          <div class="step-head">
            <span class="step-num">5</span>
            <div>
              <div class="step-title">Bad Apple Injection</div>
              <div class="step-sub">Adversarial agents that steal from the public pool</div>
            </div>
          </div>
          <div class="slider-row">
            <label>Fraction of adversarial agents <span class="slider-val">{{ (form.bad_apple_frac * 100).toFixed(0) }}%</span></label>
            <input type="range" v-model.number="form.bad_apple_frac"
              min="0" max="0.3" step="0.05" class="slider" />
            <div class="slider-ticks">
              <span>0%</span><span>10%</span><span>20%</span><span>30%</span>
            </div>
          </div>
        </div>

        <!-- Launch button -->
        <button class="btn btn-primary launch-btn" @click="launch" :disabled="launching">
          <span v-if="launching" class="spin">⟳</span>
          <span v-else>▶</span>
          {{ launching ? 'Launching…' : 'Launch Simulation' }}
        </button>

        <div v-if="error" class="error-box">{{ error }}</div>

      </div>

      <!-- ── RIGHT: Summary + info ────────────────────────────────── -->
      <div class="wizard-sidebar">

        <!-- Config summary -->
        <div class="card summary-card">
          <h3 class="section-title">Config Summary</h3>
          <div class="summary-rows">
            <div class="sum-row">
              <span class="sum-key">Policy</span>
              <span class="sum-val">
                <span class="badge" :class="selectedPolicy?.gpu ? 'badge-purple' : 'badge-teal'">
                  {{ form.policy }}
                </span>
              </span>
            </div>
            <div class="sum-row" v-if="selectedPolicy?.gpu">
              <span class="sum-key">Provider</span>
              <span class="sum-val mono">{{ selectedProvider?.name ?? form.llm_backend }}</span>
            </div>
            <div class="sum-row" v-if="selectedPolicy?.gpu && form.llm_model_id">
              <span class="sum-key">Model</span>
              <span class="sum-val mono" style="font-size:.72rem">{{ form.llm_model_id }}</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Agents</span>
              <span class="sum-val mono">{{ form.agents }}</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Rounds</span>
              <span class="sum-val mono">{{ form.rounds }}</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Network</span>
              <span class="sum-val mono">{{ form.network_type }}</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Population</span>
              <span class="sum-val mono">{{ form.population_source }}</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Bad apples</span>
              <span class="sum-val mono">{{ (form.bad_apple_frac * 100).toFixed(0) }}%</span>
            </div>
            <div class="sum-row">
              <span class="sum-key">Seed</span>
              <span class="sum-val mono">{{ form.seed }}</span>
            </div>
          </div>
        </div>

        <!-- Economic actions reference -->
        <div class="card">
          <h3 class="section-title">Economic Actions</h3>
          <div class="action-list">
            <div v-for="a in actions" :key="a.name" class="action-item">
              <span class="act-name mono">{{ a.name }}</span>
              <span class="act-delta" :style="{ color: a.color }">{{ a.delta }}</span>
              <span class="act-desc">{{ a.desc }}</span>
            </div>
          </div>
        </div>

        <!-- Launched state -->
        <div v-if="launched" class="card launched-card">
          <div class="launched-icon">✓</div>
          <h3>Simulation Started</h3>
          <p class="mono" style="font-size:.78rem;color:var(--text3)">{{ launchedId }}</p>
          <div style="display:flex;gap:8px;margin-top:14px">
            <router-link :to="`/monitor/${launchedId}`" class="btn btn-primary btn-sm">
              Monitor →
            </router-link>
            <button class="btn btn-ghost btn-sm" @click="resetForm">New run</button>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'

const router = useRouter()

const form = ref({
  policy: 'rule_based',
  agents: 20,
  rounds: 10,
  network_type: 'random',
  population_source: 'synthetic',
  bad_apple_frac: 0,
  seed: 42,
  llm_backend: 'ollama',
  llm_model_id: 'llama3.2',
  llm_api_key: '',
  ess_data_file_id: '',
  design_id: '',
})

const launching      = ref(false)
const error          = ref('')
const launched       = ref(false)
const launchedId     = ref('')
const uploadStatus   = ref('')
const uploadInfo     = ref(null)
const uploadError    = ref('')
const uploadFilename = ref('')

// Server capabilities (populated on mount)
const serverCaps = ref(null)  // null = loading, then the /api/capabilities payload

onMounted(async () => {
  try {
    const r = await api.capabilities()
    serverCaps.value = r.data
  } catch {
    serverCaps.value = { design: { server_configured: false, preferred: null }, simulation: {} }
  }
})

// True when server already has a provider configured — hide all key/provider UI
const serverDesignReady = computed(() => serverCaps.value?.design?.server_configured === true)
const serverDesignLabel = computed(() => {
  const p = serverCaps.value?.design?.preferred
  if (p === 'ollama') return 'Ollama (local AI · free)'
  if (p === 'groq')   return 'Groq (free cloud AI)'
  if (p === 'openai') return 'OpenAI'
  return 'AI'
})

const ollamaDesignModels = [
  { id: 'llama3.2', name: 'Llama 3.2', note: '3B · fastest' },
  { id: 'llama3.1', name: 'Llama 3.1', note: '8B · balanced' },
  { id: 'mistral',  name: 'Mistral',   note: '7B' },
  { id: 'gemma2',   name: 'Gemma 2',   note: '9B' },
  { id: 'qwen2.5',  name: 'Qwen 2.5',  note: '7B' },
]

const designProviders = [
  {
    value: 'ollama',
    glyph: '◈', name: 'Ollama',
    available: (caps) => caps?.design?.ollama === true,
    tag:     (caps) => caps === null ? '…' : caps?.design?.ollama ? '✓ Local · free' : '✗ Not running',
    tagClass:(caps) => caps?.design?.ollama ? 'dprov-ok' : 'dprov-off',
    tooltip: (caps) => caps?.design?.ollama
      ? 'Ollama is running on this server — no key needed'
      : 'Ollama is not running on this server',
  },
  {
    value: 'groq',
    glyph: '▲', name: 'Groq',
    available: (caps) => caps?.design?.groq === true,
    tag:     (caps) => caps === null ? '…' : caps?.design?.groq ? '✓ Ready · free' : '✗ No key',
    tagClass:(caps) => caps?.design?.groq ? 'dprov-ok' : 'dprov-off',
    tooltip: (caps) => caps?.design?.groq
      ? 'Groq API key is configured — fast, free cloud AI'
      : 'No Groq key configured on this server',
  },
  {
    value: 'openai',
    glyph: '◆', name: 'GPT-4o',
    available: (caps) => caps?.design?.openai === true,
    tag:     (caps) => caps === null ? '…' : caps?.design?.openai ? '✓ Ready' : '✗ No key',
    tagClass:(caps) => caps?.design?.openai ? 'dprov-ok' : 'dprov-off',
    tooltip: (caps) => caps?.design?.openai
      ? 'OpenAI key is configured'
      : 'No OpenAI key configured on this server',
  },
]

function providerAvailable(p, caps) {
  if (!caps) return true  // optimistic before load
  if (p.value === 'ollama')      return caps.simulation?.ollama === true
  if (p.value === 'groq')        return caps.simulation?.groq   === true
  if (p.value === 'openai')      return caps.simulation?.openai === true
  if (p.value === 'huggingface') return true  // always listed; GPU requirement shown separately
  return true
}

const canDesign = computed(() => {
  const caps = serverCaps.value
  if (caps === null) return false  // still loading
  const dp = designProviders.find(d => d.value === designProvider.value)
  return dp ? dp.available(caps) : false
})

const designUnavailReason = computed(() => {
  const p = designProvider.value
  if (p === 'ollama') return 'Ollama is not running on this server. Ask the admin to run: ollama serve'
  if (p === 'groq')   return 'No Groq key is configured on this server.'
  if (p === 'openai') return 'No OpenAI key is configured on this server.'
  return 'This provider is not available.'
})

// AI design state
const designPrompt      = ref('')
const designProvider    = ref('groq')
const designOllamaModel = ref('llama3.2')
const designStatus      = ref('')   // '' | 'loading' | 'done' | 'error'
const designError       = ref('')
const designResult      = ref(null)
const designApplied     = ref(false)

const policies = [
  { value: 'mock',              glyph: '◻', name: 'Mock',        desc: 'Fixed action every round — fastest',        gpu: false },
  { value: 'random',            glyph: '◈', name: 'Random',      desc: 'Uniform random action each round',          gpu: false },
  { value: 'rule_based',        glyph: '◆', name: 'Rule-Based',  desc: 'Heuristic rules on wealth + stress',        gpu: false },
  { value: 'template',          glyph: '▣', name: 'Template',    desc: 'Template prompt, no LLM — deterministic',   gpu: false },
  { value: 'llm',               glyph: '▲', name: 'LLM',         desc: 'Mistral-7B or GPT — grounded reasoning',    gpu: true  },
  { value: 'generative_agents', glyph: '⬡', name: 'Generative',  desc: 'Park et al. 2023 fictional persona',        gpu: true  },
]

const providers = [
  {
    value: 'ollama', glyph: '◈', name: 'Ollama', free: true,
    desc: 'Local — no key, no GPU',
    needsKey: false,
    models: [
      { id: 'llama3.2',        name: 'Llama 3.2',   note: '3B · very fast' },
      { id: 'llama3.1',        name: 'Llama 3.1',   note: '8B · balanced' },
      { id: 'mistral',         name: 'Mistral',     note: '7B · capable' },
      { id: 'gemma2',          name: 'Gemma 2',     note: '9B · Google' },
      { id: 'qwen2.5',         name: 'Qwen 2.5',    note: '7B · multilingual' },
    ],
  },
  {
    value: 'groq', glyph: '▲', name: 'Groq', free: true,
    desc: 'Cloud API · free tier',
    needsKey: true,
    keyLabel: 'Groq API Key',
    keyPlaceholder: 'gsk_…',
    keyLink: 'https://console.groq.com/keys',
    models: [
      { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B', note: 'Best quality' },
      { id: 'llama-3.1-8b-instant',    name: 'Llama 3.1 8B',  note: 'Fastest' },
      { id: 'mixtral-8x7b-32768',      name: 'Mixtral 8x7B',  note: 'Long context' },
      { id: 'gemma2-9b-it',            name: 'Gemma 2 9B',    note: 'Efficient' },
    ],
  },
  {
    value: 'openai', glyph: '◆', name: 'OpenAI', free: false,
    desc: 'ChatGPT · paid API',
    needsKey: true,
    keyLabel: 'OpenAI API Key',
    keyPlaceholder: 'sk-…',
    keyLink: 'https://platform.openai.com/api-keys',
    models: [
      { id: 'gpt-4o-mini', name: 'GPT-4o mini', note: 'Fast · low cost' },
      { id: 'gpt-4o',      name: 'GPT-4o',      note: 'Best quality' },
    ],
  },
  {
    value: 'huggingface', glyph: '◻', name: 'Local GPU', free: true,
    desc: 'Mistral-7B · needs CUDA',
    needsKey: false,
    models: [
      { id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral 7B v0.3', note: 'Default' },
    ],
  },
]

const networks = [
  { value: 'random',      glyph: '◌', name: 'Erdős–Rényi',    desc: 'Random edges — each pair connected with prob p' },
  { value: 'small_world', glyph: '⬡', name: 'Watts-Strogatz', desc: 'Small-world — high clustering + short paths' },
]

const sources = [
  { value: 'synthetic', glyph: '◎', name: 'Synthetic',  desc: 'Default demographic profiles' },
  { value: 'empirical', glyph: '◉', name: 'Empirical',  desc: 'Sampled from ESS Round 11 microdata' },
]

const actions = [
  { name: 'work',      delta: '+8 wealth',  desc: 'Earn from employment',          color: 'var(--green)' },
  { name: 'save',      delta: '+4 wealth',  desc: 'Accumulate savings',            color: 'var(--teal)' },
  { name: 'cooperate', delta: '-3 + pool',  desc: 'Lose 3, generate +12 shared',   color: 'var(--blue2)' },
  { name: 'steal',     delta: '50% pool',   desc: 'Bad apples only — drains pool', color: 'var(--rose)' },
]

const selectedPolicy   = computed(() => policies.find(p => p.value === form.value.policy))
const selectedProvider = computed(() => providers.find(p => p.value === form.value.llm_backend))

const stepLabels = ['Policy', 'Scale', 'Network', 'Population', 'Adversarial']
const currentStep = computed(() => {
  if (launched.value) return 5
  if (form.value.bad_apple_frac > 0) return 4
  if (form.value.population_source !== 'synthetic') return 3
  if (form.value.network_type !== 'random') return 2
  return 1
})

function selectProvider(p) {
  form.value.llm_backend  = p.value
  form.value.llm_model_id = p.models?.[0]?.id ?? ''
  form.value.llm_api_key  = ''  // kept in form shape but never sent
}

const estimatedTime = computed(() => {
  const { policy, agents, rounds, llm_backend, llm_model_id } = form.value
  if (policy === 'mock')       return `~${Math.max(1, Math.round(agents * rounds / 10000 * 2))}s`
  if (policy === 'random')     return `~${Math.max(1, Math.round(agents * rounds / 5000 * 2))}s`
  if (policy === 'rule_based') return `~${Math.max(1, Math.round(agents * rounds / 3000 * 2 + 1))}s`
  if (policy === 'template')   return `~${Math.max(2, Math.round(agents * rounds / 2000 * 3))}s`
  if (llm_backend === 'groq')   return `~${Math.round(agents * rounds * 0.5)}s · Groq fast inference`
  if (llm_backend === 'openai') return `~${Math.round(agents * rounds * 0.8)}s · ${llm_model_id}`
  if (llm_backend === 'ollama') return `~${Math.round(agents * rounds * 2)}s · local CPU/GPU`
  return 'GPU required — minutes to hours'
})

async function uploadEssFile(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploadStatus.value = 'uploading'
  uploadError.value = ''
  uploadInfo.value = null
  uploadFilename.value = file.name
  form.value.ess_data_file_id = ''
  try {
    const fd = new FormData()
    fd.append('file', file)
    const r = await api.uploadEssData(fd)
    form.value.ess_data_file_id = r.data.file_id
    uploadInfo.value = r.data
    uploadStatus.value = 'done'
  } catch(e) {
    uploadStatus.value = 'error'
    uploadError.value = e.response?.data?.error || 'Upload failed — check server logs.'
  }
}

async function runDesign() {
  if (!designPrompt.value.trim()) return
  designStatus.value = 'loading'
  designError.value = ''
  designResult.value = null
  designApplied.value = false
  try {
    const body = {
      prompt:      designPrompt.value,
      provider:    designProvider.value,
      file_id:     form.value.ess_data_file_id || undefined,
    }
    if (designProvider.value === 'ollama') {
      body.ollama_model = designOllamaModel.value
    }
    // Keys are always server-side — never sent from the client
    const r = await api.designSimulation(body)
    designResult.value = r.data
    designStatus.value = 'done'
  } catch(e) {
    designStatus.value = 'error'
    designError.value = e.response?.data?.error || 'Design failed — check server logs.'
  }
}

function applyDesign() {
  const d = designResult.value
  if (!d) return
  const cfg = d.config || {}
  if (cfg.agents)                    form.value.agents           = cfg.agents
  if (cfg.rounds)                    form.value.rounds           = cfg.rounds
  if (cfg.policy)                    form.value.policy           = cfg.policy
  if (cfg.network_type)              form.value.network_type     = cfg.network_type
  if (cfg.bad_apple_frac !== undefined) form.value.bad_apple_frac = cfg.bad_apple_frac
  // Use the AI-generated population parquet if available
  if (d.file_id) {
    form.value.ess_data_file_id   = d.file_id
    form.value.population_source  = 'empirical'
  }
  form.value.design_id = d.design_id || ''
  designApplied.value = true
}

function clearDesign() {
  designResult.value  = null
  designStatus.value  = ''
  designError.value   = ''
  designApplied.value = false
  form.value.design_id = ''
}

async function launch() {
  error.value = ''; launching.value = true
  try {
    const body = { ...form.value }
    delete body.llm_api_key  // keys are always server-side
    if (!selectedPolicy.value?.gpu) {
      delete body.llm_backend
      delete body.llm_model_id
    }
    if (!body.ess_data_file_id) delete body.ess_data_file_id
    if (!body.design_id)        delete body.design_id
    const r = await api.simulateWizard(body)
    launchedId.value = r.data.experiment_id
    launched.value = true
    setTimeout(() => router.push(`/monitor/${launchedId.value}`), 1200)
  } catch(e) {
    error.value = e.response?.data?.error || 'Launch failed — check server logs.'
  } finally {
    launching.value = false
  }
}

function resetForm() {
  launched.value = false; launchedId.value = ''; error.value = ''
  uploadStatus.value = ''; uploadInfo.value = null; uploadError.value = ''; uploadFilename.value = ''
  form.value.ess_data_file_id = ''
  clearDesign()
  designPrompt.value = ''
}
</script>

<style scoped>
.page-header { margin-bottom: 20px; }

/* ── Step progress ──────────────────────────────────────────────── */
.step-progress {
  display: flex; align-items: center; gap: 0;
  margin-bottom: 28px; padding: 16px 20px;
  background: rgba(13,21,40,.7); border: 1px solid var(--border);
  border-radius: 14px;
}
.sp-item { display: flex; align-items: center; flex: 1; min-width: 0; }
.sp-dot {
  width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
  background: var(--bg4); border: 1px solid var(--border2);
  color: var(--text3); font-size: .72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  transition: all .3s var(--ease-spring);
}
.sp-dot.done {
  background: linear-gradient(135deg, var(--indigo), var(--blue));
  border-color: transparent; color: #fff;
  box-shadow: 0 0 14px rgba(99,102,241,.4);
}
.sp-label {
  font-size: .72rem; color: var(--text3); margin: 0 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-weight: 500;
}
.sp-line {
  flex: 1; height: 1px; min-width: 12px;
  background: var(--border); margin: 0 4px;
}
@media (max-width: 680px) { .sp-label { display: none; } }

.wizard-layout {
  display: grid; grid-template-columns: 1fr 300px;
  gap: 22px; align-items: start;
}
@media (max-width: 900px) { .wizard-layout { grid-template-columns: 1fr; } }

/* ── Step cards ─────────────────────────────────────────────────── */
.wizard-form { display: flex; flex-direction: column; gap: 16px; }
.step-card { display: flex; flex-direction: column; gap: 18px; }

.step-head { display: flex; align-items: flex-start; gap: 14px; }
.step-num {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, var(--indigo), var(--blue));
  color: #fff;
  font-size: .78rem; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 2px 12px rgba(99,102,241,.35);
  letter-spacing: -.02em;
}
.step-title { font-size: .95rem; font-weight: 600; color: var(--text); }
.step-sub   { font-size: .78rem; color: var(--text2); margin-top: 2px; }

/* ── Policy grid ────────────────────────────────────────────────── */
.policy-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}
@media (max-width: 600px) { .policy-grid { grid-template-columns: 1fr 1fr; } }

.policy-btn {
  display: flex; flex-direction: column; gap: 5px;
  padding: 14px 12px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  text-align: left; cursor: pointer; position: relative;
  transition: border-color .2s, background .2s, transform .3s var(--ease-spring);
}
.policy-btn:hover    { border-color: rgba(99,102,241,.4); background: rgba(99,102,241,.07); transform: translateY(-2px); }
.policy-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.12); }
.policy-btn.selected::after {
  content: '✓'; position: absolute; top: 8px; right: 8px;
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--blue); color: #fff;
  font-size: .62rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.p-glyph { font-size: 1.1rem; color: var(--blue2); font-family: monospace; }
.p-name  { font-size: .84rem; font-weight: 600; color: #fff; }
.p-desc  { font-size: .73rem; color: rgba(255,255,255,.55); line-height: 1.4; }
.p-tag {
  font-size: .62rem; font-weight: 700; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 4px; width: fit-content; margin-top: 2px;
}
.p-tag.cpu { background: rgba(16,185,129,.15); color: var(--green); }
.p-tag.gpu { background: rgba(139,92,246,.15); color: var(--purple); }

/* ── LLM config ─────────────────────────────────────────────────── */
.llm-config { display: flex; flex-direction: column; gap: 12px; }
.llm-config-head { font-size: .72rem; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; }

.provider-grid {
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
@media (max-width: 520px) { .provider-grid { grid-template-columns: 1fr; } }

.provider-btn {
  display: flex; flex-direction: column; gap: 4px;
  padding: 12px 13px; border-radius: 11px;
  background: var(--bg3); border: 1px solid var(--border);
  text-align: left; cursor: pointer; position: relative;
  transition: border-color .2s, background .2s, transform .25s var(--ease-spring);
}
.provider-btn:hover    { border-color: rgba(99,102,241,.4); transform: translateY(-1px); }
.provider-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.1); }
.provider-btn.selected::after {
  content: '✓'; position: absolute; top: 7px; right: 8px;
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--blue); color: #fff;
  font-size: .58rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.prov-top  { display: flex; align-items: center; gap: 6px; margin-bottom: 2px; }
.prov-glyph { font-size: .9rem; color: var(--blue2); font-family: monospace; }
.prov-free {
  font-size: .58rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;
  background: rgba(16,185,129,.15); color: var(--green); letter-spacing: .03em;
}
.prov-name { font-size: .82rem; font-weight: 600; color: #fff; }
.prov-desc { font-size: .7rem; color: rgba(255,255,255,.5); }

.info-box {
  background: rgba(20,184,166,.06); border: 1px solid rgba(20,184,166,.18);
  border-radius: 10px; padding: 10px 14px; font-size: .8rem; color: var(--teal);
  line-height: 1.6;
}
.info-box code { background: rgba(20,184,166,.12); padding: 1px 5px; border-radius: 4px; font-size: .76rem; }
.info-box-ok {
  background: rgba(16,185,129,.07); border-color: rgba(16,185,129,.25);
  color: #6ee7b7;
}

.model-row { display: flex; gap: 8px; }
.model-btn {
  flex: 1; display: flex; flex-direction: column; gap: 2px;
  padding: 10px 12px; border-radius: 9px;
  background: var(--bg3); border: 1px solid var(--border);
  cursor: pointer; text-align: left; transition: border-color .2s;
}
.model-btn:hover    { border-color: rgba(99,102,241,.35); }
.model-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.1); }
.mb-name { font-size: .82rem; font-weight: 600; color: #fff; }
.mb-note { font-size: .7rem; color: rgba(255,255,255,.45); }

.warn-box {
  background: rgba(245,158,11,.07); border: 1px solid rgba(245,158,11,.2);
  border-radius: 10px; padding: 10px 14px; font-size: .82rem; color: var(--amber);
}

/* ── Sliders ────────────────────────────────────────────────────── */
.slider-group { display: flex; flex-direction: column; gap: 20px; }
.slider-row { display: flex; flex-direction: column; gap: 8px; }
.slider-row label {
  font-size: .82rem; font-weight: 500; color: #fff;
  display: flex; align-items: center; gap: 8px;
}
.slider-val { color: var(--blue2); font-weight: 700; }

.slider {
  -webkit-appearance: none; appearance: none;
  width: 100%; height: 4px;
  background: var(--bg4); border-radius: 2px; border: none;
  outline: none; padding: 0;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--blue); cursor: pointer;
  box-shadow: 0 2px 8px rgba(99,102,241,.4);
  transition: transform .2s var(--ease-spring), box-shadow .2s;
}
.slider::-webkit-slider-thumb:hover {
  transform: scale(1.3);
  box-shadow: 0 2px 16px rgba(99,102,241,.6);
}
.slider::-moz-range-thumb {
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--blue); border: none; cursor: pointer;
}

.slider-ticks {
  display: flex; justify-content: space-between;
  font-size: .68rem; color: var(--text3); padding: 0 2px;
}

.est-box {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 9px; padding: 10px 14px;
  font-size: .82rem; display: flex; align-items: center; gap: 10px;
}
.est-label { color: var(--text3); }
.est-val   { color: var(--teal); font-weight: 600; }

/* ── Option buttons ─────────────────────────────────────────────── */
.option-row { display: flex; gap: 10px; }
.option-btn {
  flex: 1; display: flex; flex-direction: column; gap: 5px;
  padding: 14px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  text-align: left; cursor: pointer;
  transition: border-color .2s, background .2s, transform .25s var(--ease-spring);
}
.option-btn:hover    { border-color: rgba(99,102,241,.35); transform: translateY(-2px); }
.option-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.08); }
.o-glyph { font-size: 1rem; color: var(--blue2); font-family: monospace; }
.o-name  { font-size: .84rem; font-weight: 600; color: #fff; }
.o-desc  { font-size: .73rem; color: rgba(255,255,255,.5); line-height: 1.4; }

/* ── Launch button ──────────────────────────────────────────────── */
.launch-btn {
  width: 100%; justify-content: center;
  padding: 15px; font-size: 1rem; font-weight: 700;
  border-radius: 13px; letter-spacing: -.01em;
  background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
  box-shadow: 0 4px 24px rgba(99,102,241,.4), 0 1px 0 rgba(255,255,255,.1) inset;
  transition: transform .35s var(--ease-spring), box-shadow .2s, filter .2s;
}
.launch-btn:not(:disabled):hover {
  box-shadow: 0 8px 40px rgba(99,102,241,.55), 0 1px 0 rgba(255,255,255,.12) inset;
  filter: brightness(1.1);
}
.launch-btn:disabled { opacity: .6; cursor: not-allowed; }

/* ── Sidebar: summary ───────────────────────────────────────────── */
.wizard-sidebar { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 24px; }
.summary-card { display: flex; flex-direction: column; }
.summary-rows { display: flex; flex-direction: column; gap: 0; }
.sum-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 0; border-bottom: 1px solid var(--border);
  font-size: .82rem;
}
.sum-row:last-child { border-bottom: none; }
.sum-key { color: var(--text3); }
.sum-val { color: #fff; font-weight: 500; }

/* ── Action list ────────────────────────────────────────────────── */
.action-list { display: flex; flex-direction: column; gap: 8px; }
.action-item { display: flex; align-items: baseline; gap: 8px; font-size: .82rem; }
.act-name  { color: var(--text2); min-width: 70px; }
.act-delta { font-weight: 600; min-width: 70px; font-size: .8rem; }
.act-desc  { color: var(--text3); }

/* ── Field ──────────────────────────────────────────────────────── */
.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: .76rem; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .04em; }
.hint { font-size: .72rem; color: var(--text3); }
.key-link { color: var(--blue2); margin-left: 6px; font-size: .72rem; }
.key-link:hover { color: #fff; }

/* ── ESS custom data upload ─────────────────────────────────────── */
.ess-upload {
  display: flex; flex-direction: column; gap: 10px;
  padding: 14px 16px; border-radius: 10px;
  background: rgba(99,102,241,.06); border: 1px solid rgba(99,102,241,.18);
}
.ess-upload-head {
  font-size: .78rem; font-weight: 700; color: var(--text2);
  display: flex; align-items: center; gap: 8px;
}
.optional-tag {
  font-size: .65rem; font-weight: 600; color: var(--text3);
  background: var(--bg4); border: 1px solid var(--border2);
  border-radius: 4px; padding: 1px 6px; letter-spacing: .03em;
}
.ess-upload-desc { font-size: .75rem; color: var(--text3); line-height: 1.5; }
.ess-upload-desc code { color: var(--blue2); background: rgba(99,102,241,.12); border-radius: 3px; padding: 0 4px; }

.upload-label { display: inline-block; cursor: pointer; }
.upload-label.uploading { opacity: .6; pointer-events: none; }
.upload-input { display: none; }
.upload-btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 8px 16px; border-radius: 8px;
  background: var(--bg4); border: 1px solid var(--border2);
  color: var(--text2); font-size: .78rem; font-weight: 600;
  transition: border-color .2s, background .2s;
}
.upload-label:hover .upload-btn {
  border-color: rgba(99,102,241,.5); background: rgba(99,102,241,.1); color: #fff;
}

.upload-error-msg {
  font-size: .75rem; color: var(--rose);
  background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2);
  border-radius: 6px; padding: 6px 12px;
}

/* ── Analysis report ─────────────────────────────────────────────── */
.analysis-report {
  display: flex; flex-direction: column; gap: 8px;
  animation: fade-in-up .3s var(--ease-spring) both;
}

.upload-filename-chip {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 10px; border-radius: 6px;
  background: var(--bg4); border: 1px solid var(--border2);
}
.file-icon { color: var(--blue2); font-size: .8rem; flex-shrink: 0; }
.file-name { font-size: .74rem; font-weight: 600; color: var(--text); flex: 1;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-rows { font-size: .67rem; color: var(--text3); flex-shrink: 0; }

.analysis-narrative {
  font-size: .73rem; color: var(--text2); line-height: 1.55;
  padding: 8px 12px; border-radius: 8px;
  background: rgba(99,102,241,.07); border-left: 2px solid var(--indigo);
}

.analysis-summary {
  display: flex; align-items: center; gap: 8px;
  background: rgba(16,185,129,.07); border: 1px solid rgba(16,185,129,.2);
  border-radius: 8px; padding: 8px 12px;
}
.analysis-icon { color: var(--green); font-size: .9rem; flex-shrink: 0; }
.analysis-text { font-size: .74rem; color: var(--text2); flex: 1; line-height: 1.4; }
.coverage-badge {
  font-size: .68rem; font-weight: 800; border-radius: 6px;
  padding: 2px 8px; flex-shrink: 0;
}
.cov-high { background: rgba(16,185,129,.18); color: var(--green); }
.cov-mid  { background: rgba(234,179,8,.15);  color: #eab308; }
.cov-low  { background: rgba(244,63,94,.15);  color: var(--rose); }

.dim-coverage {
  display: flex; flex-direction: column; gap: 3px;
  max-height: 220px; overflow-y: auto;
  background: var(--bg4); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px;
}
.dim-row {
  display: grid; grid-template-columns: 18px 1fr 1fr auto;
  gap: 6px; align-items: center; font-size: .72rem;
  padding: 2px 0;
}
.dim-status { font-weight: 700; text-align: center; font-size: .8rem; }
.dim-direct   { color: var(--green); }
.dim-computed { color: #eab308; }
.dim-fallback { color: var(--text3); }
.dim-name  { color: var(--text2); font-weight: 500; text-transform: capitalize; }
.dim-source{ color: var(--text3); font-size: .68rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dim-mean  { color: var(--blue2); font-size: .67rem; font-family: monospace; text-align: right; }

.quality-row {
  display: flex; gap: 8px; flex-wrap: wrap;
}
.q-item {
  display: flex; flex-direction: column; align-items: center;
  flex: 1; min-width: 60px;
  background: var(--bg4); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 8px;
}
.q-label { font-size: .62rem; color: var(--text3); text-transform: uppercase; letter-spacing: .03em; }
.q-val   { font-size: .84rem; font-weight: 700; color: var(--text); margin-top: 2px; }

/* ── Launched card ──────────────────────────────────────────────── */
.launched-card {
  text-align: center; padding: 24px;
  border-color: rgba(16,185,129,.3);
  background: rgba(16,185,129,.05);
  animation: fade-in-up .4s var(--ease-spring) both;
}
.launched-icon {
  width: 44px; height: 44px; border-radius: 50%;
  background: rgba(16,185,129,.15); color: var(--green);
  font-size: 1.4rem; display: flex; align-items: center; justify-content: center;
  margin: 0 auto 12px;
}
.launched-card h3 { font-size: 1rem; margin-bottom: 6px; color: #fff; }

/* ── AI Design card ─────────────────────────────────────────────── */
.design-card {
  border-color: rgba(99,102,241,.35);
  background: rgba(99,102,241,.04);
}
.design-num {
  background: linear-gradient(135deg, #7c3aed, var(--indigo));
  font-size: .85rem;
}
.design-textarea {
  width: 100%; padding: 12px 14px; resize: vertical;
  background: var(--bg4); border: 1px solid var(--border2);
  border-radius: 10px; color: var(--text); font-size: .84rem;
  line-height: 1.5; font-family: inherit;
  transition: border-color .2s;
}
.design-textarea:focus { outline: none; border-color: var(--indigo); }
.design-textarea::placeholder { color: var(--text3); }
.design-controls {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
.design-select {
  padding: 8px 10px; border-radius: 8px;
  background: var(--bg4); border: 1px solid var(--border2);
  color: var(--text); font-size: .8rem; min-width: 130px;
}
.design-key-wrap {
  flex: 1; display: flex; align-items: center; gap: 6px; min-width: 160px;
}
.design-key-input {
  flex: 1; padding: 8px 12px; border-radius: 8px;
  background: var(--bg4); border: 1px solid var(--border2);
  color: var(--text); font-size: .8rem;
}
.key-link-sm {
  font-size: .72rem; white-space: nowrap; color: var(--blue2);
  text-decoration: none; opacity: .8;
}
.key-link-sm:hover { opacity: 1; text-decoration: underline; }
.design-btn { white-space: nowrap; }
.design-no-key-note {
  font-size: .75rem; align-self: center;
  padding: 4px 10px; border-radius: 6px;
  background: rgba(16,185,129,.08); border: 1px solid rgba(16,185,129,.2);
  color: #6ee7b7; white-space: nowrap;
}
.design-server-badge {
  font-size: .78rem; align-self: center;
  padding: 5px 12px; border-radius: 20px;
  background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.3);
  color: var(--blue2); white-space: nowrap; flex-shrink: 0;
}

/* Design provider picker */
.design-provider-row {
  display: flex; gap: 10px; flex-wrap: wrap;
}
.dprov-btn {
  flex: 1; min-width: 90px;
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 10px 12px; border-radius: 10px; cursor: pointer;
  background: var(--card2); border: 1.5px solid rgba(255,255,255,.08);
  transition: border-color .15s, background .15s, opacity .15s;
}
.dprov-btn:hover        { border-color: rgba(99,102,241,.4); }
.dprov-btn.selected     { border-color: var(--blue); background: rgba(99,102,241,.12); }
.dprov-btn.unavailable  { opacity: .45; cursor: default; }
.dprov-glyph { font-size: 1.2rem; }
.dprov-name  { font-size: .82rem; font-weight: 700; color: #fff; }
.dprov-tag   { font-size: .68rem; border-radius: 10px; padding: 1px 7px; }
.dprov-ok    { background: rgba(16,185,129,.15); color: #6ee7b7; }
.dprov-off   { background: rgba(239,68,68,.1); color: #f87171; }

/* Ollama model chips inside design section */
.design-model-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.dm-label { font-size: .78rem; color: var(--text2); white-space: nowrap; }
.dm-chips  { display: flex; gap: 6px; flex-wrap: wrap; }
.dm-chip {
  padding: 3px 10px; border-radius: 20px; font-size: .73rem; cursor: pointer;
  background: var(--card2); border: 1.5px solid rgba(255,255,255,.08);
  color: var(--text2); transition: border-color .15s, color .15s;
}
.dm-chip:hover    { border-color: rgba(99,102,241,.4); color: #fff; }
.dm-chip.selected { border-color: var(--blue); color: #fff; background: rgba(99,102,241,.12); }
.dm-note { color: var(--text3); font-size: .65rem; margin-left: 3px; }

.design-unavail-hint {
  font-size: .76rem; color: #f87171; align-self: center;
  padding: 4px 0; max-width: 300px; line-height: 1.4;
}
.prov-status {
  font-size: .65rem; margin-top: 2px; color: var(--text3);
}
.provider-btn.unavailable { opacity: .45; }
.provider-btn.unavailable.selected { opacity: 1; }
.design-result {
  display: flex; flex-direction: column; gap: 10px;
  animation: fade-in-up .3s var(--ease-spring) both;
}
.design-title {
  font-size: .97rem; font-weight: 700; color: #fff;
  letter-spacing: -.01em;
}
.design-desc { font-size: .8rem; color: var(--text2); line-height: 1.55; }
.design-chips {
  display: flex; flex-wrap: wrap; gap: 6px;
}
.chip {
  padding: 3px 9px; border-radius: 20px; font-size: .72rem; font-weight: 600;
  background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.3);
  color: var(--blue2);
}
.chip-warn {
  background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.3);
  color: #f87171;
}
.design-narrative {
  padding: 10px 12px; border-radius: 8px;
  background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.2);
  font-size: .78rem; color: var(--text2); line-height: 1.6;
}
.narrative-label, .reasoning-label {
  display: block; font-size: .65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .06em;
  color: var(--indigo); margin-bottom: 4px;
}
.design-reasoning {
  padding: 8px 12px; border-radius: 8px;
  background: var(--bg4); border: 1px solid var(--border);
  font-size: .76rem; color: var(--text3); line-height: 1.5;
}
.design-action-row {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.design-applied-badge {
  padding: 6px 14px; border-radius: 20px; font-size: .78rem; font-weight: 600;
  background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3);
  color: var(--green);
}
</style>
