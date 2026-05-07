<template>
  <div>
    <div class="page-header">
      <div>
        <h1 class="page-title">Run Simulation</h1>
        <p class="page-sub">Configure your simulation with the visual builder — no YAML or code needed.</p>
      </div>
    </div>

    <div class="wizard-layout">

      <!-- ── LEFT: Wizard form ─────────────────────────────────────── -->
      <div class="wizard-form">

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
              <span class="p-icon">{{ p.icon }}</span>
              <span class="p-name">{{ p.name }}</span>
              <span class="p-desc">{{ p.desc }}</span>
              <span class="p-tag" :class="p.gpu ? 'gpu' : 'cpu'">{{ p.gpu ? 'GPU' : 'CPU' }}</span>
            </button>
          </div>
          <div v-if="selectedPolicy?.gpu" class="warn-box">
            ⚠ LLM policies require a local GPU deployment. Cloud (Render) supports CPU policies only.
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
              <span class="o-icon">{{ n.icon }}</span>
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
              <span class="o-icon">{{ s.icon }}</span>
              <span class="o-name">{{ s.name }}</span>
              <span class="o-desc">{{ s.desc }}</span>
            </button>
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
          <h3>Simulation Started!</h3>
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
import { ref, computed } from 'vue'
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
})

const launching   = ref(false)
const error       = ref('')
const launched    = ref(false)
const launchedId  = ref('')

const policies = [
  { value: 'mock',       icon: '🤖', name: 'Mock',        desc: 'Fixed action every round — fastest',       gpu: false },
  { value: 'random',     icon: '🎲', name: 'Random',      desc: 'Uniform random action each round',         gpu: false },
  { value: 'rule_based', icon: '⚖️', name: 'Rule-Based',  desc: 'Heuristic rules on wealth + stress',       gpu: false },
  { value: 'template',   icon: '📋', name: 'Template',    desc: 'Template prompt, no LLM — deterministic',  gpu: false },
  { value: 'llm',        icon: '🧠', name: 'LLM',         desc: 'Mistral-7B or GPT — grounded reasoning',   gpu: true  },
  { value: 'generative_agents', icon: '🌐', name: 'Generative', desc: 'Park et al. 2023 fictional persona', gpu: true  },
]

const networks = [
  { value: 'random',      icon: '◌', name: 'Erdős–Rényi',     desc: 'Random edges — each pair connected with prob p' },
  { value: 'small_world', icon: '⬡', name: 'Watts-Strogatz',  desc: 'Small-world — high clustering + short paths' },
]

const sources = [
  { value: 'synthetic',  icon: '⚙',  name: 'Synthetic',  desc: 'Default demographic profiles' },
  { value: 'empirical',  icon: '📊', name: 'Empirical',   desc: 'Sampled from ESS Round 11 microdata' },
]

const actions = [
  { name: 'work',      delta: '+8 wealth',  desc: 'Earn from employment',            color: 'var(--green)' },
  { name: 'save',      delta: '+4 wealth',  desc: 'Accumulate savings',              color: 'var(--teal)' },
  { name: 'cooperate', delta: '−3 + pool',  desc: 'Lose 3, generate +12 shared',     color: 'var(--blue2)' },
  { name: 'steal',     delta: '50% pool',   desc: 'Bad apples only — drains pool',   color: 'var(--rose)' },
]

const selectedPolicy = computed(() => policies.find(p => p.value === form.value.policy))

const estimatedTime = computed(() => {
  const { policy, agents, rounds } = form.value
  if (policy === 'mock')       return `~${Math.round(agents * rounds / 10000 * 2 + 1)}s`
  if (policy === 'random')     return `~${Math.round(agents * rounds / 5000 * 2 + 1)}s`
  if (policy === 'rule_based') return `~${Math.round(agents * rounds / 3000 * 2 + 2)}s`
  if (policy === 'template')   return `~${Math.round(agents * rounds / 2000 * 3 + 3)}s`
  return 'GPU required — minutes to hours'
})

async function launch() {
  error.value = ''; launching.value = true
  try {
    const r = await api.simulateWizard({ ...form.value })
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
}
</script>

<style scoped>
.page-header { margin-bottom: 28px; }

.wizard-layout {
  display: grid; grid-template-columns: 1fr 300px;
  gap: 22px; align-items: start;
}
@media (max-width: 900px) { .wizard-layout { grid-template-columns: 1fr; } }

/* ── Step cards ─────────────────────────────────────────────────── */
.wizard-form { display: flex; flex-direction: column; gap: 16px; }
.step-card { display: flex; flex-direction: column; gap: 18px; }

.step-head {
  display: flex; align-items: flex-start; gap: 14px;
}
.step-num {
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--blue); color: #fff;
  font-size: .8rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.step-title { font-size: .95rem; font-weight: 600; }
.step-sub   { font-size: .78rem; color: var(--text2); margin-top: 2px; }

/* ── Policy grid ────────────────────────────────────────────────── */
.policy-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}
@media (max-width: 600px) { .policy-grid { grid-template-columns: 1fr 1fr; } }

.policy-btn {
  display: flex; flex-direction: column; gap: 4px;
  padding: 14px 12px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  text-align: left; cursor: pointer; position: relative;
  transition: border-color .2s, background .2s, transform .3s var(--ease-spring);
}
.policy-btn:hover { border-color: rgba(99,102,241,.35); background: rgba(99,102,241,.06); transform: translateY(-2px); }
.policy-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.1); }
.policy-btn.selected::after {
  content: '✓'; position: absolute; top: 8px; right: 8px;
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--blue); color: #fff;
  font-size: .62rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.p-icon { font-size: 1.2rem; }
.p-name { font-size: .84rem; font-weight: 600; }
.p-desc { font-size: .73rem; color: var(--text2); line-height: 1.4; }
.p-tag {
  font-size: .62rem; font-weight: 700; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 4px; width: fit-content; margin-top: 2px;
}
.p-tag.cpu { background: rgba(16,185,129,.12); color: var(--green); }
.p-tag.gpu { background: rgba(139,92,246,.12); color: var(--purple); }

.warn-box {
  background: rgba(245,158,11,.07); border: 1px solid rgba(245,158,11,.2);
  border-radius: 10px; padding: 10px 14px; font-size: .82rem; color: var(--amber);
}

/* ── Sliders ────────────────────────────────────────────────────── */
.slider-group { display: flex; flex-direction: column; gap: 20px; }
.slider-row { display: flex; flex-direction: column; gap: 8px; }
.slider-row label {
  font-size: .82rem; font-weight: 500; color: var(--text2);
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
  flex: 1; display: flex; flex-direction: column; gap: 4px;
  padding: 14px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  text-align: left; cursor: pointer;
  transition: border-color .2s, background .2s, transform .25s var(--ease-spring);
}
.option-btn:hover { border-color: rgba(99,102,241,.35); transform: translateY(-2px); }
.option-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.08); }
.o-icon { font-size: 1.1rem; }
.o-name { font-size: .84rem; font-weight: 600; }
.o-desc { font-size: .73rem; color: var(--text2); line-height: 1.4; }

/* ── Launch button ──────────────────────────────────────────────── */
.launch-btn {
  width: 100%; justify-content: center;
  padding: 14px; font-size: 1rem;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(99,102,241,.3);
}

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
.sum-val { color: var(--text); font-weight: 500; }

/* ── Action list ────────────────────────────────────────────────── */
.action-list { display: flex; flex-direction: column; gap: 8px; }
.action-item { display: flex; align-items: baseline; gap: 8px; font-size: .82rem; }
.act-name  { color: var(--text2); min-width: 70px; }
.act-delta { font-weight: 600; min-width: 70px; font-size: .8rem; }
.act-desc  { color: var(--text3); }

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
.launched-card h3 { font-size: 1rem; margin-bottom: 6px; }
</style>
