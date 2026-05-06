<template>
  <div class="container">
    <div class="page-header">
      <router-link to="/" class="back">← Dashboard</router-link>
      <h1 class="page-title">Run Simulation</h1>
      <p class="page-sub">Configure and launch a BGF simulation. Cloud runs use CPU-only policies (no GPU required).</p>
    </div>

    <div class="layout">
      <!-- Form -->
      <div class="card form-card">
        <h2 class="section-title">Configuration</h2>

        <div class="field">
          <label>Config preset</label>
          <select v-model="form.config_path">
            <option value="configs/demo_cloud.yaml">demo_cloud — rule_based · 10 rounds · 20 agents (Render-safe)</option>
            <option value="configs/base_config.yaml">base_config — LLM · 30 rounds · 100 agents (GPU required)</option>
            <option value="configs/condition_c.yaml">condition_c — generative agents (LLM)</option>
            <option v-for="c in extraConfigs" :key="c" :value="c">{{ c }}</option>
          </select>
          <span class="hint">Preset determines policy type, agent count, and rounds.</span>
        </div>

        <div class="field-row">
          <div class="field">
            <label>Override experiment ID <span class="optional">(optional)</span></label>
            <input v-model="form.override_id" placeholder="my_run_001" />
            <span class="hint">Leave blank to use the preset's default ID.</span>
          </div>
        </div>

        <div class="cloud-note card-note">
          <strong>☁️ Cloud (Render free tier)</strong><br>
          CPU-only — only <code>demo_cloud.yaml</code> is guaranteed to work. Simulations with LLM policies require local GPU deployment.
        </div>

        <div v-if="error" class="error-msg">{{ error }}</div>

        <button class="btn btn-primary launch-btn" @click="launch" :disabled="launching">
          <span v-if="launching" class="spin">⟳</span>
          {{ launching ? 'Launching…' : 'Launch Simulation' }}
        </button>
      </div>

      <!-- Info panel -->
      <div class="info-panel">
        <div class="card info-card">
          <h3>Policy types</h3>
          <ul>
            <li v-for="p in policies" :key="p.name">
              <strong>{{ p.name }}</strong> — {{ p.desc }}
              <span class="badge badge-green" v-if="p.cpu" style="font-size:.65rem;margin-left:4px">CPU</span>
              <span class="badge badge-rose"  v-else          style="font-size:.65rem;margin-left:4px">GPU</span>
            </li>
          </ul>
        </div>
        <div class="card info-card">
          <h3>Economic actions</h3>
          <ul>
            <li><strong>work</strong> — +8 wealth</li>
            <li><strong>save</strong> — +4 wealth</li>
            <li><strong>cooperate</strong> — −3 wealth, generates +12 shared pool</li>
            <li><strong>steal</strong> — takes 50% of public pool (bad apples only)</li>
          </ul>
        </div>
        <div class="card info-card" v-if="launched">
          <h3>Launched ✓</h3>
          <p>Experiment ID: <code class="mono">{{ launchedId }}</code></p>
          <router-link :to="`/monitor/${launchedId}`" class="btn btn-primary" style="margin-top:12px;display:inline-flex">
            Monitor →
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'

const router = useRouter()

const form = ref({ config_path: 'configs/demo_cloud.yaml', override_id: '' })
const launching  = ref(false)
const error      = ref('')
const launched   = ref(false)
const launchedId = ref('')
const extraConfigs = ref([])

const policies = [
  { name: 'mock',       desc: 'Returns fixed action — fastest, for testing',    cpu: true },
  { name: 'random',     desc: 'Uniformly random action each round',              cpu: true },
  { name: 'rule_based', desc: 'Heuristic rules based on wealth and stress',      cpu: true },
  { name: 'template',   desc: 'Template prompt, no LLM — deterministic',        cpu: true },
  { name: 'llm',        desc: 'Mistral-7B or GPT via API — grounded reasoning', cpu: false },
  { name: 'generative', desc: 'Park et al. 2023 proxy — fictional persona',      cpu: false },
]

onMounted(async () => {
  try {
    const r = await api.configs()
    extraConfigs.value = (r.data.configs || []).filter(
      c => !['configs/demo_cloud.yaml','configs/base_config.yaml','configs/condition_c.yaml'].includes(c)
    )
  } catch {}
})

async function launch() {
  error.value = ''
  launching.value = true
  try {
    const body = { config_path: form.value.config_path }
    const r = await api.simulate(body)
    launchedId.value = r.data.experiment_id
    launched.value = true
    setTimeout(() => router.push(`/monitor/${launchedId.value}`), 1200)
  } catch (e) {
    error.value = e.response?.data?.error || 'Launch failed — check server logs.'
  } finally {
    launching.value = false
  }
}
</script>

<style scoped>
.page-header { margin-bottom: 32px; }
.back { color: var(--text3); font-size: .84rem; display: inline-block; margin-bottom: 12px; }
.back:hover { color: var(--text); text-decoration: none; }
.page-title { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
.page-sub { color: var(--text2); font-size: .95rem; }

.layout { display: grid; grid-template-columns: 1fr 340px; gap: 24px; align-items: start; }
@media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }

.form-card { display: flex; flex-direction: column; gap: 20px; }

.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: .82rem; font-weight: 500; color: var(--text2); }
.field .hint { font-size: .75rem; color: var(--text3); }
.optional { font-weight: 400; color: var(--text3); }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 500px) { .field-row { grid-template-columns: 1fr; } }

.cloud-note {
  background: rgba(245,158,11,.06); border: 1px solid rgba(245,158,11,.2);
  border-radius: 10px; padding: 14px 16px;
  font-size: .82rem; color: var(--text2); line-height: 1.6;
}
.cloud-note code { color: var(--amber); }

.error-msg { background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2); border-radius: 8px; padding: 10px 14px; font-size: .84rem; color: var(--rose); }

.launch-btn { width: 100%; justify-content: center; padding: 13px; font-size: .95rem; }

.info-panel { display: flex; flex-direction: column; gap: 16px; }
.info-card h3 { font-size: .9rem; margin-bottom: 10px; }
.info-card ul { list-style: none; display: flex; flex-direction: column; gap: 6px; }
.info-card li { font-size: .82rem; color: var(--text2); line-height: 1.5; }
.info-card li strong { color: var(--text); }
.info-card p { font-size: .84rem; color: var(--text2); margin-bottom: 4px; }
</style>
