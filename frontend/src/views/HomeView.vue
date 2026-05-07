<template>
  <div>
    <!-- Page header -->
    <div class="page-header">
      <div>
        <h1 class="page-title">Dashboard</h1>
        <p class="page-sub">Behavioral Grounding Framework — LLM agents grounded in ESS Round 11 survey data.</p>
      </div>
      <router-link to="/run" class="btn btn-primary">
        <span>▶</span> New Simulation
      </router-link>
    </div>

    <!-- KPI row -->
    <div class="kpi-row" ref="kpiRef">
      <div class="kpi reveal" v-for="(k, i) in kpis" :key="k.label" :style="{ '--i': i }" v-tilt>
        <div class="kpi-icon">{{ k.icon }}</div>
        <div class="kpi-val" :style="{ color: k.color }">{{ k.val }}</div>
        <div class="kpi-label">{{ k.label }}</div>
      </div>
    </div>

    <!-- Two-column layout -->
    <div class="grid-2">

      <!-- Recent experiments -->
      <div class="card exp-card">
        <div class="card-head">
          <h2 class="section-title" style="margin:0">Recent Runs</h2>
          <router-link to="/experiments" class="btn btn-ghost btn-sm">View all →</router-link>
        </div>

        <div v-if="loading" class="empty"><span class="spin">⟳</span></div>
        <div v-else-if="!experiments.length" class="empty">
          No runs yet. <router-link to="/run">Launch one →</router-link>
        </div>
        <div v-else class="exp-list">
          <div v-for="e in experiments.slice(0,7)" :key="e.experiment_id"
            class="exp-item" @click="$router.push(`/results/${e.experiment_id}`)">
            <div class="exp-id mono">{{ e.experiment_id }}</div>
            <div class="exp-meta">
              <span class="badge badge-teal" style="font-size:.67rem">{{ e.policy_type || '—' }}</span>
              <span class="exp-gini" :style="giniColor(e.gini)">Gini {{ fmt(e.gini) }}</span>
              <StatusBadge :status="e.status || 'complete'" />
            </div>
          </div>
        </div>
      </div>

      <!-- Right column -->
      <div class="right-col">

        <!-- Quick launch -->
        <div class="card quick-card">
          <h2 class="section-title">Quick Launch</h2>
          <p style="font-size:.82rem;color:var(--text2);margin-bottom:16px">
            CPU-only policies run in seconds — no GPU needed.
          </p>
          <div class="quick-list">
            <button v-for="q in quickLaunches" :key="q.label"
              class="quick-btn" :class="{ loading: q.loading }"
              @click="quickRun(q)">
              <span class="q-icon">{{ q.icon }}</span>
              <span class="q-text">
                <span class="q-label">{{ q.label }}</span>
                <span class="q-desc">{{ q.desc }}</span>
              </span>
              <span v-if="q.loading" class="spin q-spin">⟳</span>
              <span v-else-if="q.launched" class="q-check">✓</span>
              <span v-else class="q-arrow">→</span>
            </button>
          </div>
          <div v-if="launchErr" class="error-box" style="margin-top:12px">{{ launchErr }}</div>
        </div>

        <!-- Feature cards -->
        <div class="feat-grid" ref="featRef">
          <div class="feat-card card reveal" v-for="(f, i) in features" :key="f.title" :style="{ '--i': i }">
            <div class="feat-icon">{{ f.icon }}</div>
            <div class="feat-body">
              <div class="feat-title">{{ f.title }}</div>
              <div class="feat-desc">{{ f.desc }}</div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- API reference strip -->
    <div class="card api-ref" style="margin-top:24px">
      <h2 class="section-title">API Endpoints</h2>
      <div class="api-grid" ref="apiRef">
        <div class="api-item reveal" v-for="(ep, i) in endpoints" :key="ep.path" :style="{ '--i': i }">
          <span class="method" :class="ep.method.toLowerCase()">{{ ep.method }}</span>
          <code class="mono" style="font-size:.8rem">{{ ep.path }}</code>
          <span style="font-size:.78rem;color:var(--text2)">{{ ep.desc }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'
import { useReveal } from '../composables/useReveal.js'
import StatusBadge from '../components/StatusBadge.vue'

const router   = useRouter()
const kpiRef   = ref(null)
const featRef  = ref(null)
const apiRef   = ref(null)
useReveal(kpiRef)
useReveal(featRef)
useReveal(apiRef)

const loading     = ref(true)
const experiments = ref([])
const online      = ref(false)
const launchErr   = ref('')

const kpis = computed(() => [
  { icon: '●', label: 'API',         val: online.value ? 'Online' : 'Offline', color: online.value ? 'var(--green)' : 'var(--rose)' },
  { icon: '◫', label: 'Experiments', val: experiments.value.length || '—',     color: 'var(--blue2)' },
  { icon: '⬡', label: 'Policy Types',val: 6,                                    color: 'var(--purple)' },
  { icon: '◉', label: 'Data Source', val: 'ESS R11',                            color: 'var(--teal)' },
])

const quickLaunches = ref([
  { label: 'Rule-based · 20 agents · 10 rounds', desc: 'CPU · ~5s', icon: '⚡', policy: 'rule_based', agents: 20, rounds: 10, loading: false, launched: false },
  { label: 'Random policy · 50 agents · 20 rounds', desc: 'CPU · ~8s', icon: '🎲', policy: 'random', agents: 50, rounds: 20, loading: false, launched: false },
  { label: 'Mock policy · 100 agents · 5 rounds', desc: 'CPU · ~2s', icon: '🤖', policy: 'mock', agents: 100, rounds: 5, loading: false, launched: false },
])

const features = [
  { icon: '📊', title: 'ESS Grounding',     desc: 'Agents sampled from ESS Round 11 microdata — real European distributions.' },
  { icon: '⚖️', title: 'Gini + 20 Metrics', desc: 'Gini coefficient, BRM, persona fidelity, trust gradient, phase transitions.' },
  { icon: '🌐', title: 'Network Topologies', desc: 'Watts-Strogatz small-world and Erdős–Rényi random graphs.' },
  { icon: '💉', title: 'Live Injection',     desc: 'Inject wealth shocks, signals, or narratives into running simulations.' },
]

const endpoints = [
  { method: 'POST', path: '/simulate-wizard', desc: 'No-code launch with wizard params' },
  { method: 'POST', path: '/simulate',        desc: 'Launch from YAML config file' },
  { method: 'GET',  path: '/status/{exp_id}', desc: 'Poll run progress' },
  { method: 'GET',  path: '/results/{exp_id}',desc: 'Wealth distribution + metrics' },
  { method: 'POST', path: '/inject/{exp_id}', desc: 'Inject exogenous event' },
  { method: 'GET',  path: '/report?q={query}',desc: 'ReACT natural-language analysis' },
]

const fmt  = v => (v != null) ? Number(v).toFixed(3) : '—'
const giniColor = g => {
  if (g == null) return {}
  return { color: `hsl(${Math.round((1-g)*120)},65%,58%)` }
}

async function quickRun(q) {
  q.loading = true; q.launched = false; launchErr.value = ''
  try {
    const r = await api.simulateWizard({ policy: q.policy, agents: q.agents, rounds: q.rounds, seed: 42 })
    q.launched = true
    setTimeout(() => router.push(`/monitor/${r.data.experiment_id}`), 800)
  } catch(e) {
    launchErr.value = e.response?.data?.error || 'Launch failed — check server logs.'
  } finally { q.loading = false }
}

onMounted(async () => {
  try { await api.health(); online.value = true } catch {}
  try {
    const r = await api.experiments()
    experiments.value = r.data.experiments || []
  } catch {}
  loading.value = false
})
</script>

<style scoped>
.page-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 20px; flex-wrap: wrap; margin-bottom: 28px;
}

/* ── KPI row ─────────────────────────────────────────────────────── */
.kpi-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 14px; margin-bottom: 24px;
}
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .kpi-row { grid-template-columns: 1fr 1fr; } }

.kpi {
  background: rgba(22,28,48,.8);
  backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 18px;
  text-align: center;
  transform-style: preserve-3d;
  transition: border-color .25s, box-shadow .25s;
}
.kpi:hover { border-color: rgba(99,102,241,.28); box-shadow: var(--glow); }
.kpi-icon  { font-size: .8rem; color: var(--text3); margin-bottom: 6px; }
.kpi-val   { font-size: 1.45rem; font-weight: 700; }
.kpi-label { font-size: .7rem; color: var(--text3); text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }

/* ── Grid layout ────────────────────────────────────────────────── */
.grid-2 {
  display: grid; grid-template-columns: 1fr 360px;
  gap: 20px; align-items: start;
}
@media (max-width: 1000px) { .grid-2 { grid-template-columns: 1fr; } }

/* ── Exp card ───────────────────────────────────────────────────── */
.exp-card { display: flex; flex-direction: column; gap: 0; padding: 20px; }
.card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }

.exp-list { display: flex; flex-direction: column; }
.exp-item {
  display: flex; flex-direction: column; gap: 6px;
  padding: 12px 0; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: background .15s, padding-left .2s var(--ease-spring);
  border-radius: 0; margin: 0 -4px; padding-left: 4px; padding-right: 4px;
}
.exp-item:hover { background: rgba(99,102,241,.05); padding-left: 10px; }
.exp-item:last-child { border-bottom: none; }
.exp-id { font-size: .82rem; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.exp-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.exp-gini { font-size: .73rem; font-weight: 600; }

/* ── Right column ───────────────────────────────────────────────── */
.right-col { display: flex; flex-direction: column; gap: 18px; }

/* ── Quick launch ───────────────────────────────────────────────── */
.quick-card { display: flex; flex-direction: column; }
.quick-list { display: flex; flex-direction: column; gap: 8px; }
.quick-btn {
  display: flex; align-items: center; gap: 12px;
  padding: 11px 14px; border-radius: 10px;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text); text-align: left; width: 100%;
  transition: border-color .2s, background .2s, transform .3s var(--ease-spring);
}
.quick-btn:hover { border-color: rgba(99,102,241,.35); background: rgba(99,102,241,.06); transform: translateX(3px); }
.quick-btn.loading { opacity: .7; pointer-events: none; }
.q-icon { font-size: 1rem; flex-shrink: 0; }
.q-text { flex: 1; display: flex; flex-direction: column; gap: 1px; }
.q-label { font-size: .82rem; font-weight: 500; }
.q-desc  { font-size: .72rem; color: var(--text3); }
.q-arrow, .q-check { font-size: .8rem; color: var(--text3); flex-shrink: 0; }
.q-check { color: var(--green); }
.q-spin  { flex-shrink: 0; color: var(--blue); }

/* ── Feature grid ───────────────────────────────────────────────── */
.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.feat-card { padding: 14px; display: flex; gap: 10px; align-items: flex-start; }
.feat-icon { font-size: 1.2rem; flex-shrink: 0; }
.feat-title { font-size: .82rem; font-weight: 600; margin-bottom: 3px; }
.feat-desc  { font-size: .75rem; color: var(--text2); line-height: 1.5; }

/* ── API reference ──────────────────────────────────────────────── */
.api-ref { padding: 20px; }
.api-grid { display: grid; gap: 8px; }
.api-item {
  display: grid; grid-template-columns: 60px 220px 1fr;
  align-items: center; gap: 12px;
  padding: 9px 14px; border-radius: 9px;
  transition: background .15s, transform .25s var(--ease-spring);
}
.api-item.revealed:hover { background: rgba(99,102,241,.05); transform: translateX(3px); }
.method { font-size: .68rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; text-align: center; letter-spacing: .04em; }
.method.get  { background: rgba(16,185,129,.13); color: var(--green); }
.method.post { background: rgba(99,102,241,.13); color: var(--blue2); }

@media (max-width: 640px) {
  .api-item { grid-template-columns: 60px 1fr; }
  .api-item span:last-child { display: none; }
}
</style>
