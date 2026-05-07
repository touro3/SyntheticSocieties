<template>
  <div>
    <!-- Hero header -->
    <div class="hero">
      <div class="hero-body">
        <div class="hero-badge">
          <span class="hero-dot"></span>
          Behavioral Grounding Framework
        </div>
        <h1 class="page-title">Synthetic Societies</h1>
        <p class="page-sub">LLM agents grounded in ESS Round 11 survey data — economic decisions, game theory, emergent inequality.</p>
      </div>
      <router-link to="/run" class="btn btn-primary hero-btn">
        <span>▶</span> New Simulation
      </router-link>
    </div>

    <!-- KPI row -->
    <div class="kpi-row" ref="kpiRef">
      <div class="kpi reveal" v-for="(k, i) in kpis" :key="k.label" :style="{ '--i': i }" v-tilt>
        <div class="kpi-icon-wrap" :style="{ background: k.iconBg }">
          <span class="kpi-icon" :style="{ color: k.color }">{{ k.icon }}</span>
        </div>
        <div class="kpi-val" :style="{ color: k.color }">{{ k.val }}</div>
        <div class="kpi-label">{{ k.label }}</div>
      </div>
    </div>

    <!-- Two-column layout -->
    <div class="grid-2">

      <!-- Recent experiments -->
      <div class="card">
        <div class="card-head">
          <h2 class="section-title" style="margin:0">Recent Runs</h2>
          <router-link to="/experiments" class="btn btn-ghost btn-sm">View all →</router-link>
        </div>

        <div v-if="loading" class="empty"><span class="spin">⟳</span></div>
        <div v-else-if="!experiments.length" class="empty">
          No runs yet. <router-link to="/run">Launch one →</router-link>
        </div>
        <div v-else class="exp-list">
          <div v-for="e in experiments.slice(0,8)" :key="e.experiment_id"
            class="exp-item" @click="$router.push(`/results/${e.experiment_id}`)">
            <div class="exp-left">
              <div class="exp-id mono">{{ e.experiment_id }}</div>
              <div class="exp-badges">
                <span class="badge badge-teal" style="font-size:.65rem">{{ e.policy_type || '—' }}</span>
                <span class="exp-gini" :style="giniColor(e.gini)">Gini {{ fmt(e.gini) }}</span>
              </div>
            </div>
            <StatusBadge :status="e.status || 'complete'" />
          </div>
        </div>
      </div>

      <!-- Right column -->
      <div class="right-col">

        <!-- Quick launch -->
        <div class="card card-glow">
          <h2 class="section-title">Quick Launch</h2>
          <p style="font-size:.82rem;color:var(--text2);margin-bottom:16px;line-height:1.5">
            CPU-only policies run in seconds — no GPU or API key needed.
          </p>
          <div class="quick-list">
            <button v-for="q in quickLaunches" :key="q.label"
              class="quick-btn" :class="{ loading: q.loading, done: q.launched }"
              @click="quickRun(q)">
              <span class="q-icon" :style="{ color: q.color }">{{ q.icon }}</span>
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
            <div class="feat-icon-wrap" :style="{ background: f.iconBg }">
              <span class="feat-icon" :style="{ color: f.color }">{{ f.icon }}</span>
            </div>
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
      <h2 class="section-title">API Reference</h2>
      <div class="api-grid" ref="apiRef">
        <div class="api-item reveal" v-for="(ep, i) in endpoints" :key="ep.path" :style="{ '--i': i }">
          <span class="method" :class="ep.method.toLowerCase()">{{ ep.method }}</span>
          <code class="mono api-path">{{ ep.path }}</code>
          <span class="api-desc">{{ ep.desc }}</span>
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

const router  = useRouter()
const kpiRef  = ref(null)
const featRef = ref(null)
const apiRef  = ref(null)
useReveal(kpiRef)
useReveal(featRef)
useReveal(apiRef)

const loading     = ref(true)
const experiments = ref([])
const online      = ref(false)
const launchErr   = ref('')

const kpis = computed(() => [
  { icon: '●', label: 'API',          val: online.value ? 'Online' : 'Offline',  color: online.value ? 'var(--green)' : 'var(--rose)', iconBg: online.value ? 'rgba(16,185,129,.12)' : 'rgba(244,63,94,.1)' },
  { icon: '◫', label: 'Experiments',  val: experiments.value.length || '—',       color: 'var(--blue2)',  iconBg: 'rgba(99,102,241,.12)' },
  { icon: '⬡', label: 'Policy Types', val: 6,                                      color: 'var(--violet)', iconBg: 'rgba(139,92,246,.12)' },
  { icon: '◉', label: 'Data Source',  val: 'ESS R11',                              color: 'var(--cyan)',   iconBg: 'rgba(34,211,238,.1)'  },
])

const quickLaunches = ref([
  { label: 'Rule-based · 20 agents · 10 rounds', desc: 'CPU · ~5s', icon: '◆', color: 'var(--blue2)',  policy: 'rule_based', agents: 20,  rounds: 10, loading: false, launched: false },
  { label: 'Random · 50 agents · 20 rounds',     desc: 'CPU · ~8s', icon: '◈', color: 'var(--violet)', policy: 'random',     agents: 50,  rounds: 20, loading: false, launched: false },
  { label: 'Mock · 100 agents · 5 rounds',       desc: 'CPU · ~2s', icon: '◻', color: 'var(--teal)',   policy: 'mock',       agents: 100, rounds: 5,  loading: false, launched: false },
])

const features = [
  { icon: '◉', title: 'ESS Grounding',      desc: 'Agents sampled from ESS Round 11 microdata.', color: 'var(--cyan)',   iconBg: 'rgba(34,211,238,.1)'  },
  { icon: '◆', title: 'Gini + 20 Metrics',  desc: 'Inequality, BRM, trust gradient, fidelity.',  color: 'var(--blue2)', iconBg: 'rgba(99,102,241,.12)' },
  { icon: '⬡', title: 'Network Topologies', desc: 'Watts-Strogatz and Erdős–Rényi graphs.',       color: 'var(--violet)',iconBg: 'rgba(139,92,246,.1)'  },
  { icon: '▲', title: 'Live Injection',      desc: 'Inject shocks, signals, or narratives live.',  color: 'var(--amber)', iconBg: 'rgba(245,158,11,.1)'  },
]

const endpoints = [
  { method: 'POST', path: '/simulate-wizard', desc: 'No-code launch with wizard params' },
  { method: 'POST', path: '/simulate',        desc: 'Launch from YAML config file' },
  { method: 'GET',  path: '/status/{exp_id}', desc: 'Poll run progress' },
  { method: 'GET',  path: '/results/{exp_id}',desc: 'Wealth distribution + metrics' },
  { method: 'POST', path: '/inject/{exp_id}', desc: 'Inject exogenous event into running sim' },
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
/* ── Hero header ────────────────────────────────────────────────── */
.hero {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 20px; flex-wrap: wrap; margin-bottom: 32px;
  padding: 28px 32px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(79,70,229,.08) 0%, rgba(139,92,246,.06) 50%, rgba(34,211,238,.04) 100%);
  border: 1px solid rgba(99,102,241,.14);
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px; border-radius: 50%;
  background: radial-gradient(circle, rgba(99,102,241,.12), transparent);
  pointer-events: none;
}
.hero-body { max-width: 600px; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--blue2); margin-bottom: 12px;
  padding: 4px 12px; border-radius: 99px;
  background: rgba(99,102,241,.1); border: 1px solid rgba(99,102,241,.2);
}
.hero-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan); flex-shrink: 0;
  box-shadow: 0 0 8px var(--cyan);
  animation: pulse 2s infinite;
}
.hero-btn { flex-shrink: 0; align-self: flex-start; margin-top: 8px; }

/* ── KPI row ─────────────────────────────────────────────────────── */
.kpi-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 14px; margin-bottom: 26px;
}
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .kpi-row { grid-template-columns: 1fr 1fr; } }

.kpi {
  background: rgba(13,21,40,.85);
  backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 20px;
  display: flex; flex-direction: column; gap: 12px;
  transform-style: preserve-3d;
  transition: border-color .25s, box-shadow .25s;
}
.kpi:hover { border-color: rgba(99,102,241,.25); box-shadow: var(--glow); }
.kpi-icon-wrap {
  width: 38px; height: 38px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.kpi-icon  { font-size: .85rem; }
.kpi-val   { font-size: 1.6rem; font-weight: 800; line-height: 1; letter-spacing: -.02em; }
.kpi-label { font-size: .68rem; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }

/* ── Grid layout ────────────────────────────────────────────────── */
.grid-2 {
  display: grid; grid-template-columns: 1fr 360px;
  gap: 20px; align-items: start;
}
@media (max-width: 1020px) { .grid-2 { grid-template-columns: 1fr; } }

/* ── Experiment list ────────────────────────────────────────────── */
.card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }

.exp-list { display: flex; flex-direction: column; }
.exp-item {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 11px 8px; border-bottom: 1px solid var(--border);
  cursor: pointer;
  border-radius: 8px; margin: 0 -8px;
  transition: background .15s, padding-left .2s var(--ease-spring);
}
.exp-item:hover { background: rgba(99,102,241,.05); padding-left: 14px; }
.exp-item:last-child { border-bottom: none; }
.exp-left { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.exp-id { font-size: .8rem; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.exp-badges { display: flex; align-items: center; gap: 8px; }
.exp-gini { font-size: .71rem; font-weight: 600; }

/* ── Right column ───────────────────────────────────────────────── */
.right-col { display: flex; flex-direction: column; gap: 18px; }

/* ── Quick launch ───────────────────────────────────────────────── */
.quick-list { display: flex; flex-direction: column; gap: 8px; }
.quick-btn {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 14px; border-radius: 11px;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text); text-align: left; width: 100%;
  transition: border-color .2s, background .2s, transform .3s var(--ease-spring);
}
.quick-btn:hover  { border-color: rgba(99,102,241,.3); background: rgba(99,102,241,.06); transform: translateX(4px); }
.quick-btn.loading { opacity: .7; pointer-events: none; }
.quick-btn.done { border-color: rgba(16,185,129,.35); background: rgba(16,185,129,.05); }
.q-icon  { font-size: 1.1rem; flex-shrink: 0; width: 22px; text-align: center; }
.q-text  { flex: 1; display: flex; flex-direction: column; gap: 1px; }
.q-label { font-size: .82rem; font-weight: 600; color: var(--text); }
.q-desc  { font-size: .7rem; color: var(--text3); }
.q-arrow { font-size: .8rem; color: var(--text3); flex-shrink: 0; transition: transform .2s; }
.quick-btn:hover .q-arrow { color: var(--blue2); transform: translateX(3px); }
.q-check { font-size: .8rem; color: var(--green); flex-shrink: 0; }
.q-spin  { flex-shrink: 0; color: var(--blue); }

/* ── Feature grid ───────────────────────────────────────────────── */
.feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.feat-card { padding: 15px; display: flex; gap: 12px; align-items: flex-start; }
.feat-icon-wrap {
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.feat-icon  { font-size: .85rem; }
.feat-title { font-size: .82rem; font-weight: 700; margin-bottom: 4px; color: var(--text); }
.feat-desc  { font-size: .73rem; color: var(--text2); line-height: 1.5; }

/* ── API reference ──────────────────────────────────────────────── */
.api-ref { padding: 22px; }
.api-grid { display: grid; gap: 6px; }
.api-item {
  display: grid; grid-template-columns: 64px 220px 1fr;
  align-items: center; gap: 14px;
  padding: 9px 14px; border-radius: 10px;
  transition: background .15s, transform .25s var(--ease-spring);
}
.api-item.revealed:hover { background: rgba(99,102,241,.05); transform: translateX(3px); }
.method {
  font-size: .67rem; font-weight: 700; padding: 3px 9px; border-radius: 5px;
  text-align: center; letter-spacing: .05em;
}
.method.get  { background: rgba(16,185,129,.1); color: var(--green); border: 1px solid rgba(16,185,129,.15); }
.method.post { background: rgba(99,102,241,.1); color: var(--blue2); border: 1px solid rgba(99,102,241,.15); }
.api-path    { font-size: .79rem; color: var(--text); }
.api-desc    { font-size: .77rem; color: var(--text2); }

@media (max-width: 640px) {
  .api-item { grid-template-columns: 64px 1fr; }
  .api-desc { display: none; }
  .hero { padding: 22px 20px; }
}
</style>
