<template>
  <div class="container">

    <!-- ── Hero (scroll-driven parallax) ──────────────────────────────── -->
    <header class="hero" ref="heroRef">
      <div class="hero-badge badge badge-blue">
        <span class="live-dot"></span> LIVE API
      </div>

      <!-- Parallax via CSS custom property set by JS / scroll-driven CSS  -->
      <h1 class="hero-title parallax-hero">Synthetic Societies</h1>

      <p class="hero-sub parallax-hero-sub">
        <strong>Behavioral Grounding Framework</strong> — LLM agents grounded in empirical data
        from the <strong>European Social Survey (ESS Round 11)</strong>. Agents make economic decisions
        across simulation rounds in a game-theoretic social network.
      </p>

      <div class="hero-actions">
        <router-link to="/run"          class="btn btn-primary">Run Simulation</router-link>
        <router-link to="/experiments"  class="btn btn-outline">Browse Experiments</router-link>
        <a href="https://github.com/touro3/SyntheticSocieties" target="_blank" class="btn btn-outline">GitHub</a>
      </div>
    </header>

    <!-- ── Stats row (staggered reveal) ───────────────────────────────── -->
    <div class="stats-row" ref="statsRef">
      <div class="stat-card reveal" v-for="s in statsData" :key="s.label">
        <div class="stat-val" :style="{ color: s.color }">{{ s.val }}</div>
        <div class="stat-label">{{ s.label }}</div>
      </div>
    </div>

    <!-- ── Feature cards (v-tilt 3D spring + staggered reveal) ────────── -->
    <div class="features" ref="featRef">
      <div
        class="feat-card glass-card reveal"
        v-for="f in features" :key="f.title"
        v-tilt
      >
        <div class="feat-icon">{{ f.icon }}</div>
        <h3>{{ f.title }}</h3>
        <p>{{ f.desc }}</p>
      </div>
    </div>

    <!-- ── Recent experiments ──────────────────────────────────────────── -->
    <section class="experiments-section">
      <div class="section-header">
        <h2 class="section-title">Recent Experiments</h2>
        <router-link to="/experiments" class="btn btn-ghost" style="font-size:.82rem;padding:6px 14px">View all →</router-link>
      </div>

      <div v-if="loading" class="empty"><span class="spin">⟳</span> Loading…</div>
      <div v-else-if="experiments.length === 0" class="empty card">
        No experiments yet.
        <router-link to="/run"> Run your first simulation →</router-link>
      </div>
      <div v-else class="exp-table card" ref="tableRef">
        <table>
          <thead>
            <tr>
              <th>Experiment ID</th><th>Policy</th><th>Seed</th>
              <th>Gini</th><th>Wealth Mean</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="e in experiments.slice(0,8)" :key="e.experiment_id"
              class="exp-row reveal"
              @click="goTo(e.experiment_id)"
            >
              <td class="mono exp-id">{{ e.experiment_id }}</td>
              <td><span class="badge badge-teal" style="font-size:.7rem">{{ e.policy_type || '—' }}</span></td>
              <td class="mono">{{ e.seed ?? '—' }}</td>
              <td class="mono gini-val">{{ fmt(e.gini) }}</td>
              <td class="mono">{{ fmt(e.wealth_mean) }}</td>
              <td><StatusBadge :status="e.status || 'complete'" /></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="note" class="note">{{ note }}</p>
    </section>

    <!-- ── API reference (staggered reveal) ──────────────────────────── -->
    <section style="margin-top:44px" ref="apiRef">
      <h2 class="section-title">API Quick Reference</h2>
      <div class="api-grid">
        <div class="api-item reveal glass-card" v-for="ep in endpoints" :key="ep.path">
          <span class="method" :class="ep.method === 'GET' ? 'get' : 'post'">{{ ep.method }}</span>
          <code class="mono ep-path">{{ ep.path }}</code>
          <span class="ep-desc">{{ ep.desc }}</span>
        </div>
      </div>
    </section>

    <footer class="footer">
      <p>Synthetic Societies · Behavioral Grounding Framework v1.0 · MIT License</p>
    </footer>

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'
import { useReveal } from '../composables/useReveal.js'
import StatusBadge from '../components/StatusBadge.vue'

const router = useRouter()

/* Refs for stagger targets */
const statsRef = ref(null)
const featRef  = ref(null)
const tableRef = ref(null)
const apiRef   = ref(null)

useReveal(statsRef)
useReveal(featRef)
useReveal(tableRef)
useReveal(apiRef)

/* Data */
const loading  = ref(true)
const experiments = ref([])
const note     = ref('')
const online   = ref(false)

const statsData = computed(() => [
  { label: 'API Status', val: online.value ? '● Online' : '○ Offline', color: online.value ? 'var(--green)' : 'var(--rose)' },
  { label: 'Experiments', val: experiments.value.length || '…', color: 'var(--blue)' },
  { label: 'Policy Types', val: 6, color: 'var(--purple)' },
  { label: 'Data Source',  val: 'ESS R11', color: 'var(--teal)' },
  { label: 'Metric',       val: 'Gini', color: 'var(--amber)' },
])

const features = [
  { icon: '📊', title: 'ESS Grounding',    desc: 'Agent profiles are sampled from ESS Round 11 microdata — real demographic, trust, and economic distributions from 31 European countries.' },
  { icon: '🧠', title: 'Multi-Policy',     desc: 'Six pluggable policy backends: random, rule-based, template, mock (CPU) and LLM-powered (Mistral-7B, GPT-4o) requiring GPU.' },
  { icon: '⚖️', title: 'Inequality Metrics', desc: 'Every run computes Gini coefficient, Behavioral Realism Metric (BRM), persona fidelity decay, and 20+ research-grade indicators.' },
  { icon: '🌐', title: 'Network Topology', desc: 'Small-world (Watts-Strogatz) and random (Erdős–Rényi) structures with configurable edge probability and rewiring coefficient.' },
  { icon: '💉', title: 'Live Injection',   desc: 'Inject exogenous events (wealth shock, signal update, narrative) into running simulations via the IPC bridge — no restart needed.' },
  { icon: '🔬', title: 'ReACT Analysis',   desc: 'Ask research questions in natural language — a ReACT agent queries the DuckDB experiment index and returns a structured report.' },
]

const endpoints = [
  { method: 'GET',  path: '/health',               desc: 'Liveness probe' },
  { method: 'POST', path: '/simulate',              desc: 'Trigger async simulation — returns experiment_id' },
  { method: 'GET',  path: '/status/{exp_id}',       desc: 'Poll run progress (round, state, heartbeat)' },
  { method: 'GET',  path: '/results/{exp_id}',      desc: 'Wealth distribution, Gini, action counts' },
  { method: 'GET',  path: '/experiments',           desc: 'List all experiments from DuckDB tracker' },
  { method: 'POST', path: '/interview/{exp}/{aid}', desc: 'Interview a live agent via IPC bridge' },
  { method: 'POST', path: '/inject/{exp_id}',       desc: 'Inject wealth_shock / signal_update / narrative' },
  { method: 'GET',  path: '/report?q={query}',      desc: 'ReACT agent analysis (needs OpenAI key)' },
  { method: 'GET',  path: '/incomplete',            desc: 'List resumable (non-complete) runs' },
]

const fmt  = v => (v !== undefined && v !== null) ? Number(v).toFixed(3) : '—'
const goTo = id => router.push(`/results/${id}`)

onMounted(async () => {
  try { await api.health(); online.value = true } catch {}
  try {
    const r = await api.experiments()
    experiments.value = r.data.experiments || []
    note.value = r.data.note || ''
  } catch {
    note.value = 'Could not load experiments.'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
/* ── Hero ──────────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 60px 0 40px; }
.hero-badge { margin-bottom: 20px; }
.live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); animation: pulse 2s infinite;
  display: inline-block;
}
.hero-title {
  font-size: 3.2rem; font-weight: 800; letter-spacing: -.03em;
  background: var(--grad);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 16px; line-height: 1.1;
}
.hero-sub {
  color: var(--text2); font-size: 1.05rem; max-width: 640px;
  margin: 0 auto 30px; line-height: 1.75;
}
.hero-actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }

/* Scroll-driven parallax — JS sets --scroll-y, CSS uses it */
.parallax-hero {
  transform: translateY(calc(var(--scroll-y, 0px) * -0.14));
  opacity: calc(1 - var(--scroll-y, 0px) / 520);
  will-change: transform, opacity;
}
.parallax-hero-sub {
  transform: translateY(calc(var(--scroll-y, 0px) * -0.07));
  opacity: calc(1 - var(--scroll-y, 0px) / 700);
  will-change: transform, opacity;
}

/* Native CSS scroll-driven parallax where supported */
@supports (animation-timeline: scroll()) {
  .parallax-hero {
    transform: none; opacity: 1;
    animation: parallax-title linear both;
    animation-timeline: scroll(root block);
    animation-range: 0px 480px;
  }
  .parallax-hero-sub {
    transform: none; opacity: 1;
    animation: parallax-sub linear both;
    animation-timeline: scroll(root block);
    animation-range: 0px 640px;
  }
  @keyframes parallax-title {
    to { opacity: .08; transform: translateY(-60px) scale(.96); }
  }
  @keyframes parallax-sub {
    to { opacity: .08; transform: translateY(-32px); }
  }
}

/* ── Stats ─────────────────────────────────────────────────────────── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px; margin: 40px 0;
}
.stat-card {
  background: rgba(26,31,53,.6);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 12px; padding: 18px; text-align: center;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
  transition: border-color .25s, transform .35s var(--ease-spring);
}
.stat-card:hover {
  border-color: rgba(99,102,241,.3);
  transform: translateY(-3px) scale(1.02);
}
.stat-val   { font-size: 1.5rem; font-weight: 700; }
.stat-label {
  font-size: .72rem; color: var(--text3);
  text-transform: uppercase; letter-spacing: .06em; margin-top: 5px;
}

/* ── Feature cards (glassmorphism 3D) ─────────────────────────────── */
.features {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
  gap: 18px; margin: 0 0 52px;
}

/* .glass-card is applied alongside .card or standalone */
.glass-card {
  background: rgba(26, 31, 53, 0.68);
  backdrop-filter: blur(22px) saturate(170%);
  -webkit-backdrop-filter: blur(22px) saturate(170%);
  border: 1px solid rgba(255,255,255,.06);
  box-shadow:
    0 4px 28px rgba(0,0,0,.3),
    inset 0 1px 0 rgba(255,255,255,.05),
    0 0 0 1px rgba(99,102,241,.06);
}

.feat-card {
  border-radius: 16px; padding: 26px;
  transform-style: preserve-3d;
  /* Transition only fires on mouseleave (spring snap back).
     The v-tilt directive sets it to none during tracking. */
  transition:
    border-color .25s,
    box-shadow   .25s;
  cursor: default;
}
.feat-card:hover {
  border-color: rgba(99,102,241,.35);
  box-shadow:
    0 12px 48px rgba(0,0,0,.35),
    inset 0 1px 0 rgba(255,255,255,.07),
    0 0 0 1px rgba(99,102,241,.16),
    var(--glow);
}
.feat-icon { font-size: 1.6rem; margin-bottom: 12px; }
.feat-card h3 { font-size: .95rem; font-weight: 600; margin-bottom: 7px; }
.feat-card p  { font-size: .82rem; color: var(--text2); line-height: 1.65; }

/* ── Experiments table ─────────────────────────────────────────────── */
.experiments-section { margin-top: 4px; }
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }

.exp-table { padding: 0; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  padding: 12px 16px; text-align: left;
  font-size: .72rem; font-weight: 600; letter-spacing: .05em;
  color: var(--text3); text-transform: uppercase;
  border-bottom: 1px solid rgba(255,255,255,.05);
}
td { padding: 12px 16px; font-size: .84rem; border-bottom: 1px solid rgba(255,255,255,.03); }
.exp-row {
  cursor: pointer;
  transition: background .15s, transform .25s var(--ease-spring);
}
.exp-row:hover td { background: rgba(99,102,241,.05); }
.exp-row.revealed:hover { transform: translateX(3px); }
.exp-id { color: var(--text2); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gini-val { color: var(--amber); }

.note { font-size: .76rem; color: var(--text3); margin-top: 10px; text-align: right; }

/* ── API grid ──────────────────────────────────────────────────────── */
.api-grid { display: grid; gap: 10px; }
.api-item {
  display: grid; grid-template-columns: 52px 220px 1fr;
  align-items: center; gap: 14px;
  border-radius: 10px; padding: 12px 18px;
  transition: border-color .2s, transform .3s var(--ease-spring);
}
.api-item.revealed:hover {
  border-color: rgba(99,102,241,.3);
  transform: translateX(4px);
}
.method { font-size: .7rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; text-align: center; letter-spacing: .04em; }
.method.get  { background: rgba(16,185,129,.15); color: var(--green); }
.method.post { background: rgba(99,102,241,.15); color: var(--blue);  }
.ep-path { font-size: .84rem; color: var(--text); }
.ep-desc { font-size: .79rem; color: var(--text2); }

.footer { text-align: center; padding: 52px 0 16px; color: var(--text3); font-size: .76rem; }

@media (max-width: 640px) {
  .hero-title { font-size: 2.1rem; }
  .api-item { grid-template-columns: 52px 1fr; }
  .ep-desc  { display: none; }
}
</style>
