<template>
  <div class="container">

    <!-- Hero -->
    <header class="hero">
      <div class="hero-badge badge badge-blue">
        <span class="live-dot"></span> LIVE API
      </div>
      <h1 class="hero-title">Synthetic Societies</h1>
      <p class="hero-sub">
        <strong>Behavioral Grounding Framework</strong> — LLM agents grounded in empirical data
        from the <strong>European Social Survey (ESS Round 11)</strong>. Agents make economic decisions
        (work, save, cooperate, steal) in a game-theoretic setting across simulation rounds.
      </p>
      <div class="hero-actions">
        <router-link to="/run" class="btn btn-primary">Run Simulation</router-link>
        <router-link to="/experiments" class="btn btn-outline">Browse Experiments</router-link>
        <a href="https://github.com/touro3/SyntheticSocieties" target="_blank" class="btn btn-outline">GitHub</a>
      </div>
    </header>

    <!-- Stats row -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-val" :style="{ color: online ? 'var(--green)' : 'var(--rose)' }">
          {{ online ? '● Online' : '○ Offline' }}
        </div>
        <div class="stat-label">API Status</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{{ expCount }}</div>
        <div class="stat-label">Experiments</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">6</div>
        <div class="stat-label">Policy Types</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">ESS R11</div>
        <div class="stat-label">Data Source</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">Gini</div>
        <div class="stat-label">Inequality Metric</div>
      </div>
    </div>

    <!-- Feature cards -->
    <div class="features">
      <div class="feat-card">
        <div class="feat-icon">📊</div>
        <h3>ESS Grounding</h3>
        <p>Agent profiles are sampled from ESS Round 11 microdata — real demographic, trust, and economic distributions from 31 European countries.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🧠</div>
        <h3>Multi-Policy Engine</h3>
        <p>Six pluggable policy backends: random, rule-based, template, mock (CPU) and LLM-powered (Mistral-7B, GPT-4o) requiring GPU.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">⚖️</div>
        <h3>Inequality Metrics</h3>
        <p>Every run computes Gini coefficient, Behavioral Realism Metric (BRM), persona fidelity decay, and 20+ other research-grade indicators.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🌐</div>
        <h3>Network Topology</h3>
        <p>Small-world (Watts-Strogatz) and random (Erdős–Rényi) network structures with configurable edge probability and rewiring.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">💉</div>
        <h3>Live Injection</h3>
        <p>Inject exogenous events (wealth shock, signal update, narrative) into running simulations via the IPC bridge — no restart needed.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🔬</div>
        <h3>ReACT Analysis</h3>
        <p>Ask research questions in natural language — a ReACT agent queries the DuckDB experiment index and returns a structured report.</p>
      </div>
    </div>

    <!-- Recent experiments -->
    <section>
      <div class="section-header">
        <h2 class="section-title">Recent Experiments</h2>
        <router-link to="/experiments" class="btn btn-ghost" style="font-size:.82rem;padding:6px 14px">View all →</router-link>
      </div>

      <div v-if="loading" class="empty"><span class="spin">⟳</span> Loading…</div>
      <div v-else-if="experiments.length === 0" class="empty card">
        No experiments found.
        <router-link to="/run"> Run your first simulation →</router-link>
      </div>
      <div v-else class="exp-table card">
        <table>
          <thead>
            <tr>
              <th>Experiment ID</th>
              <th>Policy</th>
              <th>Seed</th>
              <th>Gini</th>
              <th>Wealth Mean</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="e in experiments.slice(0,8)" :key="e.experiment_id"
                class="exp-row" @click="goTo(e.experiment_id)">
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

    <!-- API quick-ref -->
    <section style="margin-top:40px">
      <h2 class="section-title">API Quick Reference</h2>
      <div class="api-grid">
        <div class="api-item" v-for="ep in endpoints" :key="ep.path">
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
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/index.js'
import StatusBadge from '../components/StatusBadge.vue'

const router   = useRouter()
const online   = ref(false)
const loading  = ref(true)
const experiments = ref([])
const expCount = ref('…')
const note     = ref('')

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

const fmt = v => (v !== undefined && v !== null) ? Number(v).toFixed(3) : '—'
const goTo = id => router.push(`/results/${id}`)

onMounted(async () => {
  try { await api.health(); online.value = true } catch {}
  try {
    const r = await api.experiments()
    experiments.value = r.data.experiments || []
    expCount.value = experiments.value.length
    note.value = r.data.note || ''
  } catch {
    note.value = 'Could not load experiments.'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.hero { text-align: center; padding: 52px 0 32px; }
.hero-badge { margin-bottom: 18px; }
.live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); animation: pulse 2s infinite;
  display: inline-block;
}
.hero-title {
  font-size: 3rem; font-weight: 700; letter-spacing: -.03em;
  background: var(--grad); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 14px;
}
.hero-sub {
  color: var(--text2); font-size: 1.05rem; max-width: 660px;
  margin: 0 auto 28px; line-height: 1.75;
}
.hero-actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px; margin: 36px 0;
}
.stat-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px; text-align: center;
  transition: border-color .2s;
}
.stat-card:hover { border-color: var(--blue); }
.stat-val { font-size: 1.5rem; font-weight: 700; color: var(--blue); }
.stat-label { font-size: .74rem; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }

.features {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px; margin: 0 0 48px;
}
.feat-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 14px; padding: 22px;
  transition: border-color .2s, transform .2s;
}
.feat-card:hover { border-color: rgba(99,102,241,.35); transform: translateY(-2px); }
.feat-icon { font-size: 1.5rem; margin-bottom: 10px; }
.feat-card h3 { font-size: .95rem; margin-bottom: 6px; }
.feat-card p  { font-size: .82rem; color: var(--text2); line-height: 1.6; }

.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }

.exp-table { padding: 0; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  padding: 12px 16px; text-align: left;
  font-size: .74rem; font-weight: 600; letter-spacing: .05em;
  color: var(--text3); text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
td { padding: 12px 16px; font-size: .85rem; border-bottom: 1px solid rgba(42,48,80,.5); }
.exp-row { cursor: pointer; transition: background .15s; }
.exp-row:hover td { background: rgba(99,102,241,.05); }
.exp-id { color: var(--text2); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gini-val { color: var(--amber); }

.note { font-size: .78rem; color: var(--text3); margin-top: 10px; text-align: right; }

.api-grid { display: grid; gap: 10px; }
.api-item {
  display: grid; grid-template-columns: 54px 240px 1fr;
  align-items: center; gap: 14px;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 18px;
  transition: border-color .15s;
}
.api-item:hover { border-color: rgba(99,102,241,.3); }
.method { font-size: .7rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; text-align: center; letter-spacing: .04em; }
.method.get  { background: rgba(16,185,129,.15); color: var(--green); }
.method.post { background: rgba(99,102,241,.15); color: var(--blue);  }
.ep-path { font-size: .84rem; color: var(--text); }
.ep-desc { font-size: .8rem; color: var(--text2); }

.footer { text-align: center; padding: 48px 0 16px; color: var(--text3); font-size: .78rem; }

@media (max-width: 640px) {
  .hero-title { font-size: 2rem; }
  .api-item { grid-template-columns: 54px 1fr; }
  .ep-desc { display: none; }
}
</style>
