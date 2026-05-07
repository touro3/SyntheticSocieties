<template>
  <div>
    <div class="page-header">
      <div class="breadcrumb">
        <router-link to="/experiments">Experiments</router-link>
        <span>›</span>
        <span class="mono">{{ expId }}</span>
      </div>
      <h1 class="page-title" style="margin-top:8px">Results</h1>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="empty card">
      <span class="spin">⟳</span> Loading results…
    </div>

    <!-- Error -->
    <div v-else-if="err" class="card" style="padding:40px;text-align:center">
      <div style="color:var(--rose);margin-bottom:12px">{{ err }}</div>
      <router-link to="/experiments" class="btn btn-outline btn-sm">← Experiments</router-link>
    </div>

    <template v-else>
      <!-- KPI row -->
      <div class="kpi-row" ref="kpiRef">
        <div class="kpi reveal" v-for="(m, i) in kpis" :key="m.label" :style="{ '--i': i }" v-tilt>
          <div class="kpi-icon">{{ m.icon }}</div>
          <div class="kpi-val" :style="{ color: m.color }">{{ m.val }}</div>
          <div class="kpi-label">{{ m.label }}</div>
        </div>
      </div>

      <div class="results-grid">

        <!-- LEFT col: charts + metadata -->
        <div class="left-col">

          <!-- Wealth distribution chart -->
          <div class="card chart-card" v-if="wealthValues.length">
            <h2 class="section-title">Wealth Distribution</h2>
            <div class="gini-row">
              <span>Gini coefficient: <strong style="color:var(--amber)">{{ computedGini.toFixed(4) }}</strong></span>
              <span class="gini-hint">0 = equality · 1 = inequality</span>
            </div>
            <div class="chart-wrap">
              <canvas ref="wealthCanvas"></canvas>
            </div>
            <div class="wealth-stats">
              <span>Min <strong class="mono">{{ wealthMin.toFixed(1) }}</strong></span>
              <span>Mean <strong class="mono">{{ wealthMean.toFixed(1) }}</strong></span>
              <span>Median <strong class="mono">{{ wealthMedian.toFixed(1) }}</strong></span>
              <span>Max <strong class="mono">{{ wealthMax.toFixed(1) }}</strong></span>
            </div>
          </div>

          <!-- Action breakdown -->
          <div class="card chart-card" v-if="hasActions">
            <h2 class="section-title">Action Breakdown</h2>
            <div class="action-chart-wrap">
              <div class="action-bars">
                <div v-for="(a, i) in actionBreakdown" :key="a.name" class="action-bar-row">
                  <span class="act-name mono">{{ a.name }}</span>
                  <div class="act-track">
                    <div class="act-fill" :style="{ width: a.pct + '%', background: a.color }" />
                  </div>
                  <span class="act-count">{{ a.count }}</span>
                  <span class="act-pct" :style="{ color: a.color }">{{ a.pct.toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Metadata -->
          <div class="card" v-if="meta && Object.keys(meta).length">
            <h2 class="section-title">Run Configuration</h2>
            <dl class="dl-grid">
              <template v-for="(v, k) in meta" :key="k">
                <dt>{{ k }}</dt>
                <dd class="mono">{{ v ?? '—' }}</dd>
              </template>
            </dl>
          </div>

          <!-- Flat metrics -->
          <div class="card" v-if="hasMetrics">
            <h2 class="section-title">Metrics</h2>
            <dl class="dl-grid">
              <template v-for="(v, k) in flatMetrics" :key="k">
                <dt>{{ k }}</dt>
                <dd class="mono">{{ typeof v === 'number' ? v.toFixed(4) : String(v) }}</dd>
              </template>
            </dl>
          </div>

        </div>

        <!-- RIGHT col: actions -->
        <div class="right-col">

          <div class="card">
            <h2 class="section-title">Actions</h2>
            <div class="action-btns">
              <router-link :to="`/monitor/${expId}`" class="btn btn-outline">
                ◫ Monitor View
              </router-link>
              <router-link :to="`/interact/${expId}`" class="btn btn-outline">
                ◎ Interview Agent
              </router-link>
              <a :href="`/results/${expId}`" target="_blank" rel="noopener" class="btn btn-ghost">
                { } Raw JSON →
              </a>
            </div>
          </div>

          <!-- Quick inject -->
          <div class="card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
              <h2 class="section-title" style="margin:0">Inject Event</h2>
              <span v-if="isRunning" class="live-chip">● Live</span>
              <span v-else class="stopped-chip">◌ Completed</span>
            </div>

            <div v-if="!isRunning" class="inject-note">
              Inject works on actively running simulations only.
              <router-link to="/run" style="color:var(--blue2);margin-left:4px">New run →</router-link>
            </div>

            <template v-else>
              <div class="field">
                <label>Event type</label>
                <select v-model="inject.type">
                  <option value="wealth_shock">▼ wealth_shock</option>
                  <option value="signal_update">◎ signal_update</option>
                  <option value="narrative">▶ narrative</option>
                </select>
              </div>
              <div class="field" style="margin-top:10px">
                <label>Payload (JSON)</label>
                <textarea v-model="inject.payload" rows="3"
                  style="font-family:monospace;font-size:.8rem"
                  placeholder='{"factor": 0.5}' />
              </div>
              <button class="btn btn-outline" style="width:100%;justify-content:center;margin-top:10px"
                @click="doInject" :disabled="injecting">
                <span v-if="injecting" class="spin">⟳</span> Inject
              </button>
              <div v-if="injectResult" class="inject-result mono">{{ injectResult }}</div>
            </template>
          </div>

          <!-- Behavior summary -->
          <div class="card" v-if="behavior">
            <h2 class="section-title">Behavior Summary</h2>
            <dl class="dl-grid">
              <dt>Cooperation rate</dt>
              <dd class="mono" :style="{ color: cooperationColor }">{{ fmtPct(behavior.cooperation_rate) }}</dd>
              <dt>Defection rate</dt>
              <dd class="mono" style="color:var(--rose)">{{ fmtPct(behavior.defection_rate) }}</dd>
            </dl>
          </div>

        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { api, gini, mean } from '../api/index.js'
import { useReveal } from '../composables/useReveal.js'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

const route  = useRoute()
const expId  = route.params.expId

const kpiRef = ref(null)
useReveal(kpiRef)

const loading = ref(true)
const err     = ref('')
const summary = ref(null)
const metrics = ref(null)
const meta    = ref(null)
const wealthCanvas = ref(null)
let chart = null

const inject    = ref({ type: 'wealth_shock', payload: '{"factor": 0.5}' })
const injecting = ref(false)
const injectResult = ref('')

// Whether the sim associated with these results is still running
const isRunning = computed(() => {
  const st = meta.value?.status ?? summary.value?.status
  return st === 'running' || st === 'pending'
})

// ── Wealth data — try multiple paths the API might use ─────────────
const wealthValues = computed(() => {
  const s = summary.value
  if (!s) return []
  // Most common: summary.wealth.values
  if (Array.isArray(s.wealth?.values)) return s.wealth.values
  // Fallback: summary.agent_wealth
  if (Array.isArray(s.agent_wealth)) return s.agent_wealth
  // Fallback: summary.final_wealth
  if (Array.isArray(s.final_wealth)) return s.final_wealth
  return []
})

const behavior = computed(() => summary.value?.behavior ?? null)

const cooperationColor = computed(() => {
  const r = behavior.value?.cooperation_rate ?? 0
  return `hsl(${Math.round(r * 120)},65%,58%)`
})

const computedGini   = computed(() => gini(wealthValues.value))
const wealthMean     = computed(() => mean(wealthValues.value))
const wealthMin      = computed(() => wealthValues.value.length ? Math.min(...wealthValues.value) : 0)
const wealthMax      = computed(() => wealthValues.value.length ? Math.max(...wealthValues.value) : 0)
const wealthMedian   = computed(() => {
  const s = [...wealthValues.value].sort((a,b)=>a-b)
  const m = Math.floor(s.length/2)
  return s.length ? (s.length%2 ? s[m] : (s[m-1]+s[m])/2) : 0
})

const hasActions = computed(() => {
  const counts = summary.value?.event_action_counts ?? summary.value?.actions
  return counts && typeof counts === 'object' && Object.keys(counts).length > 0
})

const actionBreakdown = computed(() => {
  const counts = summary.value?.event_action_counts ?? summary.value?.actions ?? {}
  const colors = { work: 'var(--green)', save: 'var(--teal)', cooperate: 'var(--blue2)', steal: 'var(--rose)' }
  const total = Object.values(counts).reduce((a,b) => a + b, 0) || 1
  return Object.entries(counts).map(([name, count]) => ({
    name, count,
    pct: (count / total) * 100,
    color: colors[name] ?? 'var(--text2)',
  })).sort((a,b) => b.count - a.count)
})

const kpis = computed(() => [
  { icon: '◆', label: 'Agents',      val: summary.value?.num_agents ?? meta.value?.population_size ?? '—', color: 'var(--blue2)' },
  { icon: '◎', label: 'Gini',        val: computedGini.value.toFixed(4),                                   color: 'var(--amber)' },
  { icon: '▲', label: 'Wealth Mean', val: wealthValues.value.length ? wealthMean.value.toFixed(1) : '—',   color: 'var(--teal)' },
  { icon: '◈', label: 'Policy',      val: meta.value?.policy_type ?? '—',                                  color: 'var(--purple)' },
  { icon: '◻', label: 'Rounds',      val: meta.value?.rounds ?? '—',                                       color: 'var(--text2)' },
  { icon: '⬡', label: 'Coop Rate',   val: behavior.value?.cooperation_rate != null ? fmtPct(behavior.value.cooperation_rate) : '—', color: 'var(--green)' },
])

const hasMetrics  = computed(() => metrics.value && Object.keys(flatMetrics.value).length > 0)
const flatMetrics = computed(() => {
  if (!metrics.value) return {}
  const out = {}
  const walk = (obj, prefix = '') => {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (v !== null && typeof v === 'object' && !Array.isArray(v)) walk(v, key)
      else if (!Array.isArray(v) && v !== null) out[key] = v
    }
  }
  walk(metrics.value)
  // Limit to 30 most relevant entries
  return Object.fromEntries(Object.entries(out).slice(0, 30))
})

function fmtPct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : '—' }

function buildHistogram(values, bins = 20) {
  const min = Math.min(...values), max = Math.max(...values)
  const step = (max - min) / bins || 1
  const counts = Array(bins).fill(0)
  const labels = Array.from({ length: bins }, (_, i) => (min + i * step).toFixed(0))
  for (const v of values) counts[Math.min(bins-1, Math.floor((v-min)/step))]++
  return { labels, counts }
}

function drawChart() {
  if (!wealthCanvas.value || !wealthValues.value.length) return
  if (chart) { chart.destroy(); chart = null }
  const { labels, counts } = buildHistogram(wealthValues.value)
  chart = new Chart(wealthCanvas.value, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Agents', data: counts,
        backgroundColor: 'rgba(99,102,241,.55)',
        borderColor: 'rgba(99,102,241,.85)',
        borderWidth: 1, borderRadius: 4 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#475569', font:{size:10} }, grid: { color:'rgba(42,48,80,.3)' } },
        y: { ticks: { color: '#475569', font:{size:10} }, grid: { color:'rgba(42,48,80,.3)' } },
      },
    },
  })
}

async function doInject() {
  injecting.value = true; injectResult.value = ''
  try {
    const payload = JSON.parse(inject.value.payload)
    const r = await api.inject(expId, inject.value.type, payload)
    injectResult.value = JSON.stringify(r.data, null, 2)
  } catch(e) {
    injectResult.value = e.response?.data?.error || e.message
  } finally { injecting.value = false }
}

onMounted(async () => {
  try {
    const r = await api.results(expId)
    summary.value = r.data.summary  ?? null
    metrics.value = r.data.metrics  ?? null
    meta.value    = r.data.metadata ?? null
    await nextTick()
    drawChart()
  } catch(e) {
    err.value = e.response?.data?.error || 'Failed to load results for this experiment.'
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => { if (chart) { chart.destroy(); chart = null } })
</script>

<style scoped>
.page-header { margin-bottom: 22px; }
.breadcrumb {
  display: flex; align-items: center; gap: 6px;
  font-size: .8rem; color: var(--text3); margin-bottom: 8px;
}
.breadcrumb a { color: var(--text3); }
.breadcrumb a:hover { color: var(--text2); text-decoration: none; }

/* ── KPI row ─────────────────────────────────────────────────────── */
.kpi-row {
  display: grid; grid-template-columns: repeat(6, 1fr);
  gap: 12px; margin-bottom: 24px;
}
@media (max-width: 1100px) { .kpi-row { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px)  { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
.kpi {
  background: rgba(13,21,40,.85);
  border: 1px solid var(--border); border-radius: 14px; padding: 16px;
  text-align: center; transform-style: preserve-3d;
  transition: border-color .25s, box-shadow .25s;
}
.kpi:hover { border-color: rgba(99,102,241,.22); box-shadow: var(--glow); }
.kpi-icon  { font-size: .82rem; color: var(--text3); margin-bottom: 8px; }
.kpi-val   { font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; }
.kpi-label { font-size: .65rem; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; margin-top: 5px; font-weight: 600; }

/* ── Grid ───────────────────────────────────────────────────────── */
.results-grid { display: grid; grid-template-columns: 1fr 260px; gap: 20px; align-items: start; }
@media (max-width: 860px) { .results-grid { grid-template-columns: 1fr; } }

.left-col  { display: flex; flex-direction: column; gap: 18px; }
.right-col { display: flex; flex-direction: column; gap: 16px; }

/* ── Charts ─────────────────────────────────────────────────────── */
.chart-card { display: flex; flex-direction: column; gap: 12px; }
.gini-row { font-size: .83rem; color: var(--text2); display: flex; align-items: center; gap: 12px; }
.gini-hint { font-size: .74rem; color: var(--text3); }
.chart-wrap { height: 200px; position: relative; }
.wealth-stats {
  display: flex; gap: 18px; flex-wrap: wrap;
  font-size: .78rem; color: var(--text2);
  padding-top: 8px; border-top: 1px solid var(--border);
}
.wealth-stats strong { color: var(--text); }

/* ── Action bars ────────────────────────────────────────────────── */
.action-bars { display: flex; flex-direction: column; gap: 12px; }
.action-bar-row { display: grid; grid-template-columns: 80px 1fr 50px 55px; align-items: center; gap: 10px; }
.act-name { font-size: .8rem; color: var(--text2); }
.act-track { height: 8px; background: var(--bg4); border-radius: 4px; overflow: hidden; }
.act-fill  { height: 100%; border-radius: 4px; transition: width .6s var(--ease-out); }
.act-count { font-size: .76rem; color: var(--text3); text-align: right; }
.act-pct   { font-size: .76rem; font-weight: 600; text-align: right; }

/* ── DL ─────────────────────────────────────────────────────────── */
.dl-grid { display: grid; grid-template-columns: auto 1fr; gap: 5px 14px; }
dt { font-size: .74rem; color: var(--text3); }
dd { font-size: .8rem; color: var(--text); word-break: break-all; }

.action-btns { display: flex; flex-direction: column; gap: 8px; }

.inject-result {
  margin-top: 10px; background: var(--bg2);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 10px; font-size: .73rem; color: var(--teal);
  white-space: pre-wrap; max-height: 120px; overflow-y: auto;
}

.live-chip {
  font-size: .67rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
  color: var(--green); padding: 2px 8px; border-radius: 99px;
  background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.2);
}
.stopped-chip {
  font-size: .67rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text3); padding: 2px 8px; border-radius: 99px;
  background: rgba(100,116,139,.08); border: 1px solid rgba(100,116,139,.15);
}
.inject-note {
  font-size: .82rem; color: var(--text3); padding: 12px 0;
  border-bottom: 1px solid var(--border); margin-bottom: 2px;
  line-height: 1.6;
}
</style>
