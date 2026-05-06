<template>
  <div class="container">
    <div class="page-header">
      <router-link to="/experiments" class="back">← Experiments</router-link>
      <h1 class="page-title mono">{{ expId }}</h1>
    </div>

    <div v-if="loading" class="empty"><span class="spin">⟳</span> Loading results…</div>
    <div v-else-if="err" class="empty card" style="color:var(--rose)">{{ err }}</div>

    <template v-else>
      <!-- Summary row -->
      <div class="summary-row">
        <div class="metric-card" v-for="m in summaryMetrics" :key="m.label">
          <div class="metric-val" :style="{ color: m.color }">{{ m.val }}</div>
          <div class="metric-label">{{ m.label }}</div>
        </div>
      </div>

      <div class="results-layout">
        <!-- Left: data -->
        <div class="data-col">

          <!-- Wealth distribution chart -->
          <div class="card chart-card" v-if="wealthValues.length">
            <h2 class="section-title">Wealth Distribution</h2>
            <div class="gini-badge">
              Gini coefficient: <strong>{{ computedGini.toFixed(4) }}</strong>
              <span class="gini-hint">(0 = perfect equality, 1 = maximal inequality)</span>
            </div>
            <div class="chart-wrap">
              <canvas ref="wealthCanvas"></canvas>
            </div>
            <div class="wealth-stats">
              <span>Min: <strong class="mono">{{ wealthMin.toFixed(1) }}</strong></span>
              <span>Mean: <strong class="mono">{{ wealthMean.toFixed(1) }}</strong></span>
              <span>Median: <strong class="mono">{{ wealthMedian.toFixed(1) }}</strong></span>
              <span>Max: <strong class="mono">{{ wealthMax.toFixed(1) }}</strong></span>
            </div>
          </div>

          <!-- Metadata -->
          <div class="card" v-if="meta">
            <h2 class="section-title">Run Metadata</h2>
            <dl class="details">
              <template v-for="(v, k) in meta" :key="k">
                <dt>{{ k }}</dt>
                <dd class="mono">{{ v ?? '—' }}</dd>
              </template>
            </dl>
          </div>

          <!-- Raw metrics -->
          <div class="card" v-if="metrics && Object.keys(metrics).length">
            <h2 class="section-title">Metrics</h2>
            <dl class="details">
              <template v-for="(v, k) in flatMetrics" :key="k">
                <dt>{{ k }}</dt>
                <dd class="mono">{{ typeof v === 'number' ? v.toFixed(4) : v }}</dd>
              </template>
            </dl>
          </div>

        </div>

        <!-- Right: actions -->
        <div class="action-col">
          <div class="card actions-card">
            <h2 class="section-title">Quick Actions</h2>
            <div class="action-list">
              <router-link :to="`/interact/${expId}`" class="btn btn-primary action-btn">
                Interview Agent
              </router-link>
              <router-link :to="`/monitor/${expId}`" class="btn btn-outline action-btn">
                Monitor View
              </router-link>
              <a :href="`/results/${expId}`" target="_blank" class="btn btn-ghost action-btn">
                Raw JSON →
              </a>
            </div>
          </div>

          <!-- Inject event -->
          <div class="card inject-card">
            <h2 class="section-title">Inject Event</h2>
            <p class="inject-note">Inject an exogenous event into a running simulation.</p>
            <div class="field">
              <label>Event type</label>
              <select v-model="inject.type">
                <option value="wealth_shock">wealth_shock</option>
                <option value="signal_update">signal_update</option>
                <option value="narrative">narrative</option>
              </select>
            </div>
            <div class="field">
              <label>Payload (JSON)</label>
              <textarea v-model="inject.payload" rows="3" style="font-family:monospace;font-size:.8rem" placeholder='{"factor": 0.5}'></textarea>
            </div>
            <button class="btn btn-outline action-btn" @click="doInject" :disabled="injecting">
              <span v-if="injecting" class="spin">⟳</span> Inject
            </button>
            <div v-if="injectResult" class="inject-result mono">{{ injectResult }}</div>
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
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

const route  = useRoute()
const expId  = route.params.expId

const loading = ref(true)
const err     = ref('')
const summary = ref(null)
const metrics = ref(null)
const meta    = ref(null)
const wealthCanvas = ref(null)
let   chart   = null

const inject   = ref({ type: 'wealth_shock', payload: '{"factor": 0.5}' })
const injecting  = ref(false)
const injectResult = ref('')

const wealthValues = computed(() => summary.value?.wealth?.values ?? [])
const computedGini = computed(() => gini(wealthValues.value))
const wealthMean   = computed(() => mean(wealthValues.value))
const wealthMin    = computed(() => wealthValues.value.length ? Math.min(...wealthValues.value) : 0)
const wealthMax    = computed(() => wealthValues.value.length ? Math.max(...wealthValues.value) : 0)
const wealthMedian = computed(() => {
  const s = [...wealthValues.value].sort((a,b)=>a-b)
  const m = Math.floor(s.length/2)
  return s.length ? (s.length%2 ? s[m] : (s[m-1]+s[m])/2) : 0
})

const summaryMetrics = computed(() => [
  { label: 'Agents',      val: summary.value?.num_agents ?? meta.value?.population_size ?? '—', color: 'var(--blue)' },
  { label: 'Gini',        val: computedGini.value.toFixed(4),                                   color: 'var(--amber)' },
  { label: 'Wealth Mean', val: wealthMean.value.toFixed(1),                                     color: 'var(--teal)' },
  { label: 'Wealth Max',  val: wealthMax.value.toFixed(1),                                      color: 'var(--green)' },
  { label: 'Policy',      val: meta.value?.policy_type ?? '—',                                  color: 'var(--purple)' },
  { label: 'Rounds',      val: meta.value?.rounds ?? '—',                                       color: 'var(--text2)' },
])

const flatMetrics = computed(() => {
  if (!metrics.value) return {}
  const out = {}
  const walk = (obj, prefix = '') => {
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === 'object' && v !== null && !Array.isArray(v)) walk(v, prefix ? `${prefix}.${k}` : k)
      else if (!Array.isArray(v)) out[prefix ? `${prefix}.${k}` : k] = v
    }
  }
  walk(metrics.value)
  return out
})

function buildHistogram(values, bins = 20) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const step = (max - min) / bins || 1
  const counts = Array(bins).fill(0)
  const labels = Array.from({ length: bins }, (_, i) => (min + i * step).toFixed(0))
  for (const v of values) {
    const i = Math.min(bins - 1, Math.floor((v - min) / step))
    counts[i]++
  }
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
      datasets: [{
        label: 'Agents',
        data: counts,
        backgroundColor: 'rgba(99,102,241,.6)',
        borderColor: 'rgba(99,102,241,.9)',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(42,48,80,.4)' } },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(42,48,80,.4)' } },
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
  } catch (e) {
    injectResult.value = e.message
  } finally { injecting.value = false }
}

onMounted(async () => {
  try {
    const r = await api.results(expId)
    summary.value = r.data.summary || null
    metrics.value = r.data.metrics || null
    meta.value    = r.data.metadata || null
    await nextTick()
    drawChart()
  } catch (e) {
    err.value = e.response?.data?.error || 'Failed to load results.'
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => { if (chart) chart.destroy() })
</script>

<style scoped>
.page-header { margin-bottom: 28px; }
.back { color: var(--text3); font-size: .84rem; display: inline-block; margin-bottom: 12px; }
.back:hover { color: var(--text); text-decoration: none; }
.page-title { font-size: 1.6rem; font-weight: 700; color: var(--text2); word-break: break-all; }

.summary-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px; margin-bottom: 24px;
}
.metric-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px; text-align: center;
}
.metric-val   { font-size: 1.4rem; font-weight: 700; }
.metric-label { font-size: .72rem; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }

.results-layout { display: grid; grid-template-columns: 1fr 280px; gap: 24px; align-items: start; }
@media (max-width: 800px) { .results-layout { grid-template-columns: 1fr; } }

.data-col { display: flex; flex-direction: column; gap: 20px; }

.chart-card { display: flex; flex-direction: column; gap: 12px; }
.gini-badge { font-size: .84rem; color: var(--text2); }
.gini-badge strong { color: var(--amber); font-size: 1rem; }
.gini-hint  { font-size: .76rem; color: var(--text3); margin-left: 6px; }
.chart-wrap { height: 220px; position: relative; }
.wealth-stats {
  display: flex; gap: 20px; flex-wrap: wrap;
  font-size: .8rem; color: var(--text2);
  padding-top: 8px; border-top: 1px solid var(--border);
}
.wealth-stats strong { color: var(--text); }

.details { display: grid; grid-template-columns: auto 1fr; gap: 5px 16px; }
dt { font-size: .76rem; color: var(--text3); font-weight: 500; }
dd { font-size: .82rem; color: var(--text); }

.action-col { display: flex; flex-direction: column; gap: 16px; }
.actions-card { display: flex; flex-direction: column; gap: 12px; }
.action-list { display: flex; flex-direction: column; gap: 8px; }
.action-btn  { justify-content: center; }

.inject-card { display: flex; flex-direction: column; gap: 12px; }
.inject-note { font-size: .8rem; color: var(--text2); }
.field { display: flex; flex-direction: column; gap: 5px; }
.field label { font-size: .78rem; color: var(--text3); }
.inject-result {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px; font-size: .76rem;
  color: var(--teal); white-space: pre-wrap; word-break: break-all;
  max-height: 120px; overflow-y: auto;
}
</style>
