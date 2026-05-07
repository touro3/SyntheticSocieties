<template>
  <div>
    <div class="page-header">
      <div class="breadcrumb">
        <router-link to="/experiments">Experiments</router-link>
        <span class="sep">›</span>
        <span class="mono crumb-id">{{ expId }}</span>
      </div>
      <h1 class="page-title" style="margin-top:8px">Monitor Run</h1>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="empty card">
      <span class="spin">⟳</span> Connecting to simulation…
    </div>

    <!-- Not found -->
    <div v-else-if="notFound" class="card empty-card">
      <div class="empty-icon">◌</div>
      <h3>Experiment not found</h3>
      <p>No run found for <code class="mono">{{ expId }}</code>. It may still be initialising.</p>
      <div style="display:flex;gap:10px;margin-top:16px;justify-content:center">
        <button class="btn btn-outline btn-sm" @click="refresh">⟳ Retry</button>
        <router-link to="/run" class="btn btn-primary btn-sm">Launch a run →</router-link>
      </div>
    </div>

    <template v-else>
      <!-- Status banner -->
      <div class="status-banner" :class="state.status">
        <div class="banner-left">
          <StatusBadge :status="state.status || 'pending'" />
          <span class="banner-id mono">{{ expId }}</span>
        </div>
        <div class="banner-right">
          <span class="updated-at">Updated {{ relTime }}</span>
          <button class="btn btn-ghost btn-sm" @click="refresh">⟳</button>
          <div class="poll-pill" v-if="polling">
            <span class="spin" style="font-size:.7rem">⟳</span> Polling
          </div>
        </div>
      </div>

      <!-- KPIs -->
      <div class="monitor-kpis">
        <div class="mkpi">
          <div class="mkpi-val" :class="state.status === 'complete' ? 'complete' : 'running'">
            {{ state.completed_rounds ?? 0 }}
            <span class="mkpi-sep">/</span>
            <span class="mkpi-total">{{ state.total_rounds ?? '?' }}</span>
          </div>
          <div class="mkpi-label">Rounds</div>
        </div>
        <div class="mkpi" v-if="heartbeat">
          <div class="mkpi-val" style="color:var(--teal)">{{ heartbeat.n_agents ?? '—' }}</div>
          <div class="mkpi-label">Agents</div>
        </div>
        <div class="mkpi" v-if="heartbeat?.mean_wealth != null">
          <div class="mkpi-val" style="color:var(--amber)">{{ Number(heartbeat.mean_wealth).toFixed(1) }}</div>
          <div class="mkpi-label">Avg Wealth</div>
        </div>
        <div class="mkpi" v-if="state.total_rounds">
          <div class="mkpi-val" style="color:var(--blue2)">{{ pct.toFixed(0) }}%</div>
          <div class="mkpi-label">Complete</div>
        </div>
      </div>

      <!-- Progress bar -->
      <div class="progress-track" v-if="state.total_rounds">
        <div class="progress-fill" :style="{ width: pct + '%' }"></div>
      </div>

      <!-- Main grid -->
      <div class="monitor-grid">

        <!-- Run details -->
        <div class="card details-card">
          <h2 class="section-title">Run Details</h2>
          <dl class="dl">
            <dt>Experiment ID</dt>
            <dd class="mono">{{ expId }}</dd>
            <dt>Status</dt>
            <dd><StatusBadge :status="state.status || 'pending'" /></dd>
            <dt>Total rounds</dt>
            <dd class="mono">{{ state.total_rounds ?? '—' }}</dd>
            <dt>Completed rounds</dt>
            <dd class="mono">{{ state.completed_rounds ?? '—' }}</dd>
            <dt>Started</dt>
            <dd>{{ fmtTime(state.started_at) }}</dd>
            <dt>Finished</dt>
            <dd>{{ fmtTime(state.finished_at) }}</dd>
          </dl>
          <div v-if="state.error_message" class="error-box" style="margin-top:14px">
            {{ state.error_message }}
          </div>
        </div>

        <!-- Heartbeat -->
        <div class="card hb-card" v-if="heartbeat">
          <h2 class="section-title">Live Heartbeat</h2>
          <dl class="dl">
            <dt>Round</dt>
            <dd class="mono">{{ heartbeat.round_id ?? heartbeat.round ?? '—' }}</dd>
            <dt>Agents</dt>
            <dd class="mono">{{ heartbeat.n_agents ?? '—' }}</dd>
            <dt>Avg wealth</dt>
            <dd class="mono">{{ heartbeat.mean_wealth != null ? Number(heartbeat.mean_wealth).toFixed(2) : '—' }}</dd>
            <dt>Last action</dt>
            <dd class="mono">{{ heartbeat.last_action ?? '—' }}</dd>
          </dl>
        </div>

        <!-- Actions -->
        <div class="card actions-card">
          <h2 class="section-title">Actions</h2>
          <div class="action-btns">
            <router-link v-if="state.status === 'complete'"
              :to="`/results/${expId}`" class="btn btn-primary">
              View Results →
            </router-link>
            <router-link :to="`/interact/${expId}`" class="btn btn-outline">
              Interview Agent
            </router-link>
            <router-link to="/run" class="btn btn-ghost">
              New Run
            </router-link>
          </div>
        </div>

      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'
import StatusBadge from '../components/StatusBadge.vue'

const route     = useRoute()
const expId     = route.params.expId
const state     = ref({})
const heartbeat = ref(null)
const loading   = ref(true)
const notFound  = ref(false)
const polling   = ref(false)
let timer       = null

const pct = computed(() =>
  state.value.total_rounds
    ? Math.min(100, ((state.value.completed_rounds ?? 0) / state.value.total_rounds) * 100)
    : 0
)

const relTime = computed(() => {
  const ts = state.value.updated_at
  if (!ts) return '—'
  const s = Math.max(0, Math.round(Date.now() / 1000 - ts))
  if (s < 60)  return `${s}s ago`
  if (s < 3600) return `${Math.round(s/60)}m ago`
  return `${Math.round(s/3600)}h ago`
})

const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : '—'

async function refresh() {
  try {
    const r = await api.status(expId)
    state.value     = r.data || {}
    heartbeat.value = r.data?.heartbeat || null
    notFound.value  = false
  } catch(e) {
    if (e.response?.status === 404) notFound.value = true
  }
}

function startPolling() {
  polling.value = true
  timer = setInterval(async () => {
    await refresh()
    if (['complete','failed'].includes(state.value.status)) stopPolling()
  }, 3000)
}

function stopPolling() {
  polling.value = false
  clearInterval(timer)
}

onMounted(async () => {
  await refresh()
  loading.value = false
  if (!notFound.value && !['complete','failed'].includes(state.value.status)) {
    startPolling()
  }
})

onBeforeUnmount(stopPolling)
</script>

<style scoped>
.page-header { margin-bottom: 22px; }
.breadcrumb { display: flex; align-items: center; gap: 7px; font-size: .8rem; color: var(--text3); }
.breadcrumb a { color: var(--text3); }
.breadcrumb a:hover { color: var(--text); }
.sep { color: var(--text3); }
.crumb-id { color: var(--text2); }

/* ── Empty / not found ──────────────────────────────────────────── */
.empty-card { text-align: center; padding: 48px 24px; }
.empty-icon { font-size: 2rem; color: var(--text3); margin-bottom: 12px; }
.empty-card h3 { margin-bottom: 8px; }
.empty-card p  { font-size: .86rem; color: var(--text2); }
.empty-card code { background: var(--bg3); padding: 2px 6px; border-radius: 5px; }

/* ── Status banner ──────────────────────────────────────────────── */
.status-banner {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
  padding: 13px 18px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  margin-bottom: 18px;
  transition: border-color .3s;
}
.status-banner.running { border-color: rgba(99,102,241,.3); }
.status-banner.complete { border-color: rgba(16,185,129,.3); }
.status-banner.failed   { border-color: rgba(244,63,94,.3); }

.banner-left  { display: flex; align-items: center; gap: 12px; }
.banner-right { display: flex; align-items: center; gap: 10px; }
.banner-id    { font-size: .82rem; color: var(--text2); }
.updated-at   { font-size: .74rem; color: var(--text3); }
.poll-pill {
  display: flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 20px;
  background: rgba(99,102,241,.1); color: var(--blue2);
  font-size: .72rem;
}

/* ── KPIs ───────────────────────────────────────────────────────── */
.monitor-kpis {
  display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 14px;
}
.mkpi {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 22px; min-width: 100px;
}
.mkpi-val { font-size: 2rem; font-weight: 700; display: flex; align-items: baseline; gap: 4px; }
.mkpi-val.running  { color: var(--blue2); }
.mkpi-val.complete { color: var(--green); }
.mkpi-sep   { font-size: 1.2rem; color: var(--text3); }
.mkpi-total { font-size: 1.2rem; color: var(--text2); font-weight: 600; }
.mkpi-label { font-size: .72rem; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }

/* ── Progress bar ───────────────────────────────────────────────── */
.progress-track {
  height: 5px; background: var(--bg4); border-radius: 99px;
  overflow: hidden; margin-bottom: 22px;
}
.progress-fill {
  height: 100%; background: var(--grad);
  border-radius: 99px; transition: width .5s ease;
}

/* ── Grid ───────────────────────────────────────────────────────── */
.monitor-grid {
  display: grid; grid-template-columns: 1fr 1fr 200px;
  gap: 16px; align-items: start;
}
@media (max-width: 860px) { .monitor-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .monitor-grid { grid-template-columns: 1fr; } }

/* ── DL ─────────────────────────────────────────────────────────── */
.dl {
  display: grid; grid-template-columns: auto 1fr;
  gap: 7px 16px;
}
dt { font-size: .76rem; color: var(--text3); font-weight: 500; align-self: center; }
dd { font-size: .83rem; color: var(--text); }

.action-btns { display: flex; flex-direction: column; gap: 9px; }
</style>
