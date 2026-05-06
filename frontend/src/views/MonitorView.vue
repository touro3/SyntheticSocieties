<template>
  <div class="container">
    <div class="page-header">
      <router-link to="/experiments" class="back">← Experiments</router-link>
      <h1 class="page-title mono">{{ expId }}</h1>
    </div>

    <div v-if="loading" class="empty"><span class="spin">⟳</span> Connecting…</div>
    <div v-else class="monitor-layout">

      <!-- Status card -->
      <div class="card status-card">
        <div class="status-row">
          <StatusBadge :status="state.status" />
          <span class="updated-at">Updated {{ relTime }}</span>
        </div>

        <div class="rounds-display">
          <span class="rounds-big">{{ state.completed_rounds ?? 0 }}</span>
          <span class="rounds-sep">/</span>
          <span class="rounds-total">{{ state.total_rounds ?? '?' }}</span>
          <span class="rounds-label">rounds</span>
        </div>

        <div class="progress-bar" v-if="state.total_rounds">
          <div class="progress-fill" :style="{ width: pct + '%' }"></div>
        </div>
        <div class="pct-label" v-if="state.total_rounds">{{ pct.toFixed(0) }}% complete</div>

        <div v-if="state.error_message" class="error-msg">{{ state.error_message }}</div>

        <div class="actions-row">
          <router-link v-if="state.status === 'complete'" :to="`/results/${expId}`" class="btn btn-primary">
            View Results →
          </router-link>
          <router-link :to="`/interact/${expId}`" class="btn btn-outline">
            Interview Agent
          </router-link>
          <button class="btn btn-ghost" @click="refresh">⟳ Refresh</button>
        </div>
      </div>

      <!-- Info panel -->
      <div class="info-col">
        <div class="card">
          <h3 class="section-title" style="font-size:.9rem">Run details</h3>
          <dl class="details">
            <dt>Experiment ID</dt>
            <dd class="mono">{{ expId }}</dd>
            <dt>Status</dt>
            <dd>{{ state.status ?? '—' }}</dd>
            <dt>Started</dt>
            <dd>{{ fmtTime(state.started_at) }}</dd>
            <dt>Finished</dt>
            <dd>{{ fmtTime(state.finished_at) }}</dd>
            <dt>Total rounds</dt>
            <dd>{{ state.total_rounds ?? '—' }}</dd>
            <dt>Completed rounds</dt>
            <dd>{{ state.completed_rounds ?? '—' }}</dd>
          </dl>
        </div>

        <div class="card" v-if="heartbeat">
          <h3 class="section-title" style="font-size:.9rem">Heartbeat</h3>
          <dl class="details">
            <dt>Round</dt>
            <dd>{{ heartbeat.round ?? '—' }}</dd>
            <dt>Agents</dt>
            <dd>{{ heartbeat.n_agents ?? '—' }}</dd>
            <dt>Avg wealth</dt>
            <dd>{{ heartbeat.mean_wealth?.toFixed(2) ?? '—' }}</dd>
            <dt>Last action</dt>
            <dd>{{ heartbeat.last_action ?? '—' }}</dd>
          </dl>
        </div>

        <div class="poll-note">
          <span class="spin" v-if="polling">⟳</span>
          {{ polling ? 'Polling every 3s…' : 'Run finished — polling stopped.' }}
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'
import StatusBadge from '../components/StatusBadge.vue'

const route   = useRoute()
const expId   = route.params.expId
const state   = ref({})
const heartbeat = ref(null)
const loading = ref(true)
const polling = ref(false)
let   timer   = null

const pct     = computed(() => state.value.total_rounds
  ? Math.min(100, (state.value.completed_rounds / state.value.total_rounds) * 100) : 0)

const relTime = computed(() => {
  if (!state.value.updated_at) return '—'
  const s = Math.round(Date.now() / 1000 - state.value.updated_at)
  if (s < 60) return `${s}s ago`
  return `${Math.round(s/60)}m ago`
})

const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : '—'

async function refresh() {
  try {
    const r = await api.status(expId)
    state.value = r.data
    heartbeat.value = r.data.heartbeat || null
  } catch {}
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
  if (!['complete','failed'].includes(state.value.status)) startPolling()
})

onBeforeUnmount(stopPolling)
</script>

<style scoped>
.page-header { margin-bottom: 28px; }
.back { color: var(--text3); font-size: .84rem; display: inline-block; margin-bottom: 12px; }
.back:hover { color: var(--text); text-decoration: none; }
.page-title { font-size: 1.6rem; font-weight: 700; color: var(--text2); word-break: break-all; }

.monitor-layout { display: grid; grid-template-columns: 1fr 300px; gap: 24px; align-items: start; }
@media (max-width: 760px) { .monitor-layout { grid-template-columns: 1fr; } }

.status-card { display: flex; flex-direction: column; gap: 20px; }
.status-row { display: flex; align-items: center; gap: 12px; }
.updated-at { font-size: .78rem; color: var(--text3); margin-left: auto; }

.rounds-display { display: flex; align-items: baseline; gap: 6px; }
.rounds-big   { font-size: 3.5rem; font-weight: 700; color: var(--blue); }
.rounds-sep   { font-size: 2rem; color: var(--text3); }
.rounds-total { font-size: 2rem; font-weight: 600; color: var(--text2); }
.rounds-label { font-size: .85rem; color: var(--text3); margin-left: 4px; }

.progress-bar {
  height: 6px; background: var(--border); border-radius: 99px; overflow: hidden;
}
.progress-fill {
  height: 100%; background: var(--blue);
  border-radius: 99px; transition: width .5s ease;
}
.pct-label { font-size: .78rem; color: var(--text3); }

.error-msg {
  background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2);
  border-radius: 8px; padding: 10px 14px; font-size: .84rem; color: var(--rose);
}

.actions-row { display: flex; gap: 10px; flex-wrap: wrap; }

.info-col { display: flex; flex-direction: column; gap: 16px; }

.details { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; }
dt { font-size: .78rem; color: var(--text3); font-weight: 500; align-self: center; white-space: nowrap; }
dd { font-size: .84rem; color: var(--text); }

.poll-note { font-size: .78rem; color: var(--text3); text-align: center; display: flex; align-items: center; justify-content: center; gap: 6px; }
</style>
