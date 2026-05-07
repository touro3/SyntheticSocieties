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

    <!-- Initialising (404, still within patience window) -->
    <div v-else-if="!notFound && !state.status" class="card empty-card">
      <div class="empty-icon"><span class="spin" style="font-size:1.5rem">⟳</span></div>
      <h3>Initialising…</h3>
      <p>Waiting for the simulation process to start. This takes a few seconds for LLM policies.</p>
    </div>

    <!-- Not found (still polling in background) -->
    <div v-else-if="notFound" class="card empty-card">
      <div class="empty-icon">◌</div>
      <h3>Waiting for run…</h3>
      <p>
        No state yet for <code class="mono">{{ expId }}</code>.
        Checking every 3 s — LLM policy startup can take 30–60 s on first launch.
      </p>
      <div style="display:flex;gap:10px;margin-top:16px;justify-content:center;align-items:center">
        <span class="spin" style="font-size:.9rem;color:var(--text3)">⟳</span>
        <span style="font-size:.78rem;color:var(--text3)">Still polling…</span>
        <router-link to="/run" class="btn btn-ghost btn-sm" style="margin-left:8px">Launch another →</router-link>
      </div>
    </div>

    <template v-else>

      <!-- GPU Wait banner -->
      <div v-if="state.gpu_wait" class="gpu-wait-banner">
        <span class="gw-icon">🔲</span>
        <div class="gw-body">
          <div class="gw-title">Waiting for GPU · LLM loading…</div>
          <div class="gw-sub">
            The <code>{{ state.policy_type }}</code> policy requires a CUDA-capable GPU.
            Model weights are being loaded into VRAM — the first round typically starts in 2–5 min.
            <span v-if="staleMinutes > 0" style="color:rgba(245,158,11,.6)">
              (stalled {{ staleMinutes }}m)
            </span>
          </div>
        </div>
        <button class="btn btn-ghost btn-sm" @click="refresh">⟳</button>
      </div>

      <!-- Stale (non-GPU) banner -->
      <div v-else-if="state.stale && state.status === 'running'" class="warn-box" style="margin-bottom:14px">
        ⚠ Run appears stalled — no progress in the last {{ staleMinutes }}m.
        <button class="btn btn-ghost btn-sm" style="margin-left:10px" @click="refresh">⟳ Retry</button>
      </div>

      <!-- Status banner -->
      <div class="status-banner" :class="state.status">
        <div class="banner-left">
          <StatusBadge :status="state.status || 'pending'" />
          <span class="banner-id mono">{{ expId }}</span>
          <span v-if="state.policy_type" class="policy-chip">{{ state.policy_type }}</span>
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
        <div class="progress-fill" :class="{ active: isRunning }" :style="{ width: pct + '%' }"></div>
      </div>

      <!-- Agents Flock — show while running AND on completed runs -->
      <div v-if="isRunning || state.status === 'complete'" class="flock-section">
        <AgentsFlock
          :active="isRunning && !state.gpu_wait"
          :n-agents="agentCount"
          :bad-apple-frac="badAppleFrac"
          :show-llm-label="isLlmPolicy && isRunning"
        />
        <p class="flock-caption">
          <span v-if="state.gpu_wait">Agents initialising — waiting for GPU inference engine</span>
          <span v-else-if="isRunning && heartbeat">Round {{ heartbeat.round_id ?? heartbeat.round ?? '?' }} — agents deliberating</span>
          <span v-else-if="isRunning">Simulation starting…</span>
          <span v-else>{{ agentCount }} agents · {{ state.completed_rounds }} rounds completed</span>
        </p>
      </div>

      <!-- Main grid -->
      <div class="monitor-grid">

        <!-- Run details -->
        <div class="card details-card">
          <h2 class="section-title">Run Details</h2>
          <dl class="dl">
            <dt>Experiment ID</dt>
            <dd class="mono">{{ expId }}</dd>
            <dt>Policy</dt>
            <dd>
              <span class="badge" :class="isLlmPolicy ? 'badge-amber' : 'badge-teal'">
                {{ state.policy_type || '—' }}
              </span>
            </dd>
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

        <!-- GPU wait placeholder when no heartbeat yet -->
        <div class="card hb-card" v-else-if="state.gpu_wait">
          <h2 class="section-title">Live Heartbeat</h2>
          <div class="gpu-loading-rows">
            <div v-for="i in 4" :key="i" class="loading-row">
              <div class="loading-label shimmer"></div>
              <div class="loading-val shimmer"></div>
            </div>
          </div>
        </div>

        <!-- Actions -->
        <div class="card actions-card">
          <h2 class="section-title">Actions</h2>
          <div class="action-btns">
            <router-link v-if="state.status === 'complete'"
              :to="`/results/${expId}`" class="btn btn-primary">
              View Results →
            </router-link>
            <button v-if="isRunning" class="btn btn-inject" @click="injectOpen = true">
              <span class="inject-btn-icon">⚡</span> Inject Event
            </button>
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

    <!-- ── Inject modal ──────────────────────────────────────────── -->
    <teleport to="body">
      <transition name="modal">
        <div v-if="injectOpen" class="modal-backdrop" @click.self="closeInject">
          <div class="modal-panel">

            <!-- Header -->
            <div class="modal-head">
              <div class="modal-title-row">
                <span class="modal-icon">⚡</span>
                <div>
                  <div class="modal-title">Inject Event</div>
                  <div class="modal-sub">Send a live signal to <span class="mono">{{ expId }}</span></div>
                </div>
              </div>
              <button class="modal-close" @click="closeInject">✕</button>
            </div>

            <!-- Event type cards -->
            <div class="ev-types">
              <button v-for="t in eventTypes" :key="t.value"
                class="ev-card" :class="{ selected: injectType === t.value, [t.color]: true }"
                @click="selectType(t)">
                <span class="ev-icon">{{ t.icon }}</span>
                <div class="ev-info">
                  <div class="ev-name">{{ t.label }}</div>
                  <div class="ev-desc">{{ t.shortDesc }}</div>
                </div>
                <span v-if="injectType === t.value" class="ev-check">✓</span>
              </button>
            </div>

            <!-- ── Wealth Shock form ──────────────────────────── -->
            <div v-if="injectType === 'wealth_shock'" class="ev-form">
              <div class="ef-label">Wealth multiplier</div>
              <div class="factor-row">
                <span class="factor-tag" :class="factorClass">{{ wsFactor }}×</span>
                <input type="range" v-model.number="wsFactor"
                  min="0.1" max="2" step="0.05" class="slider ev-slider" />
                <span class="factor-hint">{{ wsFactorLabel }}</span>
              </div>
              <div class="factor-marks">
                <span>Crash<br><small>0.1×</small></span>
                <span>Halve<br><small>0.5×</small></span>
                <span>Neutral<br><small>1.0×</small></span>
                <span>Boom<br><small>2.0×</small></span>
              </div>
              <div class="ef-label" style="margin-top:16px">Target agents</div>
              <div class="target-pills">
                <button class="tpill" :class="{ active: wsTarget === 'all' }" @click="wsTarget = 'all'">All agents</button>
                <button class="tpill" :class="{ active: wsTarget === 'grounded' }" @click="wsTarget = 'grounded'">Grounded only</button>
                <button class="tpill" :class="{ active: wsTarget === 'bad' }" @click="wsTarget = 'bad'">Bad apples only</button>
              </div>
            </div>

            <!-- ── Signal Update form ────────────────────────── -->
            <div v-else-if="injectType === 'signal_update'" class="ev-form">
              <div class="ef-label">Economic climate</div>
              <div class="climate-grid">
                <button v-for="c in climates" :key="c.value"
                  class="climate-btn" :class="{ active: wsClimate === c.value, [c.cls]: true }"
                  @click="wsClimate = c.value">
                  <span class="cb-icon">{{ c.icon }}</span>
                  <span class="cb-name">{{ c.label }}</span>
                </button>
              </div>
              <div class="ef-label" style="margin-top:16px">
                Severity <span class="ef-val">{{ (wsSeverity * 100).toFixed(0) }}%</span>
              </div>
              <input type="range" v-model.number="wsSeverity"
                min="0.1" max="1" step="0.05" class="slider ev-slider" />
              <div class="slider-ticks" style="margin-top:4px">
                <span>Mild</span><span>Moderate</span><span>Severe</span>
              </div>
            </div>

            <!-- ── Narrative form ────────────────────────────── -->
            <div v-else-if="injectType === 'narrative'" class="ev-form">
              <div class="ef-label">Narrative presets</div>
              <div class="preset-chips">
                <button v-for="p in narrativePresets" :key="p"
                  class="nchip" @click="wsNarrative = p">{{ p }}</button>
              </div>
              <div class="ef-label" style="margin-top:14px">Custom narrative</div>
              <textarea v-model="wsNarrative" rows="3"
                class="ev-textarea"
                placeholder="Describe the event agents will observe…" />
              <div class="ef-hint">{{ wsNarrative.length }} / 300 characters</div>
            </div>

            <!-- Footer -->
            <div class="modal-foot">
              <div class="payload-preview mono">
                <span class="pp-label">Payload</span>
                {{ JSON.stringify(builtPayload, null, 0) }}
              </div>
              <div class="modal-actions">
                <button class="btn btn-outline btn-sm" @click="closeInject">Cancel</button>
                <button class="btn btn-inject-cta" @click="doInject" :disabled="injecting || !canInject">
                  <span v-if="injecting" class="spin">⟳</span>
                  <span v-else>⚡</span>
                  {{ injecting ? 'Sending…' : 'Inject Now' }}
                </button>
              </div>
            </div>

            <!-- Result toast -->
            <transition name="toast">
              <div v-if="injectResult" class="inject-toast" :class="injectIsErr ? 'err' : 'ok'">
                <span>{{ injectIsErr ? '✕' : '✓' }}</span>
                {{ injectResult }}
              </div>
            </transition>

          </div>
        </div>
      </transition>
    </teleport>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'
import StatusBadge from '../components/StatusBadge.vue'
import AgentsFlock from '../components/AgentsFlock.vue'

// ── Inject modal state ─────────────────────────────────────────────
const injectOpen   = ref(false)
const injectType   = ref('wealth_shock')
const injecting    = ref(false)
const injectResult = ref('')
const injectIsErr  = ref(false)

// Per-type form fields
const wsFactor    = ref(0.5)
const wsTarget    = ref('all')
const wsClimate   = ref('recession')
const wsSeverity  = ref(0.6)
const wsNarrative = ref('A major market crash has occurred.')

const eventTypes = [
  { value: 'wealth_shock',  icon: '▼', label: 'Wealth Shock',   shortDesc: 'Scale agent wealth',       color: 'rose'   },
  { value: 'signal_update', icon: '◎', label: 'Signal Update',  shortDesc: 'Change economic climate',  color: 'amber'  },
  { value: 'narrative',     icon: '▶', label: 'Narrative',       shortDesc: 'Broadcast a world event',  color: 'violet' },
]

const climates = [
  { value: 'recession',   icon: '▼', label: 'Recession',  cls: 'rose'   },
  { value: 'crisis',      icon: '▼▼',label: 'Crisis',     cls: 'rose'   },
  { value: 'neutral',     icon: '◎', label: 'Neutral',    cls: 'teal'   },
  { value: 'boom',        icon: '▲', label: 'Boom',       cls: 'green'  },
  { value: 'uncertainty', icon: '◈', label: 'Uncertain',  cls: 'amber'  },
  { value: 'recovery',    icon: '▲', label: 'Recovery',   cls: 'blue'   },
]

const narrativePresets = [
  'A major market crash has occurred.',
  'The government announced new social welfare policies.',
  'A technological breakthrough increased productivity.',
  'Trade sanctions reduced available resources.',
  'A natural disaster disrupted the local economy.',
]

const builtPayload = computed(() => {
  if (injectType.value === 'wealth_shock')
    return { factor: wsFactor.value, agent_ids: wsTarget.value }
  if (injectType.value === 'signal_update')
    return { economy: wsClimate.value, severity: wsSeverity.value }
  return { content: wsNarrative.value }
})

const canInject = computed(() => {
  if (injectType.value === 'narrative') return wsNarrative.value.trim().length > 0
  return true
})

const factorClass = computed(() => {
  if (wsFactor.value < 0.5)  return 'factor-crash'
  if (wsFactor.value < 0.9)  return 'factor-down'
  if (wsFactor.value < 1.1)  return 'factor-neutral'
  return 'factor-up'
})

const wsFactorLabel = computed(() => {
  if (wsFactor.value < 0.3)  return 'Severe crash'
  if (wsFactor.value < 0.7)  return 'Major loss'
  if (wsFactor.value < 0.95) return 'Minor loss'
  if (wsFactor.value < 1.05) return 'No change'
  if (wsFactor.value < 1.5)  return 'Growth'
  return 'Major boom'
})

function selectType(t) { injectType.value = t.value }

function closeInject() {
  injectOpen.value = false
  setTimeout(() => { injectResult.value = ''; injectIsErr.value = false }, 300)
}

async function doInject() {
  injecting.value = true; injectResult.value = ''; injectIsErr.value = false
  try {
    await api.inject(expId, injectType.value, builtPayload.value)
    injectResult.value = `${eventTypes.find(e => e.value === injectType.value)?.label} injected successfully`
    injectIsErr.value = false
    setTimeout(closeInject, 2200)
  } catch(e) {
    injectIsErr.value = true
    injectResult.value = e.response?.data?.error || e.message
  } finally {
    injecting.value = false
  }
}

const route     = useRoute()
const expId     = route.params.expId
const state     = ref({})
const heartbeat = ref(null)
const loading   = ref(true)
const notFound  = ref(false)
const polling   = ref(false)
let timer       = null

const isRunning   = computed(() => state.value.status === 'running' || state.value.status === 'pending')
const isLlmPolicy = computed(() => ['llm', 'generative_agents'].includes(state.value.policy_type))
const agentCount  = computed(() =>
  heartbeat.value?.n_agents
  ?? state.value.total_agents
  ?? state.value.metadata?.population_size
  ?? 10
)
const badAppleFrac = computed(() =>
  state.value.metadata?.bad_apple_frac
  ?? state.value.bad_apple_frac
  ?? 0
)

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

const staleMinutes = computed(() => {
  const ts = state.value.updated_at || state.value.started_at
  if (!ts) return 0
  return Math.floor((Date.now() / 1000 - ts) / 60)
})

const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : '—'

let notFoundRetries = 0
// After PATIENCE_RETRIES consecutive 404s, show the "not found" card —
// but keep polling so the user doesn't need F5 if startup is just slow.
const PATIENCE_RETRIES = 20  // 20 × 3s = 60s before showing not-found card

async function refresh() {
  try {
    const r = await api.status(expId)
    state.value     = r.data || {}
    heartbeat.value = r.data?.heartbeat || null
    notFound.value  = false
    notFoundRetries = 0
  } catch(e) {
    if (e.response?.status === 404) {
      notFoundRetries++
      if (notFoundRetries >= PATIENCE_RETRIES) notFound.value = true
    }
  }
}

function startPolling() {
  polling.value = true
  timer = setInterval(async () => {
    await refresh()
    // Stop only when the run reaches a terminal state; keep going on notFound
    // so the experiment is picked up automatically once the subprocess writes
    // run_state.json (no F5 required)
    if (['complete', 'failed'].includes(state.value.status)) stopPolling()
  }, 3000)
}

function stopPolling() {
  polling.value = false
  clearInterval(timer)
}

onMounted(async () => {
  await refresh()
  loading.value = false
  // Always poll — stops only on terminal status or component unmount
  startPolling()
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

/* ── GPU Wait banner ────────────────────────────────────────────── */
.gpu-wait-banner {
  display: flex; align-items: flex-start; gap: 14px;
  padding: 16px 18px; margin-bottom: 16px;
  background: rgba(245,158,11,.06);
  border: 1px solid rgba(245,158,11,.22);
  border-radius: 12px;
  animation: pulse-border 2.5s ease-in-out infinite;
}
@keyframes pulse-border {
  0%,100% { border-color: rgba(245,158,11,.22); }
  50%      { border-color: rgba(245,158,11,.45); }
}
.gw-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.gw-body { flex: 1; }
.gw-title { font-size: .88rem; font-weight: 600; color: var(--amber); margin-bottom: 4px; }
.gw-sub   { font-size: .78rem; color: var(--text2); line-height: 1.6; }
.gw-sub code { background: rgba(245,158,11,.12); padding: 1px 5px; border-radius: 4px; font-size: .76rem; color: var(--amber); }

.warn-box {
  background: rgba(245,158,11,.06); border: 1px solid rgba(245,158,11,.2);
  border-radius: 10px; padding: 11px 14px; font-size: .82rem; color: var(--amber);
}

/* ── Status banner ──────────────────────────────────────────────── */
.status-banner {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
  padding: 13px 18px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  margin-bottom: 18px;
  transition: border-color .3s;
}
.status-banner.running  { border-color: rgba(99,102,241,.3); }
.status-banner.complete { border-color: rgba(16,185,129,.3); }
.status-banner.failed   { border-color: rgba(244,63,94,.3); }

.banner-left  { display: flex; align-items: center; gap: 12px; }
.banner-right { display: flex; align-items: center; gap: 10px; }
.banner-id    { font-size: .82rem; color: var(--text2); }
.updated-at   { font-size: .74rem; color: var(--text3); }
.policy-chip  {
  font-size: .68rem; padding: 2px 8px; border-radius: 20px;
  background: rgba(99,102,241,.1); color: var(--blue2);
  border: 1px solid rgba(99,102,241,.2); font-family: monospace;
}
.poll-pill {
  display: flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 20px;
  background: rgba(99,102,241,.1); color: var(--blue2);
  font-size: .72rem;
}

/* ── KPIs ───────────────────────────────────────────────────────── */
.monitor-kpis { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 16px; }
.mkpi {
  background: rgba(13,21,40,.85); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 22px; min-width: 110px;
  transition: border-color .25s, box-shadow .25s;
}
.mkpi:first-child { border-color: rgba(99,102,241,.2); box-shadow: 0 0 30px rgba(99,102,241,.07); }
.mkpi-val { font-size: 2.1rem; font-weight: 800; display: flex; align-items: baseline; gap: 4px; letter-spacing: -.03em; }
.mkpi-val.running  { color: var(--blue2); }
.mkpi-val.complete { color: var(--green); }
.mkpi-sep   { font-size: 1.2rem; color: var(--text3); font-weight: 400; }
.mkpi-total { font-size: 1.2rem; color: var(--text2); font-weight: 600; }
.mkpi-label { font-size: .68rem; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; margin-top: 5px; font-weight: 600; }

/* ── Progress bar ───────────────────────────────────────────────── */
.progress-track {
  height: 6px; background: var(--bg4); border-radius: 99px;
  overflow: hidden; margin-bottom: 22px;
}
.progress-fill {
  height: 100%; background: var(--grad);
  border-radius: 99px; transition: width .6s var(--ease-out);
  box-shadow: 0 0 12px rgba(99,102,241,.4);
}
.progress-fill.active {
  background-size: 30px 100%;
  animation: progress-stripe 1.5s linear infinite;
  background-image: linear-gradient(135deg, rgba(255,255,255,.12) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.12) 50%, rgba(255,255,255,.12) 75%, transparent 75%, transparent);
  background-color: var(--blue);
}
@keyframes progress-stripe {
  0% { background-position: 0 0; }
  100% { background-position: 30px 0; }
}

/* ── Flock section ──────────────────────────────────────────────── */
.flock-section { margin-bottom: 22px; }
.flock-caption {
  font-size: .76rem; color: var(--text3); text-align: center;
  margin-top: 8px;
}

/* ── Grid ───────────────────────────────────────────────────────── */
.monitor-grid {
  display: grid; grid-template-columns: 1fr 1fr 200px;
  gap: 16px; align-items: start;
}
@media (max-width: 860px) { .monitor-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .monitor-grid { grid-template-columns: 1fr; } }

/* ── DL ─────────────────────────────────────────────────────────── */
.dl { display: grid; grid-template-columns: auto 1fr; gap: 7px 16px; }
dt { font-size: .76rem; color: var(--text3); font-weight: 500; align-self: center; }
dd { font-size: .83rem; color: var(--text); }

.action-btns { display: flex; flex-direction: column; gap: 9px; }

/* ── GPU shimmer placeholders ───────────────────────────────────── */
.gpu-loading-rows { display: flex; flex-direction: column; gap: 12px; }
.loading-row { display: flex; justify-content: space-between; align-items: center; }
.loading-label { width: 60px; height: 10px; border-radius: 5px; }
.loading-val   { width: 80px; height: 10px; border-radius: 5px; }
.shimmer {
  background: linear-gradient(90deg, var(--bg3) 25%, rgba(99,102,241,.08) 50%, var(--bg3) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.8s infinite;
}
@keyframes shimmer { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }

/* ── Badge variants ─────────────────────────────────────────────── */
.badge { font-size: .7rem; padding: 2px 8px; border-radius: 20px; }
.badge-amber { background: rgba(245,158,11,.12); color: var(--amber); }
.badge-teal  { background: rgba(20,184,166,.12);  color: var(--teal); }

/* ── Inject button (actions card) ───────────────────────────────── */
.btn-inject {
  background: linear-gradient(135deg, rgba(245,158,11,.15), rgba(239,68,68,.1));
  color: var(--amber); border: 1px solid rgba(245,158,11,.3);
  font-weight: 600;
}
.btn-inject:hover {
  background: linear-gradient(135deg, rgba(245,158,11,.25), rgba(239,68,68,.15));
  border-color: rgba(245,158,11,.5);
  box-shadow: 0 4px 20px rgba(245,158,11,.2);
}
.inject-btn-icon { font-size: .9rem; }

/* ── Modal backdrop ─────────────────────────────────────────────── */
.modal-backdrop {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,.65);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  padding: 20px;
}
.modal-enter-active { transition: opacity .2s var(--ease-out); }
.modal-leave-active { transition: opacity .18s ease; }
.modal-enter-from, .modal-leave-to { opacity: 0; }
.modal-enter-active .modal-panel { animation: modal-pop .3s var(--ease-spring) both; }
@keyframes modal-pop {
  from { opacity:0; transform: scale(.94) translateY(16px); }
  to   { opacity:1; transform: none; }
}

/* ── Modal panel ────────────────────────────────────────────────── */
.modal-panel {
  width: 100%; max-width: 520px;
  background: rgba(8,15,30,.97);
  backdrop-filter: blur(32px);
  border: 1px solid rgba(99,102,241,.2);
  border-radius: 20px;
  box-shadow: 0 24px 80px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.04) inset;
  overflow: hidden;
  position: relative;
}

.modal-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  padding: 22px 24px 18px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(245,158,11,.06), rgba(99,102,241,.04));
}
.modal-title-row { display: flex; align-items: center; gap: 14px; }
.modal-icon {
  width: 40px; height: 40px; border-radius: 12px;
  background: linear-gradient(135deg, rgba(245,158,11,.2), rgba(239,68,68,.12));
  border: 1px solid rgba(245,158,11,.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.modal-title { font-size: 1rem; font-weight: 700; color: var(--text); }
.modal-sub   { font-size: .78rem; color: var(--text3); margin-top: 2px; }
.modal-close {
  width: 28px; height: 28px; border-radius: 8px;
  background: rgba(255,255,255,.05); border: 1px solid var(--border);
  color: var(--text3); font-size: .75rem; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .15s; flex-shrink: 0;
}
.modal-close:hover { background: rgba(244,63,94,.15); color: var(--rose); border-color: rgba(244,63,94,.3); }

/* ── Event type cards ───────────────────────────────────────────── */
.ev-types { display: flex; flex-direction: column; gap: 6px; padding: 18px 24px 0; }
.ev-card {
  display: flex; align-items: center; gap: 14px;
  padding: 12px 14px; border-radius: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  cursor: pointer; text-align: left;
  transition: border-color .15s, background .15s, transform .2s var(--ease-spring);
}
.ev-card:hover { border-color: var(--border2); transform: translateX(3px); }
.ev-card.selected.rose   { border-color: rgba(244,63,94,.4);  background: rgba(244,63,94,.07);  }
.ev-card.selected.amber  { border-color: rgba(245,158,11,.4); background: rgba(245,158,11,.07); }
.ev-card.selected.violet { border-color: rgba(139,92,246,.4); background: rgba(139,92,246,.07); }
.ev-icon { font-size: 1.1rem; color: var(--text2); flex-shrink: 0; width: 22px; text-align: center; }
.ev-card.selected.rose   .ev-icon { color: var(--rose); }
.ev-card.selected.amber  .ev-icon { color: var(--amber); }
.ev-card.selected.violet .ev-icon { color: var(--violet); }
.ev-info { flex: 1; }
.ev-name { font-size: .84rem; font-weight: 600; color: var(--text); }
.ev-desc { font-size: .72rem; color: var(--text3); margin-top: 1px; }
.ev-check { color: var(--green); font-size: .8rem; flex-shrink: 0; }

/* ── Event forms ────────────────────────────────────────────────── */
.ev-form { padding: 18px 24px; border-top: 1px solid var(--border); margin-top: 14px; }
.ef-label { font-size: .7rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px; }
.ef-val   { color: var(--blue2); font-weight: 700; margin-left: 6px; }
.ef-hint  { font-size: .7rem; color: var(--text3); margin-top: 5px; text-align: right; }

/* Wealth shock */
.factor-row { display: flex; align-items: center; gap: 14px; margin-bottom: 8px; }
.factor-tag {
  min-width: 46px; text-align: center; font-size: .9rem; font-weight: 800;
  padding: 4px 10px; border-radius: 8px; letter-spacing: -.02em;
  flex-shrink: 0;
}
.factor-crash   { background: rgba(244,63,94,.12); color: var(--rose); }
.factor-down    { background: rgba(245,158,11,.12); color: var(--amber); }
.factor-neutral { background: rgba(20,184,166,.1);  color: var(--teal); }
.factor-up      { background: rgba(16,185,129,.1);  color: var(--green); }
.factor-hint    { font-size: .76rem; color: var(--text3); min-width: 80px; }
.factor-marks {
  display: flex; justify-content: space-between;
  font-size: .66rem; color: var(--text3); text-align: center; line-height: 1.3;
}
.ev-slider { flex: 1; }

/* Target pills */
.target-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.tpill {
  padding: 5px 13px; border-radius: 99px;
  font-size: .76rem; font-weight: 500;
  background: var(--bg4); border: 1px solid var(--border);
  color: var(--text3); cursor: pointer; transition: all .15s;
}
.tpill:hover  { border-color: var(--border2); color: var(--text2); }
.tpill.active { background: rgba(99,102,241,.12); border-color: rgba(99,102,241,.3); color: var(--blue2); }

/* Climate grid */
.climate-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
.climate-btn {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 10px 8px; border-radius: 10px;
  background: var(--bg4); border: 1px solid var(--border);
  cursor: pointer; transition: all .15s;
}
.climate-btn:hover  { border-color: var(--border2); }
.climate-btn.active.rose   { background: rgba(244,63,94,.1); border-color: rgba(244,63,94,.35); }
.climate-btn.active.green  { background: rgba(16,185,129,.1); border-color: rgba(16,185,129,.35); }
.climate-btn.active.teal   { background: rgba(20,184,166,.1); border-color: rgba(20,184,166,.35); }
.climate-btn.active.amber  { background: rgba(245,158,11,.1); border-color: rgba(245,158,11,.35); }
.climate-btn.active.blue   { background: rgba(99,102,241,.1); border-color: rgba(99,102,241,.35); }
.cb-icon { font-size: .9rem; color: var(--text3); }
.cb-name { font-size: .72rem; font-weight: 600; color: var(--text2); }
.climate-btn.active .cb-icon,
.climate-btn.active .cb-name { color: var(--text); }

/* Narrative */
.preset-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.nchip {
  font-size: .73rem; padding: 4px 11px; border-radius: 99px;
  background: var(--bg4); border: 1px solid var(--border);
  color: var(--text2); cursor: pointer; text-align: left; line-height: 1.4;
  transition: border-color .15s, background .15s;
}
.nchip:hover { border-color: rgba(139,92,246,.35); background: rgba(139,92,246,.07); color: var(--violet); }
.ev-textarea {
  width: 100%; background: var(--bg3); border: 1px solid var(--border2);
  border-radius: 10px; color: var(--text); font-family: inherit;
  font-size: .84rem; padding: 10px 13px; resize: vertical;
  transition: border-color .15s, box-shadow .15s;
  line-height: 1.6;
}
.ev-textarea:focus { outline: none; border-color: var(--blue); box-shadow: 0 0 0 3px rgba(99,102,241,.12); }

/* ── Modal footer ───────────────────────────────────────────────── */
.modal-foot {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 16px 24px 20px;
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
}
.payload-preview {
  font-size: .68rem; color: var(--text3);
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 7px; padding: 5px 10px;
  max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  flex: 1;
}
.pp-label {
  font-size: .6rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; color: var(--text3); margin-right: 6px;
}
.modal-actions { display: flex; gap: 8px; flex-shrink: 0; }

.btn-inject-cta {
  background: linear-gradient(135deg, #d97706, #f59e0b);
  color: #000; border: none; font-weight: 700;
  padding: 8px 20px; border-radius: 10px; font-size: .875rem;
  box-shadow: 0 3px 16px rgba(245,158,11,.35);
  transition: transform .3s var(--ease-spring), box-shadow .2s, filter .2s;
}
.btn-inject-cta:not(:disabled):hover {
  transform: translateY(-2px) scale(1.03);
  box-shadow: 0 6px 28px rgba(245,158,11,.5);
  filter: brightness(1.08);
}
.btn-inject-cta:disabled { opacity: .5; cursor: not-allowed; }

/* ── Result toast ───────────────────────────────────────────────── */
.inject-toast {
  position: absolute; bottom: 80px; left: 50%; transform: translateX(-50%);
  display: flex; align-items: center; gap: 8px;
  padding: 10px 20px; border-radius: 99px;
  font-size: .84rem; font-weight: 600;
  white-space: nowrap;
  box-shadow: 0 8px 32px rgba(0,0,0,.4);
}
.inject-toast.ok  { background: rgba(16,185,129,.15); color: var(--green); border: 1px solid rgba(16,185,129,.3); }
.inject-toast.err { background: rgba(244,63,94,.12);  color: var(--rose);  border: 1px solid rgba(244,63,94,.25); }
.toast-enter-active { transition: opacity .25s, transform .3s var(--ease-spring); }
.toast-leave-active { transition: opacity .2s, transform .2s ease; }
.toast-enter-from   { opacity: 0; transform: translateX(-50%) translateY(10px) scale(.95); }
.toast-leave-to     { opacity: 0; transform: translateX(-50%) translateY(6px); }

/* Range slider override for event forms */
.ev-slider {
  -webkit-appearance: none; appearance: none;
  height: 4px; background: var(--bg5); border-radius: 2px; border: none; outline: none; padding: 0;
}
.ev-slider::-webkit-slider-thumb {
  -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
  background: var(--amber); cursor: pointer;
  box-shadow: 0 2px 8px rgba(245,158,11,.4);
  transition: transform .2s var(--ease-spring);
}
.ev-slider::-webkit-slider-thumb:hover { transform: scale(1.3); }
.slider-ticks { display: flex; justify-content: space-between; font-size: .66rem; color: var(--text3); }
</style>
