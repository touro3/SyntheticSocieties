<template>
  <div>
    <div class="page-header">
      <div class="breadcrumb">
        <router-link to="/experiments">Experiments</router-link>
        <span>›</span>
        <router-link :to="`/results/${expId}`">{{ expId }}</router-link>
        <span>›</span>
        <span>Interact</span>
      </div>
      <h1 class="page-title" style="margin-top:8px">Agent Interaction</h1>
      <p class="page-sub">
        Interview agents from completed runs or inject live events into running simulations.
      </p>
    </div>

    <div class="interact-layout">

      <!-- ── Interview panel ────────────────────────────────────── -->
      <div class="card panel">
        <h2 class="section-title">Interview Agent</h2>

        <!-- IPC warning — only shown if live interview was attempted and IPC unavailable AND no events fallback worked -->
        <div v-if="ipcDown && !hasReplied" class="warn-box">
          <strong>Live IPC not available.</strong><br>
          This simulation is not running. Interview will use the saved event log instead — just ask your question.
          <button class="btn btn-ghost btn-sm" style="margin-top:8px" @click="ipcDown = false">Dismiss</button>
        </div>

        <!-- Agent ID input -->
        <div class="field">
          <label>Agent ID</label>
          <div class="agent-input-row">
            <input v-model="agentId" placeholder="agent_0" @keydown.enter="ask" />
            <div class="agent-presets">
              <button v-for="n in [0,1,2,3,4]" :key="n"
                class="preset-btn" :class="{ active: agentId === `agent_${n}` }"
                @click="agentId = `agent_${n}`">
                {{ n }}
              </button>
            </div>
          </div>
        </div>

        <!-- Agent tab strip (show tabs when >1 agent has been chatted) -->
        <div v-if="chattedAgents.length > 1" class="agent-tabs">
          <button v-for="a in chattedAgents" :key="a"
            class="agent-tab" :class="{ active: agentId === a }"
            @click="agentId = a">
            {{ a }}
            <span class="atab-count">{{ (chatsByAgent[a] ?? []).length / 2 | 0 }}</span>
          </button>
        </div>

        <!-- Chat window — per-agent conversation -->
        <div class="chat-window" ref="chatRef">
          <div v-if="!currentMessages.length" class="chat-empty">
            Ask <strong>{{ agentId }}</strong> about their decisions, strategy, or wealth.
          </div>
          <template v-for="(m, i) in currentMessages" :key="i">
            <div class="msg" :class="m.role">
              <div class="msg-label">
                {{ m.sender }}
                <span v-if="m.replay" class="replay-tag">replay</span>
              </div>
              <div class="msg-body">{{ m.text }}</div>
            </div>
          </template>
          <div v-if="thinking" class="msg agent">
            <div class="msg-label">{{ agentId }}</div>
            <div class="msg-body thinking">
              <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
          </div>
        </div>

        <!-- Quick questions -->
        <div class="quick-qs">
          <span class="qq-label">Quick:</span>
          <button v-for="q in quickQs" :key="q" class="qq-btn" @click="setQ(q)">{{ q }}</button>
        </div>

        <!-- Input bar -->
        <div class="chat-bar">
          <input v-model="question" placeholder="Ask something…"
            @keydown.enter="ask" :disabled="thinking" />
          <button class="btn btn-primary" @click="ask"
            :disabled="thinking || !agentId.trim() || !question.trim()">
            <span v-if="thinking" class="spin">⟳</span>
            <span v-else>Send</span>
          </button>
        </div>

        <div v-if="chatErr" class="error-box">{{ chatErr }}</div>
      </div>

      <!-- ── Inject panel ───────────────────────────────────────── -->
      <div class="card panel">
        <div class="inject-header">
          <h2 class="section-title" style="margin:0">Inject Event</h2>
          <span class="sim-status-chip" :class="simRunning ? 'live' : 'stopped'">
            <span class="ssc-dot"></span>
            {{ simRunning ? 'Live' : simStatus }}
          </span>
        </div>

        <!-- Not-running notice -->
        <div v-if="!simRunning" class="inject-offline-notice">
          <div class="ion-icon">◌</div>
          <div class="ion-body">
            <div class="ion-title">Simulation not running</div>
            <div class="ion-desc">
              Inject requires an actively running simulation.
              <router-link to="/run" class="ion-link">Start a new run →</router-link>
            </div>
          </div>
        </div>

        <template v-else>
          <p style="font-size:.82rem;color:var(--text2);margin-bottom:16px;line-height:1.5">
            Inject exogenous events into the running simulation — no restart required.
          </p>

          <!-- Event type selector -->
          <div class="event-types">
            <button v-for="t in eventTypes" :key="t.value"
              class="event-btn" :class="{ selected: injectType === t.value }"
              @click="selectEvent(t)">
              <span class="et-icon">{{ t.icon }}</span>
              <div class="et-body">
                <div class="et-name">{{ t.label }}</div>
                <div class="et-desc">{{ t.desc }}</div>
              </div>
            </button>
          </div>

          <!-- Payload editor -->
          <div class="field">
            <label>Payload (JSON)</label>
            <textarea v-model="injectPayload" rows="4" class="mono"
              style="font-size:.8rem;resize:vertical" spellcheck="false" />
            <span v-if="jsonErr" class="hint" style="color:var(--rose)">
              Invalid JSON — {{ jsonErr }}
            </span>
          </div>

          <button class="btn btn-outline" style="width:100%;justify-content:center"
            @click="doInject" :disabled="injecting || !!jsonErr">
            <span v-if="injecting" class="spin">⟳</span>
            {{ injecting ? 'Injecting…' : 'Inject Event' }}
          </button>

          <div v-if="injectMsg" class="inject-result"
            :class="injectIsErr ? 'error-box' : 'success-box'">
            <pre style="font-size:.75rem;white-space:pre-wrap;max-height:120px;overflow-y:auto">{{ injectMsg }}</pre>
          </div>
        </template>

        <!-- Schema docs always visible -->
        <div class="schema-docs">
          <div class="schema-title">Payload schemas</div>
          <div v-for="t in eventTypes" :key="t.value" class="schema-item">
            <span class="mono" style="font-size:.76rem;color:var(--text2)">{{ t.value }}</span>
            <code class="mono schema-code">{{ t.schema }}</code>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'

const route  = useRoute()
const expId  = route.params.expId

// ── Simulation status ──────────────────────────────────────────────
const simStatus  = ref('unknown')
const simRunning = computed(() => ['running', 'pending'].includes(simStatus.value))

// ── Interview state ────────────────────────────────────────────────
const agentId   = ref('agent_0')
const question  = ref('')
// Per-agent conversation store: { 'agent_0': [{role, sender, text, replay}, …], … }
const chatsByAgent = ref({})
const thinking  = ref(false)
const chatErr   = ref('')
const ipcDown   = ref(false)
const chatRef   = ref(null)

// Derived: messages for the currently selected agent
const currentMessages = computed(() => chatsByAgent.value[agentId.value] ?? [])

// Agents we've talked to (for tab strip)
const chattedAgents = computed(() => Object.keys(chatsByAgent.value))

// ── Inject state ───────────────────────────────────────────────────
const injectType    = ref('wealth_shock')
const injectPayload = ref('{"factor": 0.5, "agent_ids": "all"}')
const injecting     = ref(false)
const injectMsg     = ref('')
const injectIsErr   = ref(false)

const quickQs = [
  'What was your last decision and why?',
  'How do you feel about your current wealth?',
  'Do you trust your neighbors?',
  'What would change your strategy?',
]

const eventTypes = [
  {
    value: 'wealth_shock', icon: '▼', label: 'Wealth Shock',
    desc: 'Multiply all agent wealth by a factor',
    schema: '{"factor": 0.5, "agent_ids": "all"}',
  },
  {
    value: 'signal_update', icon: '◎', label: 'Signal Update',
    desc: 'Change the public economic signal',
    schema: '{"economy": "recession", "severity": 0.8}',
  },
  {
    value: 'narrative', icon: '▶', label: 'Narrative',
    desc: 'Inject a narrative agents observe',
    schema: '{"content": "A market crash occurred."}',
  },
]

const hasReplied = computed(() =>
  currentMessages.value.some(m => m.role === 'agent' && m.text && !m.text.startsWith('(simulation IPC'))
)

// Validate JSON live
const jsonErr = computed(() => {
  try { JSON.parse(injectPayload.value); return '' }
  catch(e) { return e.message.split('\n')[0] }
})

function selectEvent(t) {
  injectType.value    = t.value
  injectPayload.value = t.schema
}

function setQ(q) { question.value = q }

async function scrollChat() {
  await nextTick()
  if (chatRef.value) chatRef.value.scrollTop = chatRef.value.scrollHeight
}

function pushMsg(id, msg) {
  if (!chatsByAgent.value[id]) chatsByAgent.value[id] = []
  chatsByAgent.value[id].push(msg)
  // trigger reactivity
  chatsByAgent.value = { ...chatsByAgent.value }
}

async function ask() {
  if (!question.value.trim() || !agentId.value.trim()) return
  const q  = question.value.trim()
  const id = agentId.value.trim()
  pushMsg(id, { role: 'user', sender: 'You', text: q })
  question.value = ''
  thinking.value = true
  chatErr.value  = ''
  ipcDown.value  = false
  await scrollChat()

  try {
    const r = await api.interview(expId, id, q)
    const reply = r.data?.response ?? r.data?.answer ?? r.data?.reply
      ?? (typeof r.data === 'string' ? r.data : JSON.stringify(r.data, null, 2))
    const source = r.data?.source
    pushMsg(id, {
      role: 'agent',
      sender: id,
      text: reply,
      replay: source === 'replay_data' || source === 'replay_llm',
    })
  } catch(e) {
    const errMsg = e.response?.data?.error || e.message || 'Unknown error'
    if (errMsg.toLowerCase().includes('ipc') || errMsg.toLowerCase().includes('timeout') || e.response?.status === 504) {
      ipcDown.value = true
      pushMsg(id, { role: 'agent', sender: id, text: '(simulation IPC not available — interview works on completed runs via event log)' })
    } else {
      chatErr.value = errMsg
      pushMsg(id, { role: 'agent', sender: id, text: '(no response)' })
    }
  } finally {
    thinking.value = false
    await scrollChat()
  }
}

async function doInject() {
  injecting.value = true; injectMsg.value = ''; injectIsErr.value = false
  try {
    const payload = JSON.parse(injectPayload.value)
    const r = await api.inject(expId, injectType.value, payload)
    injectMsg.value = JSON.stringify(r.data, null, 2)
    injectIsErr.value = false
  } catch(e) {
    injectIsErr.value = true
    injectMsg.value   = e.response?.data?.error || e.message
  } finally { injecting.value = false }
}

// Scroll to bottom when switching between agent tabs
watch(agentId, () => scrollChat())

onMounted(async () => {
  try {
    const r = await api.status(expId)
    simStatus.value = r.data?.status ?? 'unknown'
  } catch { simStatus.value = 'unknown' }
})
</script>

<style scoped>
.page-header { margin-bottom: 24px; }
.breadcrumb {
  display: flex; align-items: center; gap: 6px;
  font-size: .8rem; color: var(--text2); margin-bottom: 8px;
}
.breadcrumb a { color: var(--text2); }
.breadcrumb a:hover { color: #fff; text-decoration: none; }
.ipc-note {
  display: block; margin-top: 5px;
  font-size: .8rem; color: var(--amber);
}

.interact-layout {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 20px; align-items: start;
}
@media (max-width: 860px) { .interact-layout { grid-template-columns: 1fr; } }

.panel { display: flex; flex-direction: column; gap: 16px; }

.warn-box {
  background: rgba(245,158,11,.07); border: 1px solid rgba(245,158,11,.22);
  border-radius: 10px; padding: 12px 14px; font-size: .82rem; color: var(--amber);
  line-height: 1.6;
}
.warn-box code { background: rgba(245,158,11,.15); padding: 1px 5px; border-radius: 4px; font-size: .8rem; }

/* ── Agent tabs ─────────────────────────────────────────────────── */
.agent-tabs {
  display: flex; gap: 4px; flex-wrap: wrap;
  padding: 0 0 12px;
  border-bottom: 1px solid var(--border); margin-bottom: 4px;
}
.agent-tab {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 8px;
  font-size: .76rem; font-weight: 500;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text3); cursor: pointer;
  transition: all .15s;
}
.agent-tab:hover { color: var(--text2); border-color: var(--border2); }
.agent-tab.active {
  background: rgba(99,102,241,.1);
  border-color: rgba(99,102,241,.25);
  color: var(--blue2);
}
.atab-count {
  font-size: .65rem; padding: 1px 5px; border-radius: 99px;
  background: rgba(99,102,241,.15); color: var(--blue2);
  font-weight: 700;
}

/* ── Agent ID ───────────────────────────────────────────────────── */
.agent-input-row { display: flex; gap: 8px; align-items: center; }
.agent-input-row input { flex: 1; }
.agent-presets { display: flex; gap: 5px; flex-shrink: 0; }
.preset-btn {
  width: 28px; height: 28px; border-radius: 7px;
  background: var(--bg3); border: 1px solid var(--border);
  font-size: .74rem; color: var(--text2); cursor: pointer;
  transition: all .15s;
}
.preset-btn:hover, .preset-btn.active {
  background: rgba(99,102,241,.12); border-color: var(--blue); color: var(--blue2);
}

/* ── Chat ───────────────────────────────────────────────────────── */
.chat-window {
  background: rgba(5,10,21,.9); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px;
  min-height: 220px; max-height: 400px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 14px;
  scroll-behavior: smooth;
}
.chat-empty {
  color: var(--text3); font-size: .84rem;
  text-align: center; padding: 32px 0;
  line-height: 1.7;
}

.msg { display: flex; flex-direction: column; gap: 5px; animation: slide-in .2s var(--ease-spring) both; }
.msg-label {
  font-size: .67rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
  display: flex; align-items: center; gap: 7px;
}
.replay-tag {
  font-size: .58rem; padding: 1px 6px; border-radius: 4px;
  background: rgba(245,158,11,.12); color: var(--amber);
  border: 1px solid rgba(245,158,11,.2);
  font-weight: 700; letter-spacing: .04em;
}
.msg.user  .msg-label { color: var(--blue2); }
.msg.agent .msg-label { color: var(--cyan); }
.msg-body {
  font-size: .84rem; color: var(--text); line-height: 1.7;
  border-radius: 12px; padding: 10px 14px;
  border: 1px solid var(--border);
}
.msg.agent .msg-body {
  background: rgba(13,21,40,.9);
  border-color: rgba(34,211,238,.12);
  box-shadow: 0 0 20px rgba(34,211,238,.04);
}
.msg.user .msg-body {
  background: rgba(99,102,241,.1);
  border-color: rgba(99,102,241,.2);
}

.thinking { display: flex; gap: 5px; align-items: center; padding: 14px 16px !important; background: rgba(13,21,40,.9); }
.dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan); opacity: .6;
  animation: bounce 1.3s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0) scale(1); opacity:.6; } 40% { transform: translateY(-5px) scale(1.2); opacity:1; } }

/* ── Quick questions ────────────────────────────────────────────── */
.quick-qs { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.qq-label { font-size: .68rem; color: var(--text3); flex-shrink: 0; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
.qq-btn {
  font-size: .73rem; padding: 4px 11px; border-radius: 99px;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text2); cursor: pointer;
  transition: border-color .15s, color .15s, background .15s, transform .2s var(--ease-spring);
}
.qq-btn:hover {
  border-color: rgba(34,211,238,.3);
  color: var(--cyan);
  background: rgba(34,211,238,.05);
  transform: translateY(-1px);
}

/* ── Chat bar ───────────────────────────────────────────────────── */
.chat-bar { display: flex; gap: 8px; }
.chat-bar input { flex: 1; }

/* ── Inject header ──────────────────────────────────────────────── */
.inject-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.sim-status-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 99px;
  font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
}
.sim-status-chip.live {
  background: rgba(16,185,129,.1); color: var(--green);
  border: 1px solid rgba(16,185,129,.2);
}
.sim-status-chip.stopped {
  background: rgba(100,116,139,.08); color: var(--text3);
  border: 1px solid rgba(100,116,139,.15);
}
.ssc-dot {
  width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0;
}
.sim-status-chip.live .ssc-dot { animation: pulse 2s infinite; }

/* ── Inject offline notice ──────────────────────────────────────── */
.inject-offline-notice {
  display: flex; gap: 14px; align-items: flex-start;
  padding: 16px; border-radius: 12px;
  background: rgba(22,28,48,.6); border: 1px solid var(--border);
  margin-bottom: 16px;
}
.ion-icon { font-size: 1.4rem; color: var(--text3); flex-shrink: 0; margin-top: 2px; }
.ion-title { font-size: .88rem; font-weight: 600; color: var(--text2); margin-bottom: 5px; }
.ion-desc  { font-size: .8rem; color: var(--text3); line-height: 1.6; }
.ion-link  { color: var(--blue2); margin-left: 4px; }
.ion-link:hover { color: var(--violet); }

/* ── Event types ────────────────────────────────────────────────── */
.event-types { display: flex; flex-direction: column; gap: 8px; }
.event-btn {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 14px; border-radius: 11px;
  background: var(--bg3); border: 1px solid var(--border);
  color: #fff;
  cursor: pointer; text-align: left;
  transition: border-color .2s, background .2s, transform .25s var(--ease-spring);
}
.event-btn:hover   { border-color: rgba(99,102,241,.35); transform: translateX(2px); }
.event-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.12); }
.et-icon { font-size: 1rem; color: var(--blue2); font-family: monospace; flex-shrink: 0; }
.et-name { font-size: .84rem; font-weight: 600; color: #fff; }
.et-desc { font-size: .74rem; color: rgba(255,255,255,.5); }

.inject-result pre { margin: 0; }

/* ── Schema docs ────────────────────────────────────────────────── */
.schema-docs { border-top: 1px solid var(--border); padding-top: 14px; display: flex; flex-direction: column; gap: 8px; }
.schema-title { font-size: .74rem; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; }
.schema-item { display: flex; flex-direction: column; gap: 3px; }
.schema-code {
  font-size: .72rem; color: var(--teal);
  background: rgba(20,184,166,.07); padding: 4px 8px; border-radius: 6px;
  white-space: pre-wrap;
}
</style>
