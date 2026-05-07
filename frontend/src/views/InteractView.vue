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
        Interview agents or inject live events.
        <span v-if="!simRunning" class="ipc-note">
          ⚠ IPC bridge requires a running simulation. Inject events still work on past runs.
        </span>
      </p>
    </div>

    <div class="interact-layout">

      <!-- ── Interview panel ────────────────────────────────────── -->
      <div class="card panel">
        <h2 class="section-title">Interview Agent</h2>

        <!-- IPC warning -->
        <div v-if="ipcDown" class="warn-box">
          <strong>Simulation not running via IPC.</strong><br>
          The interview endpoint requires an active simulation with
          <code>SimulationIPCServer</code> enabled. Start or resume a run first.
          <div style="margin-top:10px;display:flex;gap:8px">
            <router-link to="/run" class="btn btn-primary btn-sm">Launch a run →</router-link>
            <button class="btn btn-ghost btn-sm" @click="ipcDown = false">Dismiss</button>
          </div>
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

        <!-- Chat window -->
        <div class="chat-window" ref="chatRef">
          <div v-if="!messages.length" class="chat-empty">
            Ask an agent about their decisions, strategy, or feelings about the economy.
          </div>
          <template v-for="(m, i) in messages" :key="i">
            <div class="msg" :class="m.role">
              <div class="msg-label">{{ m.role === 'user' ? 'You' : agentId }}</div>
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
        <h2 class="section-title">Inject Event</h2>
        <p style="font-size:.82rem;color:var(--text2);margin-bottom:16px">
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

        <!-- Schema docs -->
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
import { ref, computed, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'

const route  = useRoute()
const expId  = route.params.expId

// ── Interview state ────────────────────────────────────────────────
const agentId   = ref('agent_0')
const question  = ref('')
const messages  = ref([])
const thinking  = ref(false)
const chatErr   = ref('')
const ipcDown   = ref(false)
const simRunning = ref(true)
const chatRef   = ref(null)

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
    value: 'wealth_shock', icon: '💸', label: 'Wealth Shock',
    desc: 'Multiply all agent wealth by a factor',
    schema: '{"factor": 0.5, "agent_ids": "all"}',
  },
  {
    value: 'signal_update', icon: '📡', label: 'Signal Update',
    desc: 'Change the public economic signal',
    schema: '{"economy": "recession", "severity": 0.8}',
  },
  {
    value: 'narrative', icon: '📢', label: 'Narrative',
    desc: 'Inject a narrative agents observe',
    schema: '{"content": "A market crash occurred."}',
  },
]

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

async function ask() {
  if (!question.value.trim() || !agentId.value.trim()) return
  const q = question.value.trim()
  messages.value.push({ role: 'user', text: q })
  question.value = ''
  thinking.value = true
  chatErr.value  = ''
  ipcDown.value  = false
  await scrollChat()

  try {
    const r = await api.interview(expId, agentId.value.trim(), q)
    const reply = r.data?.response ?? r.data?.answer ?? r.data?.reply
      ?? (typeof r.data === 'string' ? r.data : JSON.stringify(r.data, null, 2))
    messages.value.push({ role: 'agent', text: reply })
  } catch(e) {
    const errMsg = e.response?.data?.error || e.message || 'Unknown error'
    // Detect IPC-specific failures gracefully
    if (errMsg.toLowerCase().includes('ipc') || errMsg.toLowerCase().includes('timeout') || e.response?.status === 504) {
      ipcDown.value = true
      simRunning.value = false
      messages.value.push({ role: 'agent', text: '(simulation IPC not available — start or resume a run first)' })
    } else {
      chatErr.value = errMsg
      messages.value.push({ role: 'agent', text: '(no response)' })
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
  } catch(e) {
    injectIsErr.value = true
    injectMsg.value   = e.response?.data?.error || e.message
  } finally { injecting.value = false }
}
</script>

<style scoped>
.page-header { margin-bottom: 24px; }
.breadcrumb {
  display: flex; align-items: center; gap: 6px;
  font-size: .8rem; color: var(--text3); margin-bottom: 8px;
}
.breadcrumb a { color: var(--text3); }
.breadcrumb a:hover { color: var(--text2); text-decoration: none; }
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
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px;
  min-height: 200px; max-height: 380px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 12px;
  scroll-behavior: smooth;
}
.chat-empty { color: var(--text3); font-size: .84rem; text-align: center; padding: 24px 0; }

.msg { display: flex; flex-direction: column; gap: 4px; }
.msg-label { font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.msg.user  .msg-label { color: var(--blue2); }
.msg.agent .msg-label { color: var(--green); }
.msg-body {
  font-size: .84rem; color: var(--text2); line-height: 1.65;
  background: var(--bg3); border-radius: 9px; padding: 9px 13px;
}
.msg.user .msg-body { background: rgba(99,102,241,.1); color: var(--text); }

.thinking { display: flex; gap: 5px; align-items: center; padding: 14px 16px !important; }
.dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--text3);
  animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,80%,100% { transform: scale(1); } 40% { transform: scale(1.5); } }

/* ── Quick questions ────────────────────────────────────────────── */
.quick-qs { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.qq-label { font-size: .72rem; color: var(--text3); flex-shrink: 0; }
.qq-btn {
  font-size: .72rem; padding: 4px 10px; border-radius: 6px;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text2); cursor: pointer; transition: all .15s;
}
.qq-btn:hover { border-color: var(--blue); color: var(--blue2); }

/* ── Chat bar ───────────────────────────────────────────────────── */
.chat-bar { display: flex; gap: 8px; }
.chat-bar input { flex: 1; }

/* ── Event types ────────────────────────────────────────────────── */
.event-types { display: flex; flex-direction: column; gap: 8px; }
.event-btn {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 14px; border-radius: 11px;
  background: var(--bg3); border: 1px solid var(--border);
  cursor: pointer; text-align: left;
  transition: border-color .2s, background .2s, transform .25s var(--ease-spring);
}
.event-btn:hover   { border-color: rgba(99,102,241,.35); transform: translateX(2px); }
.event-btn.selected { border-color: var(--blue); background: rgba(99,102,241,.08); }
.et-icon { font-size: 1.2rem; flex-shrink: 0; }
.et-name { font-size: .84rem; font-weight: 500; }
.et-desc { font-size: .74rem; color: var(--text3); }

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
