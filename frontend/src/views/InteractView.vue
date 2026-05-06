<template>
  <div class="container">
    <div class="page-header">
      <router-link :to="`/results/${expId}`" class="back">← Results</router-link>
      <h1 class="page-title">Agent Interaction</h1>
      <p class="page-sub mono" style="color:var(--text3)">{{ expId }}</p>
    </div>

    <div class="interact-layout">

      <!-- Interview panel -->
      <div class="panel card">
        <h2 class="section-title">Interview Agent</h2>
        <p class="panel-note">Talk directly to a simulated agent via the IPC bridge. The simulation must be running with <code>SimulationIPCServer</code> active.</p>

        <div class="field">
          <label>Agent ID</label>
          <input v-model="agentId" placeholder="e.g. agent_0, agent_42" />
        </div>

        <div class="chat-window">
          <div v-if="messages.length === 0" class="empty" style="padding:32px 0">
            No messages yet. Ask an agent something.
          </div>
          <template v-for="(m, i) in messages" :key="i">
            <div class="msg" :class="m.role">
              <div class="msg-label">{{ m.role === 'user' ? 'You' : `Agent ${agentId}` }}</div>
              <div class="msg-body">{{ m.text }}</div>
            </div>
          </template>
          <div v-if="thinking" class="msg agent">
            <div class="msg-label">Agent {{ agentId }}</div>
            <div class="msg-body"><span class="spin">⟳</span> Thinking…</div>
          </div>
        </div>

        <div class="chat-input-row">
          <input
            v-model="question"
            placeholder="Ask the agent something…"
            @keydown.enter="ask"
            :disabled="thinking"
          />
          <button class="btn btn-primary" @click="ask" :disabled="thinking || !agentId">
            <span v-if="thinking" class="spin">⟳</span>
            <span v-else>Send</span>
          </button>
        </div>

        <div v-if="interviewErr" class="error-msg">{{ interviewErr }}</div>

        <div class="quick-questions">
          <span class="qqlabel">Quick questions:</span>
          <button v-for="q in quickQs" :key="q" class="qq" @click="setQ(q)">{{ q }}</button>
        </div>
      </div>

      <!-- Inject panel -->
      <div class="panel card">
        <h2 class="section-title">Inject Event</h2>
        <p class="panel-note">Inject an exogenous event into the running simulation — no restart required.</p>

        <div class="field">
          <label>Event type</label>
          <div class="event-types">
            <button
              v-for="t in eventTypes" :key="t.value"
              class="event-type-btn"
              :class="{ active: injectType === t.value }"
              @click="selectEvent(t)"
            >
              <span class="et-icon">{{ t.icon }}</span>
              <span class="et-label">{{ t.label }}</span>
              <span class="et-desc">{{ t.desc }}</span>
            </button>
          </div>
        </div>

        <div class="field">
          <label>Payload (JSON)</label>
          <textarea v-model="injectPayload" rows="5" class="mono" style="font-size:.8rem;resize:vertical" />
        </div>

        <button class="btn btn-outline inject-btn" @click="doInject" :disabled="injecting">
          <span v-if="injecting" class="spin">⟳</span>
          {{ injecting ? 'Injecting…' : 'Inject Event' }}
        </button>

        <div v-if="injectMsg" class="inject-result" :class="injectErr ? 'error-msg' : 'success-msg'">
          <pre>{{ injectMsg }}</pre>
        </div>

        <div class="inject-docs">
          <h4>Payload schemas</h4>
          <div class="doc-item" v-for="t in eventTypes" :key="t.value">
            <strong>{{ t.value }}</strong>
            <code class="mono doc-code">{{ t.schema }}</code>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api/index.js'

const route  = useRoute()
const expId  = route.params.expId

const agentId    = ref('agent_0')
const question   = ref('')
const messages   = ref([])
const thinking   = ref(false)
const interviewErr = ref('')

const injectType    = ref('wealth_shock')
const injectPayload = ref('{"factor": 0.5, "agent_ids": "all"}')
const injecting     = ref(false)
const injectMsg     = ref('')
const injectErr     = ref(false)

const quickQs = [
  'What was your last decision and why?',
  'How is your wealth compared to others?',
  'Do you trust your neighbors?',
  'What would change your strategy?',
]

const eventTypes = [
  {
    value: 'wealth_shock',
    icon: '💸',
    label: 'Wealth Shock',
    desc: 'Multiplies all agent wealth by a factor',
    schema: '{"factor": 0.5, "agent_ids": "all"}',
  },
  {
    value: 'signal_update',
    icon: '📡',
    label: 'Signal Update',
    desc: 'Updates the public economic signal',
    schema: '{"economy": "recession", "severity": 0.8}',
  },
  {
    value: 'narrative',
    icon: '📢',
    label: 'Narrative',
    desc: 'Injects a narrative event agents observe',
    schema: '{"text": "A market crash occurred.", "rounds": 3}',
  },
]

function selectEvent(t) {
  injectType.value = t.value
  injectPayload.value = t.schema
}

function setQ(q) { question.value = q }

async function ask() {
  if (!question.value.trim() || !agentId.value) return
  const q = question.value.trim()
  messages.value.push({ role: 'user', text: q })
  question.value = ''
  thinking.value = true
  interviewErr.value = ''
  try {
    const r = await api.interview(expId, agentId.value, q)
    const reply = r.data?.response || r.data?.answer || JSON.stringify(r.data)
    messages.value.push({ role: 'agent', text: reply })
  } catch (e) {
    interviewErr.value = e.response?.data?.error || 'Interview failed — is the simulation running?'
    messages.value.push({ role: 'agent', text: '(no response)' })
  } finally {
    thinking.value = false
  }
}

async function doInject() {
  injecting.value = true; injectMsg.value = ''; injectErr.value = false
  try {
    const payload = JSON.parse(injectPayload.value)
    const r = await api.inject(expId, injectType.value, payload)
    injectMsg.value = JSON.stringify(r.data, null, 2)
  } catch (e) {
    injectErr.value = true
    injectMsg.value = e.response?.data?.error || e.message
  } finally { injecting.value = false }
}
</script>

<style scoped>
.page-header { margin-bottom: 28px; }
.back { color: var(--text3); font-size: .84rem; display: inline-block; margin-bottom: 8px; }
.back:hover { color: var(--text); text-decoration: none; }
.page-title { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
.page-sub   { font-size: .88rem; }

.interact-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
@media (max-width: 800px) { .interact-layout { grid-template-columns: 1fr; } }

.panel { display: flex; flex-direction: column; gap: 16px; }
.panel-note { font-size: .82rem; color: var(--text2); line-height: 1.6; }
.panel-note code { color: var(--teal); background: rgba(20,184,166,.1); padding: 1px 5px; border-radius: 4px; }

.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: .78rem; color: var(--text3); font-weight: 500; }

.chat-window {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px; min-height: 220px;
  max-height: 380px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 12px;
}
.msg { display: flex; flex-direction: column; gap: 4px; }
.msg-label { font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.msg.user .msg-label  { color: var(--blue); }
.msg.agent .msg-label { color: var(--green); }
.msg-body {
  font-size: .85rem; color: var(--text2); line-height: 1.6;
  background: var(--bg3); border-radius: 8px; padding: 10px 13px;
}
.msg.user .msg-body  { background: rgba(99,102,241,.1); color: var(--text); }

.chat-input-row { display: flex; gap: 8px; }
.chat-input-row input { flex: 1; }

.error-msg {
  background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2);
  border-radius: 8px; padding: 10px 14px; font-size: .83rem; color: var(--rose);
}

.quick-questions { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.qqlabel { font-size: .74rem; color: var(--text3); }
.qq {
  font-size: .74rem; padding: 4px 10px; border-radius: 6px;
  background: var(--bg2); border: 1px solid var(--border);
  color: var(--text2); cursor: pointer; transition: all .15s;
}
.qq:hover { border-color: var(--blue); color: var(--blue); }

.event-types { display: flex; flex-direction: column; gap: 8px; }
.event-type-btn {
  display: grid; grid-template-columns: 28px 1fr;
  grid-template-rows: auto auto;
  align-items: center; gap: 0 10px;
  padding: 12px 14px; border-radius: 10px;
  background: var(--bg2); border: 1px solid var(--border);
  cursor: pointer; transition: all .15s; text-align: left;
}
.event-type-btn.active { border-color: var(--blue); background: rgba(99,102,241,.08); }
.event-type-btn:hover  { border-color: rgba(99,102,241,.4); }
.et-icon  { grid-row: 1/3; font-size: 1.2rem; }
.et-label { font-size: .84rem; font-weight: 500; color: var(--text); }
.et-desc  { font-size: .76rem; color: var(--text3); }

.inject-btn { width: 100%; justify-content: center; }

.inject-result { border-radius: 8px; padding: 10px; }
.inject-result pre { font-size: .76rem; white-space: pre-wrap; word-break: break-all; max-height: 100px; overflow-y: auto; }
.success-msg { background: rgba(20,184,166,.08); border: 1px solid rgba(20,184,166,.2); color: var(--teal); }

.inject-docs { display: flex; flex-direction: column; gap: 10px; padding-top: 4px; border-top: 1px solid var(--border); }
.inject-docs h4 { font-size: .8rem; color: var(--text3); font-weight: 600; }
.doc-item { display: flex; flex-direction: column; gap: 3px; }
.doc-item strong { font-size: .8rem; color: var(--text2); }
.doc-code {
  font-size: .74rem; color: var(--teal);
  background: rgba(20,184,166,.07); padding: 4px 8px; border-radius: 6px;
  white-space: pre-wrap;
}
</style>
