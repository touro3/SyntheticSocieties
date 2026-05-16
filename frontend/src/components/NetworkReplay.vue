<template>
  <div class="replay card">
    <div class="replay-head">
      <h2 class="section-title" style="margin:0">Network Replay</h2>
      <span class="replay-sub">
        real per-round trust / cooperation graph
        <template v-if="data">
          · {{ data.shown }}<template v-if="data.n_agents > data.shown"> of {{ data.n_agents }}</template>
          agents · {{ data.rounds.length }} rounds
        </template>
      </span>
    </div>

    <div v-if="loading" class="replay-state"><span class="spin">⟳</span> Loading events…</div>
    <div v-else-if="err" class="replay-state err">{{ err }}</div>
    <div v-else-if="!data || !data.rounds.length" class="replay-state">
      No recorded events for this run.
    </div>

    <template v-else>
      <svg class="replay-svg" viewBox="0 0 600 340" xmlns="http://www.w3.org/2000/svg">
        <!-- Real interaction edges for the current round -->
        <g class="lines-layer">
          <line
            v-for="(e, i) in edges" :key="'e'+i"
            :x1="pos[e[0]].x" :y1="pos[e[0]].y"
            :x2="pos[e[1]].x" :y2="pos[e[1]].y"
            class="edge-line" :class="e[2]"
          />
        </g>

        <!-- Agent nodes -->
        <g v-for="aid in data.agent_ids" :key="aid"
           :transform="`translate(${pos[aid].x},${pos[aid].y})`"
           class="agent-node" :class="[states[aid]?.a || 'idle', { adversarial: advSet.has(aid) }]">
          <circle r="13" class="node-body" />
          <circle r="9"  class="node-inner" />
          <text class="node-glyph" text-anchor="middle" dominant-baseline="central">
            {{ glyph(states[aid]?.a) }}
          </text>
          <g v-if="bubbleFor(aid)" class="bubble-group">
            <rect x="15" y="-28" :width="bubbleW(bubbleFor(aid))" height="18" rx="6" class="bubble-rect" />
            <text :x="15 + bubbleW(bubbleFor(aid))/2" y="-15" text-anchor="middle" class="bubble-text">
              {{ bubbleFor(aid) }}
            </text>
          </g>
        </g>
      </svg>

      <!-- Transport controls -->
      <div class="transport">
        <button class="btn btn-sm btn-outline" @click="togglePlay">
          {{ playing ? '❚❚ Pause' : '▶ Play' }}
        </button>
        <input
          class="scrub" type="range" min="0" :max="data.rounds.length - 1"
          v-model.number="idx" @input="playing = false"
        />
        <span class="round-tag">round {{ current.round }} / {{ data.rounds[data.rounds.length - 1].round }}</span>
        <select v-model.number="speed" class="speed">
          <option :value="1200">0.5×</option>
          <option :value="700">1×</option>
          <option :value="350">2×</option>
          <option :value="150">4×</option>
        </select>
      </div>

      <div class="legend">
        <span><i class="sw cooperate"></i> cooperate</span>
        <span><i class="sw steal"></i> steal</span>
        <span><i class="sw work"></i> work</span>
        <span><i class="sw save"></i> save</span>
        <span><i class="sw adv"></i> adversarial</span>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted, onBeforeUnmount, watch } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({ expId: { type: String, required: true } })

const loading = ref(true)
const err     = ref('')
const data    = ref(null)
const idx     = ref(0)
const playing = ref(false)
const speed   = ref(700)

const advSet  = computed(() => new Set(data.value?.adversarial || []))
const current = computed(() => data.value?.rounds[idx.value] || { round: 0, states: {}, edges: [] })
const states  = computed(() => current.value.states || {})
const edges   = computed(() => current.value.edges || [])

// Deterministic fixed layout — positions stay put across rounds so the eye
// tracks behaviour change, not motion. Same elliptical scatter as the flock.
const pos = reactive({})
const W = 600, H = 340, MARGIN = 46
function layout(ids) {
  const n = ids.length
  ids.forEach((aid, i) => {
    const angle = (i / n) * Math.PI * 2 + (i % 3) * 0.4
    const r = 60 + (i % 4) * 26 + (i % 2) * 16
    pos[aid] = {
      x: Math.max(MARGIN, Math.min(W - MARGIN, W / 2 + Math.cos(angle) * r * 1.45)),
      y: Math.max(MARGIN, Math.min(H - MARGIN, H / 2 + Math.sin(angle) * r * 0.82)),
    }
  })
}

const GLYPHS = { cooperate: '◈', steal: '✗', work: '◆', save: '▣' }
function glyph(a) { return GLYPHS[a] || '·' }
function bubbleFor(aid) {
  const s = states.value[aid]
  if (!s) return null
  return s.r || s.a
}
function bubbleW(t) { return Math.min(220, t.length * 6.4 + 16) }

let timer = null
function tick() {
  if (!playing.value || !data.value) return
  if (idx.value >= data.value.rounds.length - 1) { playing.value = false; return }
  idx.value += 1
  timer = setTimeout(tick, speed.value)
}
function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    if (idx.value >= data.value.rounds.length - 1) idx.value = 0
    timer = setTimeout(tick, speed.value)
  } else {
    clearTimeout(timer)
  }
}
watch(playing, (p) => { if (!p) clearTimeout(timer) })

onMounted(async () => {
  try {
    const r = await api.replay(props.expId)
    data.value = r.data
    layout(data.value.agent_ids || [])
  } catch (e) {
    err.value = e.response?.data?.error || 'Replay unavailable for this run.'
  } finally {
    loading.value = false
  }
})
onBeforeUnmount(() => clearTimeout(timer))
</script>

<style scoped>
.replay { padding: 20px; }
.replay-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.replay-sub { font-size: 12px; color: var(--muted, #8b93a7); }
.replay-state { padding: 40px; text-align: center; color: var(--muted, #8b93a7); }
.replay-state.err { color: var(--rose, #f43f5e); }
.spin { display: inline-block; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.replay-svg {
  width: 100%; display: block; border-radius: 14px;
  background: linear-gradient(135deg, rgba(5,10,21,.95), rgba(13,21,40,.9) 50%, rgba(22,26,60,.85));
  border: 1px solid rgba(99,102,241,.18);
}

.edge-line { stroke-width: 1.4; }
.edge-line.cooperate { stroke: rgba(20,184,166,.55); filter: drop-shadow(0 0 2px rgba(20,184,166,.5)); }
.edge-line.steal     { stroke: rgba(244,63,94,.6);  filter: drop-shadow(0 0 2px rgba(244,63,94,.55)); }

.node-body  { fill: rgba(99,102,241,.18); stroke: rgba(99,102,241,.5); stroke-width: 1.4; }
.node-inner { fill: rgba(99,102,241,.35); }
.agent-node.cooperate .node-body  { fill: rgba(20,184,166,.18); stroke: rgba(20,184,166,.6); }
.agent-node.cooperate .node-inner { fill: rgba(20,184,166,.4); }
.agent-node.steal .node-body  { fill: rgba(244,63,94,.18); stroke: rgba(244,63,94,.6); }
.agent-node.steal .node-inner { fill: rgba(244,63,94,.4); }
.agent-node.save .node-body  { fill: rgba(245,158,11,.16); stroke: rgba(245,158,11,.55); }
.agent-node.save .node-inner { fill: rgba(245,158,11,.36); }
.agent-node.adversarial .node-body { stroke-dasharray: 3 2; stroke: rgba(244,63,94,.75); }

.node-glyph { font-size: 8px; fill: rgba(255,255,255,.75); font-family: monospace; pointer-events: none; }

.bubble-rect { fill: rgba(22,26,60,.96); stroke: rgba(99,102,241,.4); stroke-width: .8; }
.bubble-text { font-size: 7.5px; fill: rgba(255,255,255,.82); font-family: monospace; pointer-events: none; }
.bubble-group { animation: pop .2s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes pop { from { opacity: 0; transform: scale(.6); } to { opacity: 1; transform: scale(1); } }

.transport { display: flex; align-items: center; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
.scrub { flex: 1; min-width: 160px; accent-color: var(--blue2, #6366f1); }
.round-tag { font-family: monospace; font-size: 12px; color: var(--muted, #8b93a7); min-width: 120px; }
.speed { background: transparent; color: inherit; border: 1px solid rgba(99,102,241,.3); border-radius: 6px; padding: 3px 6px; font-size: 12px; }

.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; font-size: 11px; color: var(--muted, #8b93a7); }
.legend i.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.sw.cooperate { background: rgba(20,184,166,.6); }
.sw.steal { background: rgba(244,63,94,.6); }
.sw.work { background: rgba(99,102,241,.5); }
.sw.save { background: rgba(245,158,11,.55); }
.sw.adv { border: 1.5px dashed rgba(244,63,94,.8); }
</style>
