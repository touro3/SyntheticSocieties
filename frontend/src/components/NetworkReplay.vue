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
      <svg class="replay-svg" :viewBox="`0 0 ${W} ${H}`" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arr-cooperate" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="rgba(20,184,166,.85)" />
          </marker>
          <marker id="arr-steal" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="rgba(244,63,94,.9)" />
          </marker>
        </defs>

        <!-- Interaction edges -->
        <g class="lines-layer">
          <line
            v-for="(e, i) in edgesClipped" :key="'e' + i"
            :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
            class="edge-line" :class="e.type"
            :marker-end="`url(#arr-${e.type})`"
          />
        </g>

        <!-- Agent nodes -->
        <g
          v-for="aid in data.agent_ids" :key="aid"
          v-if="pos[aid]"
          :transform="`translate(${pos[aid].x},${pos[aid].y})`"
          class="agent-node"
          :class="[states[aid]?.a || 'idle', { adversarial: advSet.has(aid), selected: selectedAgent === aid }]"
          style="cursor:pointer"
          @click="selectAgent(aid)"
        >
          <circle r="15" class="node-body" />
          <circle r="10" class="node-inner" />
          <text class="node-glyph" text-anchor="middle" dominant-baseline="central">{{ glyph(states[aid]?.a) }}</text>
          <text class="node-label" text-anchor="middle" y="27">{{ shortId(aid) }}</text>

          <!-- Speech bubble (bounded + truncated) -->
          <g v-if="bubbleFor(aid)" class="bubble-group" :transform="bubbleOffset(aid)">
            <rect :width="bubbleRectW(aid)" height="20" rx="5" class="bubble-rect" />
            <text x="6" y="14" class="bubble-text" :textLength="bubbleTextW(aid)" lengthAdjust="spacing">
              {{ truncatedBubble(bubbleFor(aid)) }}
            </text>
            <!-- Expand button shown when text is truncated -->
            <g
              v-if="isLong(bubbleFor(aid))"
              class="expand-btn"
              :transform="`translate(${bubbleRectW(aid) - 22}, 0)`"
              @click.stop="selectedAgent = aid"
            >
              <rect width="20" height="20" rx="5" class="expand-rect" />
              <text x="10" y="14" text-anchor="middle" class="expand-text">…</text>
            </g>
          </g>
        </g>
      </svg>

      <!-- Full thought expansion panel -->
      <transition name="slide-down">
        <div v-if="selectedAgent" class="thought-panel">
          <div class="thought-head">
            <span class="thought-id">{{ selectedAgent }}</span>
            <span class="thought-action" :class="states[selectedAgent]?.a || ''">
              {{ states[selectedAgent]?.a || '—' }}
            </span>
            <span class="thought-stats">
              wealth {{ states[selectedAgent]?.w ?? '—' }} &nbsp;·&nbsp; stress {{ states[selectedAgent]?.s ?? '—' }}
            </span>
            <button class="close-btn" @click="selectedAgent = null">✕</button>
          </div>
          <p class="thought-body">
            {{ states[selectedAgent]?.r || states[selectedAgent]?.a || 'No reasoning recorded.' }}
          </p>
        </div>
      </transition>

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
        <span class="legend-hint">click agent to expand thought</span>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted, onBeforeUnmount, watch } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({ expId: { type: String, required: true } })

const loading = ref(true)
const err = ref('')
const data = ref(null)
const idx = ref(0)
const playing = ref(false)
const speed = ref(700)
const selectedAgent = ref(null)

// SVG canvas dimensions (viewBox units, scales with container)
const W = 800
const H = 580
const NODE_R = 15
const MARGIN = 52
const MAX_CHARS = 22
// Maximum radius that keeps all node centres within MARGIN of the SVG edges
const MAX_R = Math.floor(Math.min(W / 2, H / 2) - MARGIN - 6)  // ≈ 232

const advSet  = computed(() => new Set(data.value?.adversarial || []))
const current = computed(() => data.value?.rounds[idx.value] || { round: 0, states: {}, edges: [] })
const states  = computed(() => current.value.states || {})
const edges   = computed(() => current.value.edges || [])

// Deterministic concentric-ring layout — no overlaps.
// Most-connected agents (sorted first by server) are placed in inner rings.
const pos = reactive({})
function layout(ids) {
  const n = ids.length
  if (n === 0) return
  const cx = W / 2, cy = H / 2

  if (n === 1) { pos[ids[0]] = { x: cx, y: cy }; return }

  let rings
  if (n <= 8) {
    // Target ≥200px arc per agent so bubbles (max ~172px wide) don't collide.
    // r = N * 200 / (2π), then clamp to [130, MAX_R].
    const r = Math.min(MAX_R, Math.max(130, Math.ceil(n * 200 / (2 * Math.PI))))
    rings = [{ r, cap: n }]
  } else if (n <= 18) {
    const n1 = Math.ceil(n * 0.35)
    const rOuter = Math.min(MAX_R, Math.max(170, Math.ceil(n * 200 / (2 * Math.PI))))
    const rInner = Math.max(70, Math.floor(rOuter * 0.45))
    rings = [
      { r: rInner, cap: n1 },
      { r: rOuter, cap: n - n1 },
    ]
  } else {
    rings = [
      { r: 45,  cap: 6  },
      { r: 95,  cap: 12 },
      { r: 150, cap: 18 },
      { r: 215, cap: 30 },
    ]
  }

  const remaining = [...ids]
  for (const ring of rings) {
    if (!remaining.length) break
    const count = Math.min(remaining.length, ring.cap)
    const batch = remaining.splice(0, count)
    batch.forEach((aid, i) => {
      const angle = (i / count) * Math.PI * 2 - Math.PI / 2
      pos[aid] = {
        x: Math.round(cx + Math.cos(angle) * ring.r),
        y: Math.round(cy + Math.sin(angle) * ring.r),
      }
    })
  }

  // Any overflow: distribute evenly around the outermost ring
  if (remaining.length) {
    const lastR = 215
    const already = ids.length - remaining.length
    remaining.forEach((aid, j) => {
      const total = already + remaining.length
      const angle = ((already + j) / total) * Math.PI * 2 - Math.PI / 2
      pos[aid] = {
        x: Math.max(MARGIN, Math.min(W - MARGIN, Math.round(cx + Math.cos(angle) * lastR))),
        y: Math.max(MARGIN, Math.min(H - MARGIN, Math.round(cy + Math.sin(angle) * lastR))),
      }
    })
  }
}

// Clip edge line endpoints to the node circle boundary + arrowhead gap
const edgesClipped = computed(() => {
  const GAP = NODE_R + 2
  const ARROW_GAP = NODE_R + 10
  const BIDIR_OFFSET = 4 // perpendicular shift when both directions exist

  const edgeSet = new Set(edges.value.map(e => `${e[0]}\0${e[1]}`))

  return edges.value.map(e => {
    const [src, tgt, type] = e
    const p1 = pos[src], p2 = pos[tgt]
    if (!p1 || !p2) return null
    const dx = p2.x - p1.x, dy = p2.y - p1.y
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < 1) return null
    const nx = dx / dist, ny = dy / dist
    const bidir = edgeSet.has(`${tgt}\0${src}`)
    const ox = bidir ? -ny * BIDIR_OFFSET : 0
    const oy = bidir ?  nx * BIDIR_OFFSET : 0
    return {
      x1: p1.x + nx * GAP       + ox,
      y1: p1.y + ny * GAP       + oy,
      x2: p2.x - nx * ARROW_GAP + ox,
      y2: p2.y - ny * ARROW_GAP + oy,
      type,
    }
  }).filter(Boolean)
})

const GLYPHS = { cooperate: '◈', steal: '✗', work: '◆', save: '▣' }
function glyph(a) { return GLYPHS[a] || '·' }

function shortId(aid) {
  const m = aid.match(/\d+$/)
  return m ? 'A' + m[0] : aid.slice(0, 4)
}

function bubbleFor(aid) {
  const s = states.value[aid]
  if (!s) return null
  return (s.r || s.a) || null
}

function isLong(text) { return text && text.length > MAX_CHARS }

function truncatedBubble(text) {
  if (!text || text.length <= MAX_CHARS) return text || ''
  return text.slice(0, MAX_CHARS)
}

// Width of the bubble background rect
function bubbleRectW(aid) {
  const text = bubbleFor(aid)
  if (!text) return 0
  const visibleChars = Math.min(text.length, MAX_CHARS)
  const textPx = visibleChars * 6.2 + 12
  return isLong(text) ? textPx + 24 : textPx
}

// textLength for the SVG text element (keeps chars from overflowing)
function bubbleTextW(aid) {
  const text = bubbleFor(aid)
  if (!text) return 0
  const visibleChars = Math.min(text.length, MAX_CHARS)
  return visibleChars * 6.2
}

// Position the bubble so it never leaves the SVG canvas
function bubbleOffset(aid) {
  const p = pos[aid]
  const bw = bubbleRectW(aid)
  const BH = 20

  let bx = NODE_R + 4
  let by = -(BH + 10)

  // Near right edge — flip left
  if (p.x + bx + bw > W - MARGIN) bx = -(bw + NODE_R + 4)
  // Near left edge after flip — clamp at margin
  if (p.x + bx < MARGIN) bx = -p.x + MARGIN
  // Near top edge — flip below node
  if (p.y + by < MARGIN) by = NODE_R + 12
  // Near bottom edge — push up
  if (p.y + by + BH > H - MARGIN) by = -(BH + 10)

  return `translate(${Math.round(bx)},${Math.round(by)})`
}

function selectAgent(aid) {
  selectedAgent.value = selectedAgent.value === aid ? null : aid
}

// Playback timer
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
watch(playing, p => { if (!p) clearTimeout(timer) })

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

/* SVG canvas */
.replay-svg {
  width: 100%; display: block; border-radius: 14px;
  background: linear-gradient(135deg, rgba(5,10,21,.95), rgba(13,21,40,.9) 50%, rgba(22,26,60,.85));
  border: 1px solid rgba(99,102,241,.18);
}

/* Edges */
.edge-line { stroke-width: 1.8; fill: none; }
.edge-line.cooperate {
  stroke: rgba(20,184,166,.7);
  filter: drop-shadow(0 0 2px rgba(20,184,166,.45));
}
.edge-line.steal {
  stroke: rgba(244,63,94,.75);
  stroke-dasharray: 5 3;
  filter: drop-shadow(0 0 2px rgba(244,63,94,.5));
}

/* Node base */
.node-body  { fill: rgba(99,102,241,.18); stroke: rgba(99,102,241,.5);  stroke-width: 1.4; transition: stroke .2s, fill .2s; }
.node-inner { fill: rgba(99,102,241,.35); transition: fill .2s; }

/* Action-coloured nodes */
.agent-node.cooperate .node-body  { fill: rgba(20,184,166,.18); stroke: rgba(20,184,166,.65); }
.agent-node.cooperate .node-inner { fill: rgba(20,184,166,.42); }
.agent-node.steal     .node-body  { fill: rgba(244,63,94,.18);  stroke: rgba(244,63,94,.65); }
.agent-node.steal     .node-inner { fill: rgba(244,63,94,.42); }
.agent-node.work      .node-body  { fill: rgba(99,102,241,.22); stroke: rgba(99,102,241,.62); }
.agent-node.save      .node-body  { fill: rgba(245,158,11,.16); stroke: rgba(245,158,11,.58); }
.agent-node.save      .node-inner { fill: rgba(245,158,11,.38); }

/* Adversarial + selected highlights */
.agent-node.adversarial .node-body { stroke-dasharray: 3 2; stroke: rgba(244,63,94,.8); }
.agent-node.selected    .node-body { stroke: rgba(255,255,255,.85); stroke-width: 2.4; }

.node-glyph { font-size: 9px;  fill: rgba(255,255,255,.82); font-family: monospace; pointer-events: none; }
.node-label { font-size: 7px;  fill: rgba(255,255,255,.42); font-family: monospace; pointer-events: none; }

/* Thought bubbles */
.bubble-rect { fill: rgba(12,18,42,.97); stroke: rgba(99,102,241,.45); stroke-width: .8; }
.bubble-text { font-size: 7px; fill: rgba(255,255,255,.88); font-family: monospace; pointer-events: none; }
.expand-rect {
  fill: rgba(99,102,241,.3); stroke: rgba(99,102,241,.6); stroke-width: .8;
  cursor: pointer;
  transition: fill .15s;
}
.expand-btn:hover .expand-rect { fill: rgba(99,102,241,.55); }
.expand-text { font-size: 9px; fill: rgba(255,255,255,.92); font-family: monospace; cursor: pointer; pointer-events: all; }
.bubble-group { animation: pop .2s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes pop { from { opacity: 0; transform: scale(.6); } to { opacity: 1; transform: scale(1); } }

/* Expanded thought panel */
.thought-panel {
  margin-top: 12px;
  padding: 12px 16px;
  background: rgba(12,18,42,.92);
  border: 1px solid rgba(99,102,241,.32);
  border-radius: 10px;
}
.thought-head {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px;
}
.thought-id { font-family: monospace; font-size: 12px; color: var(--muted, #8b93a7); }
.thought-action {
  padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
  background: rgba(99,102,241,.2); color: rgba(150,153,255,.9);
}
.thought-action.cooperate { background: rgba(20,184,166,.2);  color: rgba(20,184,166,.95); }
.thought-action.steal     { background: rgba(244,63,94,.2);   color: rgba(244,63,94,.95); }
.thought-action.work      { background: rgba(99,102,241,.2);  color: rgba(150,153,255,.95); }
.thought-action.save      { background: rgba(245,158,11,.15); color: rgba(245,158,11,.95); }
.thought-stats { font-size: 11px; color: var(--muted, #8b93a7); font-family: monospace; }
.close-btn {
  margin-left: auto; background: none; border: none;
  color: var(--muted, #8b93a7); cursor: pointer; font-size: 14px; padding: 0 4px;
}
.close-btn:hover { color: rgba(255,255,255,.9); }
.thought-body { margin: 0; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,.88); }

/* Transport */
.transport { display: flex; align-items: center; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
.scrub { flex: 1; min-width: 160px; accent-color: var(--blue2, #6366f1); }
.round-tag { font-family: monospace; font-size: 12px; color: var(--muted, #8b93a7); min-width: 120px; }
.speed {
  background: transparent; color: inherit;
  border: 1px solid rgba(99,102,241,.3); border-radius: 6px;
  padding: 3px 6px; font-size: 12px;
}

/* Legend */
.legend {
  display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px;
  font-size: 11px; color: var(--muted, #8b93a7); align-items: center;
}
.legend i.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.sw.cooperate { background: rgba(20,184,166,.6); }
.sw.steal     { background: rgba(244,63,94,.6); }
.sw.work      { background: rgba(99,102,241,.5); }
.sw.save      { background: rgba(245,158,11,.55); }
.sw.adv       { border: 1.5px dashed rgba(244,63,94,.8); background: transparent; }
.legend-hint  { font-style: italic; margin-left: auto; }

/* Slide-down transition for thought panel */
.slide-down-enter-active,
.slide-down-leave-active { transition: opacity .18s ease, transform .18s ease; }
.slide-down-enter-from,
.slide-down-leave-to    { opacity: 0; transform: translateY(-6px); }
</style>
