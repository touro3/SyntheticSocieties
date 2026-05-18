<template>
  <div class="nr">
    <div class="nr-head">
      <h2 class="section-title" style="margin:0">Network Replay</h2>
      <span class="nr-sub">
        real per-round trust / cooperation graph
        <template v-if="data">
          · {{ data.shown }}<template v-if="data.n_agents > data.shown"> of {{ data.n_agents }}</template>
          agents · {{ data.rounds.length }} rounds
        </template>
      </span>
    </div>

    <div v-if="loading" class="nr-state"><span class="spin">⟳</span> Loading…</div>
    <div v-else-if="err" class="nr-state err">{{ err }}</div>
    <div v-else-if="!data?.rounds?.length" class="nr-state">No recorded events for this run.</div>

    <template v-else>
      <svg class="nr-graph" :viewBox="`0 0 ${W} ${H}`">
        <defs>
          <!-- Dot grid background -->
          <pattern id="nr-dots" width="24" height="24" patternUnits="userSpaceOnUse">
            <circle cx="12" cy="12" r="0.75" fill="rgba(148,163,184,.09)"/>
          </pattern>
          <!-- Edge glow -->
          <filter id="nr-gf" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="2.2" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <!-- Node halo glow -->
          <filter id="nr-hf" x="-120%" y="-120%" width="340%" height="340%">
            <feGaussianBlur stdDeviation="6" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <!-- Selection pulse glow -->
          <filter id="nr-sf" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <!-- Arrowheads -->
          <marker id="ah-coop" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="rgba(20,184,166,.95)"/>
          </marker>
          <marker id="ah-steal" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="rgba(239,68,68,.95)"/>
          </marker>
        </defs>

        <!-- Canvas layers -->
        <rect :width="W" :height="H" fill="#070c18"/>
        <rect :width="W" :height="H" fill="url(#nr-dots)"/>

        <!-- ── Edges ────────────────────────────────── -->
        <g class="edges-layer">
          <line
            v-for="(e, i) in edgesClipped" :key="'e'+i"
            :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
            :class="`nr-edge nr-edge-${e.type}`"
            :marker-end="e.type === 'cooperate' ? 'url(#ah-coop)' : 'url(#ah-steal)'"
            filter="url(#nr-gf)"
          />
        </g>

        <!-- ── Halos (rendered behind nodes) ──────── -->
        <!-- v-for on <template> then v-if on child — avoids Vue3 v-if/v-for priority bug -->
        <template v-for="aid in data.agent_ids" :key="'h'+aid">
          <circle
            v-if="pos[aid] && states[aid]?.a && states[aid].a !== 'idle' && states[aid].a !== 'unknown'"
            :cx="pos[aid].x" :cy="pos[aid].y" r="22"
            :class="`nr-halo nr-halo-${states[aid].a}`"
            filter="url(#nr-hf)"
          />
        </template>

        <!-- ── Agent nodes ─────────────────────────── -->
        <template v-for="aid in data.agent_ids" :key="'n'+aid">
          <g
            v-if="pos[aid]"
            :transform="`translate(${pos[aid].x},${pos[aid].y})`"
            :class="['nr-node', states[aid]?.a || 'idle', { adv: advSet.has(aid), sel: selectedAgent === aid }]"
            style="cursor:pointer"
            @click="selectAgent(aid)"
          >
            <!-- Selection ring (outer orbit) -->
            <circle v-if="selectedAgent === aid" r="19" class="nr-sel-ring" filter="url(#nr-sf)"/>
            <!-- Adversarial hatch ring -->
            <circle v-if="advSet.has(aid)" r="16" class="nr-adv-ring"/>
            <!-- Main disc -->
            <circle r="13" class="nr-disc"/>
            <!-- Action glyph -->
            <text text-anchor="middle" dominant-baseline="central" class="nr-glyph">
              {{ glyph(states[aid]?.a) }}
            </text>
            <!-- Agent ID below -->
            <text text-anchor="middle" y="26" class="nr-label">{{ shortId(aid) }}</text>
          </g>
        </template>

        <!-- ── Thought pills (topmost layer) ─────── -->
        <template v-for="aid in data.agent_ids" :key="'p'+aid">
          <g
            v-if="pos[aid] && bubbleFor(aid)"
            :transform="`translate(${pos[aid].x + pillDx(aid)}, ${pos[aid].y + pillDy(aid)})`"
            class="nr-pill"
          >
            <rect :width="pillW(aid)" height="18" rx="9" class="nr-pill-bg"/>
            <text x="9" y="13" class="nr-pill-text">{{ truncatedBubble(bubbleFor(aid)) }}</text>
            <!-- Expand dot -->
            <g
              v-if="isLong(bubbleFor(aid))"
              :transform="`translate(${pillW(aid) - 20},0)`"
              class="nr-pill-more" style="cursor:pointer"
              @click.stop="selectedAgent = aid"
            >
              <rect width="20" height="18" rx="9" class="nr-pill-more-bg"/>
              <text x="10" y="13" text-anchor="middle" class="nr-pill-more-text">…</text>
            </g>
          </g>
        </template>
      </svg>

      <!-- Full reasoning panel -->
      <transition name="slide-up">
        <div v-if="selectedAgent" class="nr-panel">
          <div class="nr-panel-head">
            <span class="nr-panel-id">{{ selectedAgent }}</span>
            <span :class="['nr-panel-action', states[selectedAgent]?.a]">
              {{ states[selectedAgent]?.a || '—' }}
            </span>
            <span class="nr-panel-stats">
              ⬡ {{ states[selectedAgent]?.w ?? '—' }} &nbsp;·&nbsp;
              ⚡ {{ states[selectedAgent]?.s ?? '—' }}
            </span>
            <button class="nr-panel-close" @click="selectedAgent = null">✕</button>
          </div>
          <p class="nr-panel-text">
            {{ states[selectedAgent]?.r || states[selectedAgent]?.a || 'No reasoning recorded.' }}
          </p>
        </div>
      </transition>

      <!-- Transport controls -->
      <div class="nr-transport">
        <button class="btn btn-sm btn-outline" @click="togglePlay">
          {{ playing ? '❚❚ Pause' : '▶ Play' }}
        </button>
        <input
          class="nr-scrub" type="range" min="0" :max="data.rounds.length - 1"
          v-model.number="idx" @input="playing = false"
        />
        <span class="nr-round-tag">
          round {{ current.round }} / {{ data.rounds[data.rounds.length - 1].round }}
        </span>
        <select v-model.number="speed" class="nr-speed">
          <option :value="1200">0.5×</option>
          <option :value="700">1×</option>
          <option :value="350">2×</option>
          <option :value="150">4×</option>
        </select>
      </div>

      <div class="nr-legend">
        <span><i class="sw coop"></i> cooperate</span>
        <span><i class="sw steal"></i> steal</span>
        <span><i class="sw work"></i> work</span>
        <span><i class="sw save"></i> save</span>
        <span><i class="sw adv"></i> adversarial</span>
        <span class="nr-hint">click node to expand thought</span>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted, onBeforeUnmount, watch } from 'vue'
import { api } from '../api/index.js'

const props = defineProps({ expId: { type: String, required: true } })

// ── State ──────────────────────────────────────────────────────────────────
const loading       = ref(true)
const err           = ref('')
const data          = ref(null)
const idx           = ref(0)
const playing       = ref(false)
const speed         = ref(700)
const selectedAgent = ref(null)

// ── SVG constants ──────────────────────────────────────────────────────────
const W      = 860
const H      = 560
const NODE_R = 13
const MARGIN = 55
// Largest ring radius that keeps node centres inside the margin boundary
const MAX_R  = Math.floor(Math.min(W / 2, H / 2) - MARGIN - 4)  // ≈ 221

// ── Derived round data ─────────────────────────────────────────────────────
const advSet  = computed(() => new Set(data.value?.adversarial || []))
const current = computed(() => data.value?.rounds[idx.value] || { round: 0, states: {}, edges: [] })
const states  = computed(() => current.value.states || {})
const edges   = computed(() => current.value.edges || [])

// ── Layout: concentric rings, no overlaps ──────────────────────────────────
// Target ≥ 200 px of arc per agent so thought pills (max ~165 px) don't collide.
//   r = N × 200 / (2π), clamped to [130, MAX_R]
// Most-connected agents (server-sorted: highest degree first) go to inner rings.
const pos = reactive({})

function layout(ids) {
  const n = ids.length
  if (!n) return
  const cx = W / 2, cy = H / 2

  if (n === 1) { pos[ids[0]] = { x: cx, y: cy }; return }

  let rings
  if (n <= 8) {
    const r = Math.min(MAX_R, Math.max(130, Math.ceil(n * 200 / (2 * Math.PI))))
    rings = [{ r, cap: n }]
  } else if (n <= 18) {
    const rOuter = Math.min(MAX_R, Math.max(165, Math.ceil(n * 200 / (2 * Math.PI))))
    const rInner = Math.max(70, Math.round(rOuter * 0.44))
    const nInner = Math.ceil(n * 0.35)
    rings = [{ r: rInner, cap: nInner }, { r: rOuter, cap: n - nInner }]
  } else {
    rings = [
      { r: 48,  cap: 6  },
      { r: 98,  cap: 12 },
      { r: 152, cap: 18 },
      { r: 215, cap: 30 },
    ]
  }

  const remaining = [...ids]
  for (const ring of rings) {
    if (!remaining.length) break
    const count = Math.min(remaining.length, ring.cap)
    const batch = remaining.splice(0, count)
    batch.forEach((aid, i) => {
      const angle = (i / count) * 2 * Math.PI - Math.PI / 2
      pos[aid] = {
        x: Math.round(cx + Math.cos(angle) * ring.r),
        y: Math.round(cy + Math.sin(angle) * ring.r),
      }
    })
  }

  // Overflow: space remaining agents evenly around the outermost ring
  if (remaining.length) {
    const placed  = ids.length - remaining.length
    const total   = ids.length
    const lastR   = rings[rings.length - 1].r
    remaining.forEach((aid, j) => {
      const angle = ((placed + j) / total) * 2 * Math.PI - Math.PI / 2
      pos[aid] = {
        x: Math.max(MARGIN, Math.min(W - MARGIN, Math.round(cx + Math.cos(angle) * lastR))),
        y: Math.max(MARGIN, Math.min(H - MARGIN, Math.round(cy + Math.sin(angle) * lastR))),
      }
    })
  }
}

// ── Clipped edges (lines start/end at node boundary, not centre) ───────────
const edgesClipped = computed(() => {
  const GAP        = NODE_R + 2
  const ARROW_GAP  = NODE_R + 10
  const BIDIR_OFF  = 5

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
    const ox = bidir ? -ny * BIDIR_OFF : 0
    const oy = bidir ?  nx * BIDIR_OFF : 0
    return {
      x1: p1.x + nx * GAP        + ox,
      y1: p1.y + ny * GAP        + oy,
      x2: p2.x - nx * ARROW_GAP  + ox,
      y2: p2.y - ny * ARROW_GAP  + oy,
      type,
    }
  }).filter(Boolean)
})

// ── Node helpers ───────────────────────────────────────────────────────────
const GLYPHS = { cooperate: '◈', steal: '✗', work: '◆', save: '▣' }
function glyph(a) { return GLYPHS[a] || '·' }

function shortId(aid) {
  const m = aid.match(/\d+$/)
  return m ? 'A' + m[0] : aid.slice(0, 4)
}

// ── Thought pill helpers ───────────────────────────────────────────────────
const MAX_CHARS = 22

function bubbleFor(aid) {
  const s = states.value[aid]
  return s ? (s.r || s.a || null) : null
}

function isLong(text) { return !!text && text.length > MAX_CHARS }

function truncatedBubble(text) {
  return text && text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : (text || '')
}

function pillW(aid) {
  const text = bubbleFor(aid)
  if (!text) return 0
  const chars = Math.min(text.length, MAX_CHARS)
  return chars * 6.4 + 18 + (isLong(text) ? 22 : 0)
}

function pillDx(aid) {
  const p   = pos[aid]
  const bw  = pillW(aid)
  const bx  = NODE_R + 7
  // Flip left when near right edge
  if (p.x + bx + bw > W - MARGIN) return -(bw + NODE_R + 7)
  return bx
}

function pillDy(aid) {
  const p  = pos[aid]
  const by = -(18 + 12)  // default: above the node
  // Flip below when near top edge
  if (p.y + by < MARGIN) return NODE_R + 16
  return by
}

// ── Agent selection ────────────────────────────────────────────────────────
function selectAgent(aid) {
  selectedAgent.value = selectedAgent.value === aid ? null : aid
}

// ── Playback ───────────────────────────────────────────────────────────────
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
/* ── Container ─────────────────────────────────────────────────────────── */
.nr { padding: 20px; }
.nr-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.nr-sub  { font-size: 12px; color: var(--muted, #8b93a7); }
.nr-state { padding: 44px; text-align: center; color: var(--muted, #8b93a7); }
.nr-state.err { color: #ef4444; }
.spin { display: inline-block; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── SVG graph ─────────────────────────────────────────────────────────── */
.nr-graph {
  width: 100%; display: block;
  border-radius: 16px;
  border: 1px solid rgba(99,102,241,.16);
  box-shadow: 0 0 40px rgba(0,0,0,.6), inset 0 0 80px rgba(7,12,24,.8);
}

/* ── Edges ─────────────────────────────────────────────────────────────── */
.nr-edge { stroke-width: 1.5; fill: none; }
.nr-edge-cooperate {
  stroke: rgba(20,184,166,.75);
  filter: drop-shadow(0 0 3px rgba(20,184,166,.6));
}
.nr-edge-steal {
  stroke: rgba(239,68,68,.75);
  stroke-dasharray: 5 3;
  filter: drop-shadow(0 0 3px rgba(239,68,68,.6));
}

/* ── Halos (glow behind nodes) ─────────────────────────────────────────── */
.nr-halo         { opacity: .55; }
.nr-halo-cooperate { fill: rgba(20,184,166,.5); }
.nr-halo-steal     { fill: rgba(239,68,68,.55); }
.nr-halo-work      { fill: rgba(99,102,241,.45); }
.nr-halo-save      { fill: rgba(234,179,8,.4); }

/* ── Node discs ────────────────────────────────────────────────────────── */
.nr-disc {
  fill: rgba(30,41,59,.85);
  stroke: rgba(148,163,184,.35);
  stroke-width: 1.4;
  transition: fill .2s, stroke .2s;
}
.nr-node.cooperate .nr-disc { fill: rgba(13,101,95,.82); stroke: rgba(20,184,166,.8); }
.nr-node.steal     .nr-disc { fill: rgba(127,29,29,.82); stroke: rgba(239,68,68,.8); }
.nr-node.work      .nr-disc { fill: rgba(49,46,129,.82); stroke: rgba(99,102,241,.8); }
.nr-node.save      .nr-disc { fill: rgba(92,60,0,.82);   stroke: rgba(234,179,8,.8); }

/* Adversarial: dashed red outer ring */
.nr-adv-ring {
  fill: none;
  stroke: rgba(239,68,68,.7);
  stroke-width: 1.2;
  stroke-dasharray: 3.5 2;
}
/* Selected: bright white orbit ring */
.nr-sel-ring {
  fill: none;
  stroke: rgba(255,255,255,.82);
  stroke-width: 1.8;
}

.nr-glyph {
  font-size: 9px;
  fill: rgba(255,255,255,.85);
  font-family: 'ui-monospace', monospace;
  pointer-events: none;
}
.nr-label {
  font-size: 7px;
  fill: rgba(148,163,184,.6);
  font-family: 'ui-monospace', monospace;
  pointer-events: none;
}

/* ── Thought pills ─────────────────────────────────────────────────────── */
.nr-pill-bg {
  fill: rgba(7,12,24,.93);
  stroke: rgba(148,163,184,.22);
  stroke-width: 0.8;
}
.nr-pill-text {
  font: 6.5px/1 'ui-monospace', monospace;
  fill: rgba(226,232,240,.88);
  pointer-events: none;
}
.nr-pill-more-bg {
  fill: rgba(99,102,241,.28);
  stroke: rgba(99,102,241,.55);
  stroke-width: 0.8;
  cursor: pointer;
  transition: fill .15s;
}
.nr-pill-more:hover .nr-pill-more-bg { fill: rgba(99,102,241,.55); }
.nr-pill-more-text {
  font: bold 8px/1 'ui-monospace', monospace;
  fill: rgba(255,255,255,.9);
  pointer-events: none;
}
.nr-pill { animation: pill-pop .18s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes pill-pop { from { opacity: 0; transform: scale(.7); } to { opacity: 1; transform: scale(1); } }

/* ── Reasoning panel ───────────────────────────────────────────────────── */
.nr-panel {
  margin-top: 12px;
  padding: 13px 16px;
  background: rgba(7,12,24,.95);
  border: 1px solid rgba(99,102,241,.28);
  border-radius: 12px;
}
.nr-panel-head {
  display: flex; align-items: center; gap: 10px;
  flex-wrap: wrap; margin-bottom: 9px;
}
.nr-panel-id {
  font: 11px/1 'ui-monospace', monospace;
  color: rgba(148,163,184,.8);
}
.nr-panel-action {
  padding: 2px 9px; border-radius: 4px;
  font-size: 11px; font-weight: 600;
  background: rgba(99,102,241,.18); color: rgba(165,168,255,.95);
}
.nr-panel-action.cooperate { background: rgba(20,184,166,.18); color: rgba(20,184,166,.95); }
.nr-panel-action.steal     { background: rgba(239,68,68,.18);  color: rgba(239,68,68,.95); }
.nr-panel-action.work      { background: rgba(99,102,241,.18); color: rgba(165,168,255,.95); }
.nr-panel-action.save      { background: rgba(234,179,8,.14);  color: rgba(234,179,8,.95); }
.nr-panel-stats {
  font: 11px/1 'ui-monospace', monospace;
  color: rgba(148,163,184,.7);
}
.nr-panel-close {
  margin-left: auto;
  background: none; border: none;
  color: rgba(148,163,184,.6);
  cursor: pointer; font-size: 14px; padding: 0 2px;
  transition: color .15s;
}
.nr-panel-close:hover { color: rgba(255,255,255,.9); }
.nr-panel-text {
  margin: 0;
  font-size: 13px; line-height: 1.65;
  color: rgba(226,232,240,.9);
}

/* ── Transport ─────────────────────────────────────────────────────────── */
.nr-transport {
  display: flex; align-items: center; gap: 12px;
  margin-top: 14px; flex-wrap: wrap;
}
.nr-scrub { flex: 1; min-width: 160px; accent-color: #6366f1; }
.nr-round-tag {
  font: 12px/1 'ui-monospace', monospace;
  color: rgba(148,163,184,.7); min-width: 130px;
}
.nr-speed {
  background: transparent; color: inherit;
  border: 1px solid rgba(99,102,241,.3);
  border-radius: 6px; padding: 3px 6px; font-size: 12px;
}

/* ── Legend ────────────────────────────────────────────────────────────── */
.nr-legend {
  display: flex; gap: 16px; flex-wrap: wrap;
  margin-top: 12px; font-size: 11px;
  color: rgba(148,163,184,.7); align-items: center;
}
.nr-legend i.sw {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 4px; vertical-align: middle;
}
.sw.coop  { background: rgba(20,184,166,.65); }
.sw.steal { background: rgba(239,68,68,.65); }
.sw.work  { background: rgba(99,102,241,.55); }
.sw.save  { background: rgba(234,179,8,.55); }
.sw.adv   { border: 1.5px dashed rgba(239,68,68,.75); background: transparent; }
.nr-hint  { font-style: italic; margin-left: auto; }

/* ── Slide-up transition ───────────────────────────────────────────────── */
.slide-up-enter-active, .slide-up-leave-active { transition: opacity .18s ease, transform .18s ease; }
.slide-up-enter-from,   .slide-up-leave-to     { opacity: 0; transform: translateY(8px); }
</style>
