<template>
  <div class="flock-wrap" :class="{ paused: !active }">
    <svg class="flock-svg" viewBox="0 0 600 340" xmlns="http://www.w3.org/2000/svg">
      <!-- Connection lines -->
      <g class="lines-layer">
        <line v-for="(edge, i) in edges" :key="'e'+i"
          :x1="nodes[edge[0]].x" :y1="nodes[edge[0]].y"
          :x2="nodes[edge[1]].x" :y2="nodes[edge[1]].y"
          class="edge-line" :class="{ active: activeEdge === i }" />
      </g>

      <!-- Agent nodes -->
      <g v-for="(n, i) in nodes" :key="'n'+i"
        :transform="`translate(${n.x},${n.y})`"
        class="agent-node" :class="n.type">

        <!-- Pulse ring when active -->
        <circle v-if="n.pulse" r="22" class="pulse-ring" />

        <!-- Body -->
        <circle r="14" class="node-body" />
        <circle r="10" class="node-inner" />

        <!-- Icon glyph -->
        <text class="node-glyph" text-anchor="middle" dominant-baseline="central">{{ n.glyph }}</text>

        <!-- Speech bubble -->
        <g v-if="n.bubble" class="bubble-group">
          <rect x="16" y="-30" :width="bubbleWidth(n.bubble)" height="20" rx="7" class="bubble-rect" />
          <polygon :points="bubblePointer(n)" class="bubble-rect" />
          <text :x="16 + bubbleWidth(n.bubble)/2" y="-16" text-anchor="middle" class="bubble-text">{{ n.bubble }}</text>
        </g>
      </g>

      <!-- LLM label -->
      <g v-if="showLlmLabel">
        <rect x="10" y="300" width="130" height="26" rx="8" class="llm-badge-rect" />
        <text x="75" y="317" text-anchor="middle" class="llm-badge-text">LLM Inference Running</text>
      </g>

      <!-- Agents count — show real prop value, not visual-cap -->
      <text x="590" y="320" text-anchor="end" class="count-text">
        {{ props.nAgents }} agent{{ props.nAgents !== 1 ? 's' : '' }}
        <tspan v-if="props.nAgents > visualCount" style="opacity:.5"> ({{ visualCount }} shown)</tspan>
      </text>
    </svg>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  active:       { type: Boolean, default: true },
  nAgents:      { type: Number,  default: 10 },
  badAppleFrac: { type: Number,  default: 0.05 },
  showLlmLabel: { type: Boolean, default: false },
})

// ── Layout constants ────────────────────────────────────────────────
const W = 600, H = 340, MARGIN = 50
const BUBBLES = ['work', 'save', 'cooperate', '...', 'decide', 'work', 'help', 'trust?', 'save']

// ── Visual node count: cap at 16 for readability, use real count when small ──
const visualCount = computed(() => {
  const n = props.nAgents
  if (n <= 16) return n
  if (n <= 40) return 12
  return 14
})

// ── Build fixed node positions (quasi-random deterministic) ─────────
function buildNodes(n) {
  const nodes = []
  const badCount = Math.max(0, Math.round(n * props.badAppleFrac))
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * Math.PI * 2 + (i % 3) * 0.4
    const r = 60 + (i % 4) * 28 + (i % 2) * 18
    const cx = W / 2 + Math.cos(angle) * r * 1.5
    const cy = H / 2 + Math.sin(angle) * r * 0.85
    const type = i < badCount ? 'bad' : (i % 3 === 0 ? 'grounded' : 'normal')
    nodes.push({
      x: Math.max(MARGIN, Math.min(W - MARGIN, cx)),
      y: Math.max(MARGIN, Math.min(H - MARGIN, cy)),
      type,
      glyph: type === 'bad' ? '✗' : (type === 'grounded' ? '◈' : '◆'),
      bubble: null,
      pulse: false,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.2,
    })
  }
  return nodes
}

// ── Build sparse edge list ──────────────────────────────────────────
function buildEdges(n) {
  const edges = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(i - j) <= 2 || (i === 0 && j === n - 1)) edges.push([i, j])
    }
  }
  return edges
}

const nodes      = ref(buildNodes(visualCount.value))
const edges      = ref(buildEdges(visualCount.value))
const activeEdge = ref(-1)

// ── Animation loop ──────────────────────────────────────────────────
let animFrame = null, bubbleInterval = null, bubbleClearTimer = null, edgeInterval = null

function drift() {
  if (!props.active) { animFrame = requestAnimationFrame(drift); return }
  nodes.value = nodes.value.map(n => {
    let { x, y, vx, vy } = n
    x += vx; y += vy
    if (x < MARGIN || x > W - MARGIN) vx *= -1
    if (y < MARGIN || y > H - MARGIN) vy *= -1
    return { ...n, x, y, vx, vy }
  })
  animFrame = requestAnimationFrame(drift)
}

function rotateBubbles() {
  if (!props.active) {
    nodes.value = nodes.value.map(n => ({ ...n, bubble: null, pulse: false }))
    return
  }
  const i = Math.floor(Math.random() * nodes.value.length)
  nodes.value = nodes.value.map((n, idx) => ({
    ...n,
    bubble: idx === i ? BUBBLES[Math.floor(Math.random() * BUBBLES.length)] : null,
    pulse:  idx === i,
  }))
  clearTimeout(bubbleClearTimer)
  bubbleClearTimer = setTimeout(() => {
    nodes.value = nodes.value.map(n => ({ ...n, bubble: null, pulse: false }))
  }, 1400)
}

function rotateEdge() {
  if (!props.active) { activeEdge.value = -1; return }
  activeEdge.value = Math.floor(Math.random() * edges.value.length)
  setTimeout(() => { activeEdge.value = -1 }, 700)
}

onMounted(() => {
  animFrame     = requestAnimationFrame(drift)
  bubbleInterval = setInterval(rotateBubbles, 1800)
  edgeInterval   = setInterval(rotateEdge,   900)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animFrame)
  clearInterval(bubbleInterval)
  clearTimeout(bubbleClearTimer)
  clearInterval(edgeInterval)
})

// ── Helpers ─────────────────────────────────────────────────────────
function bubbleWidth(text) { return text.length * 7.4 + 18 }
function bubblePointer(n)  {
  const bx = 16, by = -10
  return `${bx},${by} ${bx+8},${by-2} ${bx+4},${by+4}`
}
</script>

<style scoped>
.flock-wrap {
  width: 100%;
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(5,10,21,.95) 0%, rgba(13,21,40,.9) 50%, rgba(22,26,60,.85) 100%);
  border: 1px solid rgba(99,102,241,.18);
  position: relative;
  box-shadow: 0 4px 32px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.03);
}
.flock-wrap.paused { opacity: .45; filter: saturate(.5); }
.flock-svg { width: 100%; display: block; }

/* ── Edges ──────────────────────────────────────────────────────── */
.edge-line {
  stroke: rgba(99,102,241,.12);
  stroke-width: 1;
  transition: stroke .3s, stroke-width .3s;
}
.edge-line.active {
  stroke: rgba(99,102,241,.55);
  stroke-width: 1.8;
  filter: drop-shadow(0 0 3px rgba(99,102,241,.6));
}

/* ── Nodes ──────────────────────────────────────────────────────── */
.node-body  { fill: rgba(99,102,241,.18); stroke: rgba(99,102,241,.5); stroke-width: 1.5; }
.node-inner { fill: rgba(99,102,241,.35); }
.agent-node.bad .node-body  { fill: rgba(244,63,94,.15); stroke: rgba(244,63,94,.5); }
.agent-node.bad .node-inner { fill: rgba(244,63,94,.3); }
.agent-node.grounded .node-body  { fill: rgba(20,184,166,.15); stroke: rgba(20,184,166,.5); }
.agent-node.grounded .node-inner { fill: rgba(20,184,166,.3); }

.node-glyph {
  font-size: 8px; fill: rgba(255,255,255,.7);
  font-family: monospace; pointer-events: none;
}

/* ── Pulse ring ─────────────────────────────────────────────────── */
.pulse-ring {
  fill: none;
  stroke: rgba(99,102,241,.4);
  stroke-width: 1.5;
  animation: pulse-out 1.4s ease-out forwards;
}
@keyframes pulse-out {
  0%   { r: 14; opacity: .8; }
  100% { r: 28; opacity: 0; }
}

/* ── Speech bubble ──────────────────────────────────────────────── */
.bubble-rect {
  fill: rgba(22,26,60,.95);
  stroke: rgba(99,102,241,.4);
  stroke-width: .8;
}
.bubble-text {
  font-size: 8px; fill: rgba(255,255,255,.8);
  font-family: monospace; pointer-events: none;
}
.bubble-group {
  animation: bubble-pop .2s var(--ease-spring, cubic-bezier(0.34,1.56,0.64,1)) both;
}
@keyframes bubble-pop {
  from { opacity: 0; transform: scale(.6) translateY(6px); }
  to   { opacity: 1; transform: scale(1)  translateY(0); }
}

/* ── LLM badge ──────────────────────────────────────────────────── */
.llm-badge-rect {
  fill: rgba(245,158,11,.12);
  stroke: rgba(245,158,11,.3);
  stroke-width: .8;
}
.llm-badge-text {
  font-size: 9px; fill: rgba(245,158,11,.9);
  font-family: monospace;
}

/* ── Count ──────────────────────────────────────────────────────── */
.count-text {
  font-size: 9px; fill: rgba(255,255,255,.25);
  font-family: monospace;
}
</style>
