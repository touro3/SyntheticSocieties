<template>
  <div class="app">
    <!-- Scroll-driven progress bar: CSS handles it natively where supported;
         JS updates --scroll-pct for the fallback via onScroll. -->
    <div class="scroll-progress" aria-hidden="true"></div>

    <NavBar />

    <main class="main">
      <router-view v-slot="{ Component }">
        <transition name="page" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount } from 'vue'
import NavBar from './components/NavBar.vue'

// JS fallback for scroll-driven progress — only fires when the native
// CSS animation-timeline: scroll() is not supported (Safari < 18).
const supportsScrollTimeline =
  typeof CSS !== 'undefined' && CSS.supports('animation-timeline', 'scroll()')

function onScroll() {
  const pct = window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
  document.documentElement.style.setProperty('--scroll-pct', pct)
}

// JS fallback for scroll-driven parallax on hero elements
function onScrollParallax() {
  document.documentElement.style.setProperty('--scroll-y', `${window.scrollY}px`)
}

onMounted(() => {
  if (!supportsScrollTimeline) {
    window.addEventListener('scroll', onScroll, { passive: true })
  }
  window.addEventListener('scroll', onScrollParallax, { passive: true })
})
onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
  window.removeEventListener('scroll', onScrollParallax)
})
</script>

<style>
/* ── Design tokens ────────────────────────────────────────────────────── */
:root {
  --bg:      #0a0e1a;
  --bg2:     #111827;
  --bg3:     #1a1f35;
  --bg4:     #222845;
  --border:  #2a3050;
  --text:    #f1f5f9;
  --text2:   #94a3b8;
  --text3:   #64748b;
  --blue:    #6366f1;
  --purple:  #8b5cf6;
  --teal:    #14b8a6;
  --amber:   #f59e0b;
  --rose:    #f43f5e;
  --green:   #10b981;
  --grad:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
  --glow:    0 0 40px rgba(99,102,241,0.18);

  /* Spring easings */
  --ease-spring:       cubic-bezier(0.34, 1.56, 0.64, 1);   /* slight overshoot */
  --ease-spring-stiff: cubic-bezier(0.175, 0.885, 0.32, 1.275); /* more bounce */
  --ease-out:          cubic-bezier(0.16, 1, 0.3, 1);

  /* Scroll-driven fallback value */
  --scroll-pct: 0;
  --scroll-y: 0px;
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

/* Animated mesh background */
body::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 600px 600px at 20% 20%, rgba(99,102,241,.07), transparent),
    radial-gradient(ellipse 900px 900px at 80% 80%, rgba(139,92,246,.05), transparent),
    radial-gradient(ellipse 500px 500px at 60% 30%, rgba(20,184,166,.04), transparent);
  pointer-events: none; z-index: 0;
}

a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

button { font-family: inherit; cursor: pointer; border: none; outline: none; }
input, select, textarea {
  font-family: inherit;
  background: rgba(17,24,39,0.8);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px; padding: 9px 13px; font-size: .88rem; width: 100%;
  transition: border-color .15s, box-shadow .15s;
}
input:focus, select:focus, textarea:focus {
  outline: none; border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(99,102,241,.15);
}

.container {
  max-width: 1100px; margin: 0 auto; padding: 0 24px;
  position: relative; z-index: 1;
}

/* ── Glassmorphism card ───────────────────────────────────────────────── */
.card {
  background: rgba(26, 31, 53, 0.72);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.055);
  border-radius: 14px; padding: 24px;
  box-shadow:
    0 4px 24px rgba(0,0,0,0.28),
    inset 0 1px 0 rgba(255,255,255,0.045),
    0 0 0 1px rgba(99,102,241,0.07);
  transition:
    border-color .3s,
    box-shadow .3s,
    transform .45s var(--ease-spring);
}
.card:hover {
  border-color: rgba(99,102,241,.28);
  box-shadow:
    0 8px 40px rgba(0,0,0,0.35),
    inset 0 1px 0 rgba(255,255,255,0.06),
    0 0 0 1px rgba(99,102,241,.14),
    var(--glow);
}

/* ── Spring button ───────────────────────────────────────────────────── */
.btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 10px 20px; border-radius: 10px;
  font-size: .88rem; font-weight: 500;
  transition:
    transform .35s var(--ease-spring),
    background .18s,
    box-shadow .18s,
    border-color .18s,
    color .15s;
  will-change: transform;
}
.btn:hover  { transform: translateY(-2px) scale(1.025); }
.btn:active { transform: scale(0.93) !important; transition-duration: .08s !important; }

.btn-primary { background: var(--blue); color: #fff; }
.btn-primary:hover { background: #5558e6; box-shadow: 0 6px 24px rgba(99,102,241,.4); }

.btn-outline {
  background: transparent; color: var(--text2);
  border: 1px solid var(--border);
}
.btn-outline:hover { border-color: var(--blue); color: var(--text); background: rgba(99,102,241,.07); }

.btn-ghost { background: rgba(99,102,241,.08); color: var(--blue); }
.btn-ghost:hover { background: rgba(99,102,241,.16); }

/* ── Badges ──────────────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 20px;
  font-size: .74rem; font-weight: 500; letter-spacing: .04em;
}
.badge-blue   { background: rgba(99,102,241,.12); color: var(--blue);   border: 1px solid rgba(99,102,241,.2); }
.badge-green  { background: rgba(16,185,129,.12);  color: var(--green);  border: 1px solid rgba(16,185,129,.2); }
.badge-amber  { background: rgba(245,158,11,.12);  color: var(--amber);  border: 1px solid rgba(245,158,11,.2); }
.badge-rose   { background: rgba(244,63,94,.12);   color: var(--rose);   border: 1px solid rgba(244,63,94,.2); }
.badge-teal   { background: rgba(20,184,166,.12);  color: var(--teal);   border: 1px solid rgba(20,184,166,.2); }

.mono { font-family: 'JetBrains Mono', monospace; }
.section-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; }
.empty { text-align: center; padding: 48px 24px; color: var(--text3); font-size: .9rem; }

/* ── Staggered reveal ────────────────────────────────────────────────── */
.reveal {
  opacity: 0;
  transform: translateY(20px) scale(0.98);
  transition:
    opacity  .55s calc(var(--i, 0) * 70ms) var(--ease-spring),
    transform .55s calc(var(--i, 0) * 70ms) var(--ease-spring);
}
.reveal.revealed {
  opacity: 1;
  transform: none;
}

/* ── Scroll-driven progress bar ──────────────────────────────────────── */
.scroll-progress {
  position: fixed; top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--grad);
  transform-origin: left center;
  transform: scaleX(var(--scroll-pct, 0));
  z-index: 9999;
  pointer-events: none;
}

/* Native CSS scroll-driven — Chrome 115+, Firefox 110+             */
/* When supported, the JS fallback scroll listener is never added.  */
@supports (animation-timeline: scroll()) {
  .scroll-progress {
    transform: none;           /* let animation handle it */
    animation: scroll-track linear both;
    animation-timeline: scroll(root block);
  }
  @keyframes scroll-track {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
  }
}

/* ── Page route transition (spring slide) ────────────────────────── */
.page-enter-active {
  transition: opacity .3s var(--ease-out), transform .4s var(--ease-spring);
}
.page-leave-active {
  transition: opacity .18s ease, transform .18s ease;
}
.page-enter-from {
  opacity: 0;
  transform: translateY(14px) scale(0.99);
}
.page-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

/* ── Utilities ───────────────────────────────────────────────────── */
.spin { display: inline-block; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .35; }
}

@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,.2); }
  50%       { box-shadow: 0 0 40px rgba(99,102,241,.45); }
}
</style>

<style scoped>
.app { display: flex; flex-direction: column; min-height: 100vh; }
.main { flex: 1; padding: 32px 0 64px; }
</style>
