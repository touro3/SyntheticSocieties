<template>
  <div class="shell" :class="{ 'sidebar-collapsed': collapsed, 'mobile-open': mobileOpen }">
    <!-- Sidebar -->
    <aside class="sidebar" :class="{ open: mobileOpen }">
      <div class="sidebar-top">
        <div class="brand-row">
          <router-link to="/" class="brand">
            <div class="brand-icon-wrap">
              <span class="brand-gem">◆</span>
            </div>
            <span class="brand-text">
              <span class="brand-name">Synthetic<span class="brand-name-accent">Societies</span></span>
              <span class="brand-sub">BGF · v2</span>
            </span>
          </router-link>
          <button class="collapse-btn" @click="collapsed = !collapsed"
            :title="collapsed ? 'Expand sidebar' : 'Collapse sidebar'"
            :aria-label="collapsed ? 'Expand sidebar' : 'Collapse sidebar'">
            {{ collapsed ? '›' : '‹' }}
          </button>
        </div>

        <nav class="nav">
          <router-link v-for="l in links" :key="l.to" :to="l.to"
            class="nav-item" active-class="active" exact-active-class="exact-active"
            :title="l.title || l.label"
            @click="mobileOpen = false">
            <span class="nav-icon">{{ l.icon }}</span>
            <span class="nav-label">{{ l.label }}</span>
            <span v-if="l.badge" class="nav-badge">{{ l.badge }}</span>
          </router-link>
        </nav>
      </div>

      <div class="sidebar-bottom">
        <div class="api-status" :class="online ? 'on' : 'off'">
          <span class="status-dot"></span>
          <span class="status-text">API {{ online ? 'Online' : 'Offline' }}</span>
        </div>
      </div>
    </aside>

    <!-- Main content -->
    <div class="main-area">
      <div class="scroll-bar" aria-hidden="true"></div>
      <main class="page-content">
        <router-view v-slot="{ Component }">
          <transition name="page" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>
    </div>

    <!-- Mobile overlay -->
    <div v-if="mobileOpen" class="mobile-overlay" @click="mobileOpen = false" />

    <!-- Mobile menu toggle -->
    <button class="mobile-toggle" @click="mobileOpen = !mobileOpen" aria-label="Open navigation menu">
      <span>☰</span>
    </button>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { api } from './api/index.js'

const online     = ref(false)
const collapsed  = ref(localStorage.getItem('sidebar-collapsed') === '1')
const mobileOpen = ref(false)

watch(collapsed, v => localStorage.setItem('sidebar-collapsed', v ? '1' : '0'))

const links = [
  { to: '/',            icon: '⬡', label: 'Dashboard' },
  { to: '/run',         icon: '▶', label: 'Run Simulation' },
  { to: '/experiments', icon: '◫', label: 'Experiments' },
  { to: '/human-eval',  icon: '◈', label: 'Human Eval', title: 'Vignette study — rate LLM vs human behavioral realism' },
]

let _keepAlive = null

onMounted(async () => {
  try { await api.health(); online.value = true } catch {}

  // Ping every 4.5 minutes to prevent Render free-tier spin-down (15-min idle timeout).
  // Uses /ping (zero I/O) so it doesn't count as real traffic for rate limits.
  _keepAlive = setInterval(async () => {
    try { await api.ping() } catch {}
  }, 4.5 * 60 * 1000)

  const supportsScroll = typeof CSS !== 'undefined' && CSS.supports?.('animation-timeline','scroll()')
  if (!supportsScroll) {
    const bar = document.querySelector('.scroll-bar')
    window.addEventListener('scroll', () => {
      const pct = window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
      if (bar) bar.style.transform = `scaleX(${pct})`
    }, { passive: true })
  }

  window.addEventListener('scroll', () => {
    document.documentElement.style.setProperty('--scroll-y', `${window.scrollY}px`)
  }, { passive: true })
})

onUnmounted(() => {
  if (_keepAlive) clearInterval(_keepAlive)
})
</script>

<style>
/* ── Google Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ───────────────────────────────────────────────── */
:root {
  --bg:      #050a15;
  --bg2:     #080f1e;
  --bg3:     #0d1528;
  --bg4:     #162038;
  --bg5:     #1c2847;
  --border:  rgba(255,255,255,.06);
  --border2: rgba(255,255,255,.10);
  --border3: rgba(99,102,241,.20);
  --text:    #f0f4ff;
  --text2:   #9db0cc;
  --text3:   #4d6080;
  --blue:    #6366f1;
  --blue2:   #818cf8;
  --indigo:  #4f46e5;
  --purple:  #8b5cf6;
  --violet:  #a78bfa;
  --cyan:    #22d3ee;
  --teal:    #14b8a6;
  --amber:   #f59e0b;
  --rose:    #f43f5e;
  --green:   #10b981;
  --grad:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #22d3ee 100%);
  --grad2:   linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  --glow:    0 0 40px rgba(99,102,241,.15), 0 0 80px rgba(99,102,241,.07);
  --glow-cyan: 0 0 30px rgba(34,211,238,.12);
  --sidebar-w: 228px;
  --ease-spring:       cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-spring-stiff: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --ease-out:          cubic-bezier(0.16, 1, 0.3, 1);
  --scroll-y: 0px;
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a { color: var(--blue); text-decoration: none; }
a:hover { color: var(--blue2); text-decoration: underline; }
button { font-family: inherit; cursor: pointer; border: none; outline: none; }
button:focus-visible { outline: 2px solid var(--blue2); outline-offset: 2px; border-radius: 6px; }
input, select, textarea {
  font-family: inherit;
  background: var(--bg3);
  border: 1px solid var(--border2);
  color: var(--text);
  border-radius: 10px;
  padding: 9px 13px;
  font-size: .875rem;
  width: 100%;
  transition: border-color .15s, box-shadow .15s, background .15s;
}
input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--blue);
  background: var(--bg4);
  box-shadow: 0 0 0 3px rgba(99,102,241,.12);
}
input::placeholder, textarea::placeholder { color: var(--text3); }
select option { background: var(--bg3); color: var(--text); }

/* ── Animated aurora background ─────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 900px 600px at 10% 5%,  rgba(99,102,241,.09),  transparent),
    radial-gradient(ellipse 700px 700px at 90% 90%, rgba(139,92,246,.07),  transparent),
    radial-gradient(ellipse 600px 400px at 60% 30%, rgba(34,211,238,.04),  transparent),
    radial-gradient(ellipse 500px 500px at 20% 80%, rgba(79,70,229,.05),   transparent);
  pointer-events: none; z-index: 0;
  animation: aurora 20s ease-in-out infinite alternate;
}
@keyframes aurora {
  0%   { opacity: 1; }
  50%  { opacity: .7; }
  100% { opacity: 1; }
}

/* Subtle dot grid */
body::after {
  content: '';
  position: fixed; inset: 0;
  background-image: radial-gradient(circle, rgba(99,102,241,.04) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none; z-index: 0;
}

/* ── Card ───────────────────────────────────────────────────────── */
.card {
  background: rgba(13, 21, 40, 0.80);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px;
  box-shadow:
    0 4px 24px rgba(0,0,0,.3),
    0 1px 0 rgba(255,255,255,.03) inset,
    0 -1px 0 rgba(0,0,0,.2) inset;
  position: relative;
  transition: border-color .25s, box-shadow .25s;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 1px; border-radius: 16px 16px 0 0;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,.18), transparent);
  pointer-events: none;
}
.card:hover {
  border-color: rgba(99,102,241,.18);
  box-shadow:
    0 8px 40px rgba(0,0,0,.35),
    0 0 0 1px rgba(99,102,241,.08) inset,
    var(--glow);
}

/* ── Card glow variant (featured cards) ─────────────────────────── */
.card-glow {
  border-color: rgba(99,102,241,.2);
  box-shadow:
    0 4px 24px rgba(0,0,0,.3),
    0 0 60px rgba(99,102,241,.08),
    inset 0 1px 0 rgba(255,255,255,.04);
}

/* ── Buttons ────────────────────────────────────────────────────── */
.btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 9px 18px; border-radius: 10px;
  font-size: .875rem; font-weight: 500;
  transition: transform .35s var(--ease-spring), background .18s, box-shadow .18s, border-color .18s;
  will-change: transform; cursor: pointer;
  white-space: nowrap; letter-spacing: -.01em;
}
.btn:hover  { transform: translateY(-2px) scale(1.02); text-decoration: none; }
.btn:active { transform: scale(0.94) !important; transition-duration: .07s !important; }

.btn-primary {
  background: linear-gradient(135deg, var(--indigo), var(--blue));
  color: #fff; border: none;
  box-shadow: 0 2px 12px rgba(99,102,241,.3);
}
.btn-primary:hover {
  background: linear-gradient(135deg, #5850eb, #6366f1);
  box-shadow: 0 6px 24px rgba(99,102,241,.45), 0 2px 6px rgba(99,102,241,.2);
}

.btn-outline {
  background: transparent; color: var(--text);
  border: 1px solid var(--border2);
}
.btn-outline:hover {
  border-color: var(--blue2);
  background: rgba(99,102,241,.06);
  color: #fff;
}

.btn-ghost {
  background: rgba(99,102,241,.08);
  color: var(--blue2); border: none;
}
.btn-ghost:hover { background: rgba(99,102,241,.16); color: var(--violet); }

.btn-danger {
  background: rgba(244,63,94,.08);
  color: var(--rose);
  border: 1px solid rgba(244,63,94,.2);
}
.btn-danger:hover { background: rgba(244,63,94,.18); }

.btn-sm { padding: 5px 12px; font-size: .8rem; border-radius: 8px; }

/* ── Badges ─────────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 9px; border-radius: 20px;
  font-size: .7rem; font-weight: 600; letter-spacing: .03em;
}
.badge-blue   { background: rgba(99,102,241,.12);  color: var(--blue2);  border: 1px solid rgba(99,102,241,.18); }
.badge-green  { background: rgba(16,185,129,.1);    color: var(--green);  border: 1px solid rgba(16,185,129,.18); }
.badge-amber  { background: rgba(245,158,11,.1);    color: var(--amber);  border: 1px solid rgba(245,158,11,.18); }
.badge-rose   { background: rgba(244,63,94,.1);     color: var(--rose);   border: 1px solid rgba(244,63,94,.18); }
.badge-teal   { background: rgba(20,184,166,.1);    color: var(--teal);   border: 1px solid rgba(20,184,166,.18); }
.badge-purple { background: rgba(139,92,246,.1);    color: var(--violet); border: 1px solid rgba(139,92,246,.18); }
.badge-cyan   { background: rgba(34,211,238,.08);   color: var(--cyan);   border: 1px solid rgba(34,211,238,.16); }

/* ── Utilities ──────────────────────────────────────────────────── */
.mono    { font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace; }
.section-title {
  font-size: .92rem; font-weight: 700; margin-bottom: 16px; color: var(--text);
  letter-spacing: -.01em;
}
.page-title {
  font-size: 1.85rem; font-weight: 800; letter-spacing: -.03em; color: var(--text);
  line-height: 1.15;
}
.page-sub { color: var(--text2); font-size: .9rem; margin-top: 7px; line-height: 1.5; }
.empty    { text-align: center; padding: 52px 24px; color: var(--text3); font-size: .9rem; }

/* ── Animations ─────────────────────────────────────────────────── */
.spin { display: inline-block; animation: spin 1s linear infinite; }
@keyframes spin    { to { transform: rotate(360deg); } }
@keyframes pulse   { 0%,100% { opacity:1; } 50% { opacity:.3; } }
@keyframes glow-pulse {
  0%,100% { box-shadow: 0 0 18px rgba(99,102,241,.25); }
  50%      { box-shadow: 0 0 36px rgba(99,102,241,.55); }
}
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(16px) scale(.98); }
  to   { opacity: 1; transform: none; }
}
@keyframes slide-in {
  from { opacity: 0; transform: translateX(-12px); }
  to   { opacity: 1; transform: none; }
}

/* ── Staggered reveal ───────────────────────────────────────────── */
.reveal {
  opacity: 0;
  transform: translateY(20px) scale(0.98);
  transition:
    opacity  .5s calc(var(--i, 0) * 70ms) var(--ease-spring),
    transform .5s calc(var(--i, 0) * 70ms) var(--ease-spring);
}
.reveal.revealed { opacity: 1; transform: none; }

/* ── Scroll progress bar ────────────────────────────────────────── */
.scroll-bar {
  position: fixed; top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--grad);
  transform-origin: left center;
  transform: scaleX(0);
  z-index: 9999;
  pointer-events: none;
}
@supports (animation-timeline: scroll()) {
  .scroll-bar {
    transform: none;
    animation: scroll-track linear both;
    animation-timeline: scroll(root block);
  }
  @keyframes scroll-track { from { transform: scaleX(0); } to { transform: scaleX(1); } }
}

/* ── Page transitions ───────────────────────────────────────────── */
.page-enter-active { transition: opacity .25s var(--ease-out), transform .35s var(--ease-spring); }
.page-leave-active { transition: opacity .15s ease, transform .15s ease; }
.page-enter-from   { opacity: 0; transform: translateY(10px) scale(.99); }
.page-leave-to     { opacity: 0; transform: translateY(-5px); }

/* ── Form fields ────────────────────────────────────────────────── */
.field { display: flex; flex-direction: column; gap: 7px; }
.field label { font-size: .74rem; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .06em; }
.field .hint  { font-size: .72rem; color: var(--text3); }

/* ── Error / success ────────────────────────────────────────────── */
.error-box {
  background: rgba(244,63,94,.07); border: 1px solid rgba(244,63,94,.18);
  border-radius: 10px; padding: 10px 14px; font-size: .84rem; color: var(--rose);
}
.success-box {
  background: rgba(16,185,129,.07); border: 1px solid rgba(16,185,129,.18);
  border-radius: 10px; padding: 10px 14px; font-size: .84rem; color: var(--green);
}
</style>

<style scoped>
/* ── Shell ──────────────────────────────────────────────────────── */
.shell {
  display: flex;
  min-height: 100vh;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
.sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  display: flex; flex-direction: column;
  background: rgba(5, 10, 21, 0.96);
  backdrop-filter: blur(32px);
  -webkit-backdrop-filter: blur(32px);
  border-right: 1px solid var(--border);
  position: fixed; top: 0; left: 0; bottom: 0;
  z-index: 200;
  transition: width .28s var(--ease-out), transform .28s var(--ease-out);
  overflow: hidden;
}

/* Right-side gradient edge */
.sidebar::after {
  content: '';
  position: absolute; top: 0; right: 0; bottom: 0; width: 1px;
  background: linear-gradient(to bottom, transparent, rgba(99,102,241,.15) 30%, rgba(99,102,241,.12) 70%, transparent);
  pointer-events: none;
}

.sidebar-top {
  display: flex; flex-direction: column; flex: 1;
  overflow-y: auto; padding: 20px 12px 12px;
}

/* ── Brand row ──────────────────────────────────────────────────── */
.brand-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 0 18px;
}

/* ── Brand ──────────────────────────────────────────────────────── */
.brand {
  display: flex; align-items: center; gap: 11px;
  padding: 8px 8px 0;
  text-decoration: none; color: inherit; flex: 1; min-width: 0;
}
.brand:hover { text-decoration: none; }

/* ── Sidebar collapse button ────────────────────────────────────── */
.collapse-btn {
  width: 24px; height: 24px; border-radius: 6px; flex-shrink: 0;
  background: rgba(255,255,255,.04); border: 1px solid var(--border2);
  color: var(--text3); font-size: .8rem; line-height: 1;
  display: flex; align-items: center; justify-content: center;
  transition: background .15s, color .15s, border-color .15s;
  cursor: pointer; padding: 0; margin-right: 4px; margin-top: 4px;
  align-self: flex-start;
}
.collapse-btn:hover { background: rgba(99,102,241,.12); color: var(--blue2); border-color: rgba(99,102,241,.3); }
.brand-icon-wrap {
  width: 32px; height: 32px; border-radius: 9px;
  background: linear-gradient(135deg, rgba(99,102,241,.25), rgba(139,92,246,.2));
  border: 1px solid rgba(99,102,241,.3);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 0 20px rgba(99,102,241,.2);
}
.brand-gem {
  font-size: .75rem; color: var(--blue2);
  filter: drop-shadow(0 0 5px rgba(99,102,241,.9));
  animation: glow-pulse 3.5s ease-in-out infinite;
}
.brand-text { display: flex; flex-direction: column; }
.brand-name {
  font-size: .88rem; font-weight: 800; letter-spacing: -.02em;
  white-space: nowrap; color: var(--text);
}
.brand-name-accent { color: var(--blue2); }
.brand-sub  { font-size: .62rem; color: var(--text3); letter-spacing: .08em; margin-top: 1px; }

/* ── Nav ────────────────────────────────────────────────────────── */
.nav { display: flex; flex-direction: column; gap: 3px; }

.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 10px;
  font-size: .85rem; font-weight: 500; color: var(--text3);
  transition: background .15s, color .15s, transform .3s var(--ease-spring);
  white-space: nowrap; overflow: hidden;
  text-decoration: none; position: relative;
}
.nav-item:hover {
  background: rgba(255,255,255,.04);
  color: var(--text2);
  transform: translateX(3px);
  text-decoration: none;
}

/* Active: left gradient accent line */
.nav-item.active::before, .nav-item.exact-active::before {
  content: '';
  position: absolute; left: 0; top: 6px; bottom: 6px; width: 3px;
  border-radius: 0 2px 2px 0;
  background: var(--grad);
}
.nav-item.active, .nav-item.exact-active {
  background: rgba(99,102,241,.08);
  color: var(--text);
}
.nav-item.exact-active { color: var(--blue2); }

.nav-icon  { font-size: .88rem; flex-shrink: 0; width: 18px; text-align: center; }
.nav-label { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.nav-badge {
  background: var(--blue); color: #fff;
  font-size: .6rem; font-weight: 700;
  padding: 1px 6px; border-radius: 99px; flex-shrink: 0;
}

/* ── Sidebar bottom ─────────────────────────────────────────────── */
.sidebar-bottom {
  padding: 12px 12px 20px;
  border-top: 1px solid var(--border);
}
.api-status {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 9px;
  font-size: .75rem; color: var(--text3);
}
.api-status.on .status-dot  { background: var(--green); animation: pulse 2.5s infinite; box-shadow: 0 0 6px var(--green); }
.api-status.off .status-dot { background: var(--rose); }
.status-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}

/* ── Main area ──────────────────────────────────────────────────── */
.main-area {
  flex: 1;
  margin-left: var(--sidebar-w);
  min-height: 100vh;
  transition: margin-left .28s var(--ease-out);
  position: relative; z-index: 1;
}

.page-content {
  max-width: 1180px;
  margin: 0 auto;
  padding: 44px 36px 100px;
}

/* ── Collapsed ──────────────────────────────────────────────────── */
.sidebar-collapsed .sidebar { width: 58px; }
.sidebar-collapsed .brand-text,
.sidebar-collapsed .nav-label,
.sidebar-collapsed .nav-badge,
.sidebar-collapsed .status-text,
.sidebar-collapsed .collapse-btn { display: none; }
.sidebar-collapsed .main-area { margin-left: 58px; }
.sidebar-collapsed .nav-item { justify-content: center; padding: 10px; }
.sidebar-collapsed .nav-item::before { display: none; }
.sidebar-collapsed .brand-row { justify-content: center; padding-bottom: 18px; }
.sidebar-collapsed .brand { justify-content: center; padding: 8px 0 0; flex: none; }
.sidebar-collapsed .brand-icon-wrap { margin: 0; }

/* ── Mobile ─────────────────────────────────────────────────────── */
.mobile-toggle {
  display: none; position: fixed; top: 14px; left: 14px;
  width: 40px; height: 40px; border-radius: 10px;
  background: rgba(13,21,40,.92);
  border: 1px solid var(--border2);
  color: var(--text2); font-size: 1rem; z-index: 300;
  box-shadow: 0 2px 12px rgba(0,0,0,.4);
  backdrop-filter: blur(12px);
}
.mobile-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.7);
  backdrop-filter: blur(4px);
  z-index: 199;
}

@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
    width: var(--sidebar-w) !important;
  }
  .sidebar.open { transform: translateX(0); }
  .sidebar-collapsed .sidebar { transform: translateX(-100%); }
  .main-area, .sidebar-collapsed .main-area { margin-left: 0; }
  .mobile-toggle, .mobile-overlay { display: flex; align-items: center; justify-content: center; }
  .page-content { padding: 68px 18px 100px; } /* top offset clears the mobile toggle */
}
</style>
