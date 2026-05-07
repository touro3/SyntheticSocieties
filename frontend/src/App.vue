<template>
  <div class="shell" :class="{ 'sidebar-collapsed': collapsed }">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-top">
        <div class="brand" @click="collapsed = !collapsed" title="Toggle sidebar">
          <span class="brand-gem">◆</span>
          <span class="brand-text">
            <span class="brand-name">Synthetic</span>
            <span class="brand-sub">Societies · BGF</span>
          </span>
        </div>

        <nav class="nav">
          <router-link v-for="l in links" :key="l.to" :to="l.to"
            class="nav-item" active-class="active" exact-active-class="exact-active">
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
      <!-- Scroll progress bar -->
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
    <button class="mobile-toggle" @click="mobileOpen = !mobileOpen" aria-label="Menu">
      <span>☰</span>
    </button>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { api } from './api/index.js'

const online      = ref(false)
const collapsed   = ref(false)
const mobileOpen  = ref(false)

const links = [
  { to: '/',           icon: '⬡', label: 'Dashboard' },
  { to: '/run',        icon: '▶', label: 'Run Simulation' },
  { to: '/experiments',icon: '◫', label: 'Experiments' },
]

onMounted(async () => {
  try { await api.health(); online.value = true } catch {}

  // Scroll-driven progress (JS fallback for Safari)
  const supportsScroll = typeof CSS !== 'undefined' && CSS.supports?.('animation-timeline','scroll()')
  if (!supportsScroll) {
    const bar = document.querySelector('.scroll-bar')
    window.addEventListener('scroll', () => {
      const pct = window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
      if (bar) bar.style.transform = `scaleX(${pct})`
    }, { passive: true })
  }

  // Parallax scroll var
  window.addEventListener('scroll', () => {
    document.documentElement.style.setProperty('--scroll-y', `${window.scrollY}px`)
  }, { passive: true })
})
</script>

<style>
/* ── Design tokens ───────────────────────────────────────────────── */
:root {
  --bg:      #0a0e1a;
  --bg2:     #0f1422;
  --bg3:     #161c30;
  --bg4:     #1e2540;
  --border:  rgba(255,255,255,.07);
  --border2: rgba(255,255,255,.12);
  --text:    #f1f5f9;
  --text2:   #94a3b8;
  --text3:   #475569;
  --blue:    #6366f1;
  --blue2:   #818cf8;
  --purple:  #8b5cf6;
  --teal:    #14b8a6;
  --amber:   #f59e0b;
  --rose:    #f43f5e;
  --green:   #10b981;
  --grad:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a78bfa 100%);
  --glow:    0 0 40px rgba(99,102,241,.18);
  --sidebar-w: 220px;
  --ease-spring:       cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-spring-stiff: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --ease-out:          cubic-bezier(0.16, 1, 0.3, 1);
  --scroll-y: 0px;
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
button { font-family: inherit; cursor: pointer; border: none; outline: none; }
input, select, textarea {
  font-family: inherit;
  background: var(--bg3);
  border: 1px solid var(--border2);
  color: var(--text);
  border-radius: 9px;
  padding: 9px 13px;
  font-size: .875rem;
  width: 100%;
  transition: border-color .15s, box-shadow .15s;
}
input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(99,102,241,.15);
}
input::placeholder, textarea::placeholder { color: var(--text3); }

/* ── Animated mesh background ───────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 700px 700px at 15% 10%, rgba(99,102,241,.06), transparent),
    radial-gradient(ellipse 900px 900px at 85% 85%, rgba(139,92,246,.045), transparent),
    radial-gradient(ellipse 500px 500px at 55% 35%, rgba(20,184,166,.03), transparent);
  pointer-events: none; z-index: 0;
}

/* ── Card ───────────────────────────────────────────────────────── */
.card {
  background: rgba(22, 28, 48, 0.75);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 4px 24px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
  transition: border-color .3s, box-shadow .3s;
}
.card:hover {
  border-color: rgba(99,102,241,.22);
  box-shadow: 0 8px 40px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.06), var(--glow);
}

/* ── Buttons ────────────────────────────────────────────────────── */
.btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 9px 18px; border-radius: 10px;
  font-size: .875rem; font-weight: 500;
  transition: transform .35s var(--ease-spring), background .18s, box-shadow .18s, border-color .18s;
  will-change: transform; cursor: pointer;
  white-space: nowrap;
}
.btn:hover  { transform: translateY(-2px) scale(1.025); text-decoration: none; }
.btn:active { transform: scale(0.94) !important; transition-duration: .07s !important; }

.btn-primary { background: var(--blue); color: #fff; border: none; }
.btn-primary:hover { background: #5558e6; box-shadow: 0 6px 24px rgba(99,102,241,.4); }

.btn-outline {
  background: transparent; color: var(--text2);
  border: 1px solid var(--border2);
}
.btn-outline:hover { border-color: var(--blue); color: var(--text); background: rgba(99,102,241,.07); }

.btn-ghost { background: rgba(99,102,241,.08); color: var(--blue); border: none; }
.btn-ghost:hover { background: rgba(99,102,241,.16); }

.btn-danger { background: rgba(244,63,94,.1); color: var(--rose); border: 1px solid rgba(244,63,94,.25); }
.btn-danger:hover { background: rgba(244,63,94,.2); }

.btn-sm { padding: 6px 12px; font-size: .8rem; border-radius: 8px; }

/* ── Badges ─────────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 20px;
  font-size: .72rem; font-weight: 500; letter-spacing: .03em;
}
.badge-blue   { background: rgba(99,102,241,.12);  color: var(--blue2);  border: 1px solid rgba(99,102,241,.2); }
.badge-green  { background: rgba(16,185,129,.12);   color: var(--green);  border: 1px solid rgba(16,185,129,.2); }
.badge-amber  { background: rgba(245,158,11,.12);   color: var(--amber);  border: 1px solid rgba(245,158,11,.2); }
.badge-rose   { background: rgba(244,63,94,.12);    color: var(--rose);   border: 1px solid rgba(244,63,94,.2); }
.badge-teal   { background: rgba(20,184,166,.12);   color: var(--teal);   border: 1px solid rgba(20,184,166,.2); }
.badge-purple { background: rgba(139,92,246,.12);   color: var(--purple); border: 1px solid rgba(139,92,246,.2); }

/* ── Utilities ──────────────────────────────────────────────────── */
.mono    { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
.section-title { font-size: 1rem; font-weight: 600; margin-bottom: 14px; }
.page-title    { font-size: 1.75rem; font-weight: 700; letter-spacing: -.02em; }
.page-sub      { color: var(--text2); font-size: .9rem; margin-top: 6px; }
.empty         { text-align: center; padding: 52px 24px; color: var(--text3); font-size: .9rem; }

.spin { display: inline-block; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }
@keyframes glow-pulse {
  0%,100% { box-shadow: 0 0 20px rgba(99,102,241,.2); }
  50%      { box-shadow: 0 0 40px rgba(99,102,241,.45); }
}
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(16px) scale(.98); }
  to   { opacity: 1; transform: none; }
}

/* ── Staggered reveal ───────────────────────────────────────────── */
.reveal {
  opacity: 0;
  transform: translateY(18px) scale(0.98);
  transition:
    opacity  .5s calc(var(--i, 0) * 65ms) var(--ease-spring),
    transform .5s calc(var(--i, 0) * 65ms) var(--ease-spring);
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
.page-enter-active { transition: opacity .28s var(--ease-out), transform .38s var(--ease-spring); }
.page-leave-active { transition: opacity .16s ease, transform .16s ease; }
.page-enter-from   { opacity: 0; transform: translateY(12px) scale(.99); }
.page-leave-to     { opacity: 0; transform: translateY(-6px); }

/* ── Form fields ────────────────────────────────────────────────── */
.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: .78rem; font-weight: 500; color: var(--text3); }
.field .hint  { font-size: .73rem; color: var(--text3); }

/* ── Error / success ────────────────────────────────────────────── */
.error-box {
  background: rgba(244,63,94,.07); border: 1px solid rgba(244,63,94,.2);
  border-radius: 10px; padding: 10px 14px; font-size: .84rem; color: var(--rose);
}
.success-box {
  background: rgba(16,185,129,.07); border: 1px solid rgba(16,185,129,.2);
  border-radius: 10px; padding: 10px 14px; font-size: .84rem; color: var(--green);
}
</style>

<style scoped>
/* ── Shell (sidebar + main) ─────────────────────────────────────── */
.shell {
  display: flex;
  min-height: 100vh;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
.sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  display: flex; flex-direction: column;
  background: rgba(10, 14, 26, 0.92);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border-right: 1px solid var(--border);
  position: fixed; top: 0; left: 0; bottom: 0;
  z-index: 200;
  transition: width .3s var(--ease-out), transform .3s var(--ease-out);
  overflow: hidden;
}

.sidebar-top { display: flex; flex-direction: column; flex: 1; overflow-y: auto; padding: 18px 12px 12px; }

/* ── Brand ──────────────────────────────────────────────────────── */
.brand {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 8px 20px;
  cursor: pointer; user-select: none;
  color: var(--text);
}
.brand-gem {
  font-size: .85rem; color: var(--blue);
  flex-shrink: 0;
  filter: drop-shadow(0 0 6px rgba(99,102,241,.7));
  animation: glow-pulse 3s ease-in-out infinite;
}
.brand-text { display: flex; flex-direction: column; }
.brand-name { font-size: .9rem; font-weight: 700; letter-spacing: -.01em; white-space: nowrap; }
.brand-sub  { font-size: .65rem; color: var(--text3); letter-spacing: .04em; white-space: nowrap; }

/* ── Nav ────────────────────────────────────────────────────────── */
.nav { display: flex; flex-direction: column; gap: 3px; }

.nav-item {
  display: flex; align-items: center; gap: 11px;
  padding: 9px 12px; border-radius: 10px;
  font-size: .86rem; font-weight: 500; color: var(--text3);
  transition: background .15s, color .15s, transform .3s var(--ease-spring);
  white-space: nowrap; overflow: hidden;
  text-decoration: none;
}
.nav-item:hover { background: rgba(255,255,255,.04); color: var(--text); transform: translateX(3px); text-decoration: none; }
.nav-item.active, .nav-item.exact-active {
  background: rgba(99,102,241,.12); color: var(--text);
}
.nav-item.exact-active { color: var(--blue2); }

.nav-icon  { font-size: .88rem; flex-shrink: 0; width: 18px; text-align: center; }
.nav-label { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.nav-badge {
  background: var(--blue); color: #fff;
  font-size: .62rem; font-weight: 700;
  padding: 1px 6px; border-radius: 99px;
  flex-shrink: 0;
}

.sidebar-bottom { padding: 12px 12px 18px; border-top: 1px solid var(--border); }
.api-status {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 9px;
  font-size: .75rem; color: var(--text3);
}
.api-status.on  .status-dot { background: var(--green); animation: pulse 2s infinite; }
.api-status.off .status-dot { background: var(--rose); }
.status-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}

/* ── Main area ──────────────────────────────────────────────────── */
.main-area {
  flex: 1;
  margin-left: var(--sidebar-w);
  min-height: 100vh;
  transition: margin-left .3s var(--ease-out);
  position: relative; z-index: 1;
}

.page-content {
  max-width: 1160px;
  margin: 0 auto;
  padding: 40px 32px 80px;
}

/* ── Collapsed sidebar ──────────────────────────────────────────── */
.sidebar-collapsed .sidebar { width: 56px; }
.sidebar-collapsed .brand-text,
.sidebar-collapsed .nav-label,
.sidebar-collapsed .nav-badge,
.sidebar-collapsed .status-text { display: none; }
.sidebar-collapsed .main-area { margin-left: 56px; }
.sidebar-collapsed .nav-item { justify-content: center; padding: 10px; }
.sidebar-collapsed .brand { justify-content: center; padding-bottom: 20px; }

/* ── Mobile ─────────────────────────────────────────────────────── */
.mobile-toggle {
  display: none; position: fixed; bottom: 20px; right: 20px;
  width: 48px; height: 48px; border-radius: 50%;
  background: var(--blue); color: #fff;
  font-size: 1.1rem; z-index: 300;
  box-shadow: 0 4px 20px rgba(99,102,241,.4);
}
.mobile-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.6); z-index: 199;
}

@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
    width: var(--sidebar-w) !important;
  }
  .sidebar-collapsed .sidebar {
    transform: translateX(-100%);
  }
  .main-area, .sidebar-collapsed .main-area { margin-left: 0; }
  .mobile-toggle, .mobile-overlay { display: flex; align-items: center; justify-content: center; }
  .page-content { padding: 24px 16px 80px; }
}
</style>
