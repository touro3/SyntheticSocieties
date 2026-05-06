<template>
  <nav class="navbar">
    <div class="container nav-inner">
      <router-link to="/" class="brand">
        <span class="brand-icon">◆</span>
        <span class="brand-name">Synthetic Societies</span>
        <span class="brand-sub">BGF</span>
      </router-link>

      <div class="nav-links">
        <router-link to="/" class="nav-link" exact-active-class="active">Dashboard</router-link>
        <router-link to="/experiments" class="nav-link" active-class="active">Experiments</router-link>
        <router-link to="/run" class="nav-link run-btn btn btn-primary" active-class="">Run Sim</router-link>
      </div>

      <div class="status-wrap">
        <span class="status-dot" :class="online ? 'on' : 'off'" />
        <span class="status-label">{{ online ? 'Online' : 'Offline' }}</span>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api/index.js'

const online = ref(false)
onMounted(async () => {
  try { await api.health(); online.value = true } catch {}
})
</script>

<style scoped>
.navbar {
  /* Glassmorphism navbar */
  background: rgba(10, 14, 26, 0.78);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border-bottom: 1px solid rgba(255,255,255,0.055);
  box-shadow:
    0 1px 0 rgba(99,102,241,0.10),
    0 4px 24px rgba(0,0,0,0.22);
  position: sticky; top: 0; z-index: 100;
}

.nav-inner {
  display: flex; align-items: center; gap: 20px; height: 58px;
}

.brand {
  display: flex; align-items: center; gap: 8px;
  text-decoration: none; color: var(--text); margin-right: auto;
  transition: opacity .2s;
}
.brand:hover { opacity: .8; text-decoration: none; }
.brand-icon {
  color: var(--blue); font-size: .9rem;
  filter: drop-shadow(0 0 6px rgba(99,102,241,.6));
  animation: glow-pulse 3s ease-in-out infinite;
}
.brand-name { font-weight: 600; font-size: .95rem; letter-spacing: -.01em; }
.brand-sub {
  font-size: .66rem; font-weight: 700; letter-spacing: .08em;
  background: rgba(99,102,241,.1);
  border: 1px solid rgba(99,102,241,.2);
  color: var(--blue); padding: 2px 7px; border-radius: 6px;
}

.nav-links { display: flex; align-items: center; gap: 2px; }

.nav-link {
  padding: 7px 14px; border-radius: 8px;
  font-size: .84rem; font-weight: 500; color: var(--text2);
  text-decoration: none;
  transition:
    color .15s,
    background .15s,
    transform .35s var(--ease-spring),
    box-shadow .25s;
  position: relative;
}
.nav-link:hover {
  color: var(--text);
  background: rgba(255,255,255,0.05);
  transform: translateY(-1px);
  text-decoration: none;
}
.nav-link.active {
  color: var(--text);
  background: rgba(99,102,241,.12);
}
.nav-link.active::after {
  content: '';
  position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%);
  width: 16px; height: 2px; border-radius: 1px;
  background: var(--blue);
}

.run-btn {
  color: #fff !important;
  padding: 7px 18px;
  background: var(--blue);
  box-shadow: 0 2px 12px rgba(99,102,241,.3);
}
.run-btn:hover {
  background: #5558e6 !important;
  box-shadow: 0 4px 20px rgba(99,102,241,.5);
}
.run-btn.active { background: #5558e6 !important; }
.run-btn::after { display: none; }

.status-wrap { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.status-dot  { width: 7px; height: 7px; border-radius: 50%; }
.status-dot.on  { background: var(--green); animation: pulse 2s infinite; }
.status-dot.off { background: var(--rose); }
.status-label { font-size: .74rem; color: var(--text3); }

@media (max-width: 600px) {
  .nav-link:not(.run-btn) { display: none; }
  .status-label { display: none; }
}
</style>
