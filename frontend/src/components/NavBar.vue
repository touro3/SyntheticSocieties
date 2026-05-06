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
        <router-link to="/run" class="nav-link btn btn-primary" active-class="">Run Sim</router-link>
      </div>

      <div class="status-dot" :class="online ? 'online' : 'offline'" :title="online ? 'API online' : 'API offline'"></div>
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
  background: rgba(10,14,26,.92);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 100;
}
.nav-inner {
  display: flex; align-items: center; gap: 24px;
  height: 58px;
}
.brand {
  display: flex; align-items: center; gap: 8px;
  text-decoration: none; color: var(--text); margin-right: auto;
}
.brand-icon { color: var(--blue); font-size: 1rem; }
.brand-name { font-weight: 600; font-size: 0.95rem; }
.brand-sub {
  font-size: 0.68rem; font-weight: 600; letter-spacing: .06em;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--text3); padding: 2px 7px; border-radius: 6px;
}
.nav-links { display: flex; align-items: center; gap: 4px; }
.nav-link {
  padding: 7px 14px; border-radius: 8px;
  font-size: 0.85rem; font-weight: 500;
  color: var(--text2); text-decoration: none;
  transition: all .15s;
}
.nav-link:hover { color: var(--text); background: rgba(99,102,241,.08); }
.nav-link.active { color: var(--text); background: rgba(99,102,241,.1); }
.nav-link.btn-primary { color: #fff; padding: 7px 18px; }
.nav-link.btn-primary:hover { color: #fff; }

.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.online  { background: var(--green); animation: pulse 2s infinite; }
.status-dot.offline { background: var(--rose); }

@media (max-width: 600px) {
  .nav-link:not(.btn-primary) { display: none; }
}
</style>
