<template>
  <div class="app">
    <NavBar />
    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import NavBar from './components/NavBar.vue'
</script>

<style>
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
  --glow:    0 0 30px rgba(99,102,241,0.15);
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

body::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 600px 600px at 20% 20%, rgba(99,102,241,.06), transparent),
    radial-gradient(ellipse 800px 800px at 80% 80%, rgba(139,92,246,.04), transparent),
    radial-gradient(ellipse 400px 400px at 60% 30%, rgba(20,184,166,.03), transparent);
  pointer-events: none; z-index: 0;
}

a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

button {
  font-family: inherit;
  cursor: pointer;
  border: none;
  outline: none;
}

input, select, textarea {
  font-family: inherit;
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  padding: 9px 13px;
  font-size: 0.88rem;
  width: 100%;
  transition: border-color .15s;
}
input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--blue);
}

.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px;
  position: relative; z-index: 1;
}

.card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color: rgba(99,102,241,.3); box-shadow: var(--glow); }

.btn {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 10px 20px;
  border-radius: 10px;
  font-size: 0.88rem; font-weight: 500;
  transition: all .18s;
}
.btn-primary {
  background: var(--blue); color: #fff;
}
.btn-primary:hover { background: #5558e6; box-shadow: 0 4px 20px rgba(99,102,241,.3); transform: translateY(-1px); }

.btn-outline {
  background: transparent;
  color: var(--text2);
  border: 1px solid var(--border);
}
.btn-outline:hover { border-color: var(--blue); color: var(--text); background: rgba(99,102,241,.06); }

.btn-ghost {
  background: rgba(99,102,241,.08);
  color: var(--blue);
}
.btn-ghost:hover { background: rgba(99,102,241,.15); }

.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 20px;
  font-size: 0.74rem; font-weight: 500; letter-spacing: .04em;
}
.badge-blue   { background: rgba(99,102,241,.12); color: var(--blue); border: 1px solid rgba(99,102,241,.2); }
.badge-green  { background: rgba(16,185,129,.12);  color: var(--green); border: 1px solid rgba(16,185,129,.2); }
.badge-amber  { background: rgba(245,158,11,.12);  color: var(--amber); border: 1px solid rgba(245,158,11,.2); }
.badge-rose   { background: rgba(244,63,94,.12);   color: var(--rose);  border: 1px solid rgba(244,63,94,.2); }
.badge-teal   { background: rgba(20,184,166,.12);  color: var(--teal);  border: 1px solid rgba(20,184,166,.2); }

.mono { font-family: 'JetBrains Mono', monospace; }

.section-title {
  font-size: 1.15rem; font-weight: 600;
  margin-bottom: 16px;
}

.empty {
  text-align: center; padding: 48px 24px;
  color: var(--text3); font-size: 0.9rem;
}

.spin {
  display: inline-block;
  animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}

.page-enter-active, .page-leave-active { transition: opacity .15s; }
.page-enter-from, .page-leave-to { opacity: 0; }
</style>

<style scoped>
.app { display: flex; flex-direction: column; min-height: 100vh; }
.main { flex: 1; padding: 32px 0 64px; }
</style>
