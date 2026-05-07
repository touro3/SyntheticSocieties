<template>
  <span class="status-badge" :class="cls">
    <span class="sb-dot"></span>
    {{ label }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({ status: String })

const cls = computed(() => {
  switch (props.status) {
    case 'complete':
    case 'completed': return 'sb-green'
    case 'running':   return 'sb-blue'
    case 'failed':    return 'sb-rose'
    case 'pending':   return 'sb-amber'
    default:          return 'sb-teal'
  }
})
const label = computed(() => props.status || 'unknown')
</script>

<style scoped>
.status-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 99px;
  font-size: .68rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
}
.sb-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

.sb-green { background: rgba(16,185,129,.1); color: #10b981; border: 1px solid rgba(16,185,129,.2); }
.sb-green .sb-dot { background: #10b981; }

.sb-blue { background: rgba(99,102,241,.1); color: #818cf8; border: 1px solid rgba(99,102,241,.2); }
.sb-blue .sb-dot { background: #818cf8; animation: pulse 2s infinite; }

.sb-rose { background: rgba(244,63,94,.1); color: #f43f5e; border: 1px solid rgba(244,63,94,.2); }
.sb-rose .sb-dot { background: #f43f5e; }

.sb-amber { background: rgba(245,158,11,.1); color: #f59e0b; border: 1px solid rgba(245,158,11,.2); }
.sb-amber .sb-dot { background: #f59e0b; animation: pulse 2s infinite; }

.sb-teal { background: rgba(20,184,166,.1); color: #14b8a6; border: 1px solid rgba(20,184,166,.2); }
.sb-teal .sb-dot { background: #14b8a6; }

@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }
</style>
