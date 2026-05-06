<template>
  <span class="badge" :class="cls">{{ label }}</span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({ status: String })

const cls = computed(() => {
  switch (props.status) {
    case 'complete':  return 'badge-green'
    case 'running':   return 'badge-blue'
    case 'failed':    return 'badge-rose'
    case 'pending':   return 'badge-amber'
    default:          return 'badge-teal'
  }
})
const label = computed(() => props.status || 'unknown')
</script>
