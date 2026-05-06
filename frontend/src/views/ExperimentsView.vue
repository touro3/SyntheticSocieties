<template>
  <div class="container">
    <div class="page-header">
      <h1 class="page-title">Experiments</h1>
      <p class="page-sub">All simulation runs tracked in the DuckDB experiment index.</p>
    </div>

    <!-- Filters -->
    <div class="filters card">
      <div class="filter-row">
        <div class="filter-group">
          <label>Policy</label>
          <div class="tab-group">
            <button
              v-for="p in policyFilters" :key="p"
              class="tab" :class="{ active: filter === p }"
              @click="setFilter(p)"
            >{{ p === '' ? 'All' : p }}</button>
          </div>
        </div>
        <div class="filter-group">
          <label>Search</label>
          <input v-model="search" placeholder="Filter by experiment ID…" style="max-width:280px" />
        </div>
        <router-link to="/run" class="btn btn-primary" style="margin-left:auto;white-space:nowrap">+ Run New</router-link>
      </div>
    </div>

    <div v-if="loading" class="empty"><span class="spin">⟳</span> Loading…</div>

    <div v-else-if="filtered.length === 0" class="empty card">
      <div v-if="allExperiments.length === 0">
        No experiments yet.
        <router-link to="/run"> Launch your first simulation →</router-link>
      </div>
      <div v-else>No results match the current filter.</div>
    </div>

    <div v-else>
      <div class="count-row">
        <span class="count">{{ filtered.length }} experiment{{ filtered.length !== 1 ? 's' : '' }}</span>
        <span class="note" v-if="note">{{ note }}</span>
      </div>

      <div class="card table-card">
        <table>
          <thead>
            <tr>
              <th @click="sortBy('experiment_id')" class="sortable">
                Experiment ID <span class="sort-arrow">{{ sortArrow('experiment_id') }}</span>
              </th>
              <th>Policy</th>
              <th @click="sortBy('seed')" class="sortable">
                Seed <span class="sort-arrow">{{ sortArrow('seed') }}</span>
              </th>
              <th @click="sortBy('gini')" class="sortable">
                Gini ↕ <span class="sort-arrow">{{ sortArrow('gini') }}</span>
              </th>
              <th @click="sortBy('wealth_mean')" class="sortable">
                Wealth Mean <span class="sort-arrow">{{ sortArrow('wealth_mean') }}</span>
              </th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="e in paginated" :key="e.experiment_id" class="exp-row">
              <td class="mono exp-id" :title="e.experiment_id">{{ e.experiment_id }}</td>
              <td>
                <span class="badge badge-teal" style="font-size:.7rem">{{ e.policy_type || '—' }}</span>
              </td>
              <td class="mono">{{ e.seed ?? '—' }}</td>
              <td class="mono">
                <span class="gini-pill" :style="giniColor(e.gini)">{{ fmt(e.gini) }}</span>
              </td>
              <td class="mono">{{ fmt(e.wealth_mean) }}</td>
              <td><StatusBadge :status="e.status || 'complete'" /></td>
              <td>
                <div class="row-actions">
                  <router-link :to="`/results/${e.experiment_id}`" class="btn btn-ghost" style="font-size:.76rem;padding:4px 10px">Results</router-link>
                  <router-link :to="`/interact/${e.experiment_id}`" class="btn btn-outline" style="font-size:.76rem;padding:4px 10px">Interact</router-link>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Pagination -->
      <div class="pagination" v-if="totalPages > 1">
        <button class="btn btn-outline" :disabled="page === 1" @click="page--">← Prev</button>
        <span class="page-info">{{ page }} / {{ totalPages }}</span>
        <button class="btn btn-outline" :disabled="page === totalPages" @click="page++">Next →</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../api/index.js'
import StatusBadge from '../components/StatusBadge.vue'

const PAGE_SIZE = 20

const allExperiments = ref([])
const loading  = ref(true)
const note     = ref('')
const filter   = ref('')
const search   = ref('')
const sortKey  = ref('experiment_id')
const sortDir  = ref(-1)
const page     = ref(1)

const policyFilters = ['', 'mock', 'random', 'rule_based', 'template', 'llm', 'generative_agents']

const fmt = v => (v !== undefined && v !== null) ? Number(v).toFixed(3) : '—'

function giniColor(g) {
  if (g === undefined || g === null) return {}
  const h = Math.round((1 - g) * 120)
  return { color: `hsl(${h},70%,60%)` }
}

const filtered = computed(() => {
  let list = allExperiments.value
  if (filter.value) list = list.filter(e => e.policy_type === filter.value)
  if (search.value) {
    const q = search.value.toLowerCase()
    list = list.filter(e => e.experiment_id?.toLowerCase().includes(q))
  }
  return [...list].sort((a, b) => {
    const av = a[sortKey.value] ?? ''
    const bv = b[sortKey.value] ?? ''
    return sortDir.value * (av < bv ? -1 : av > bv ? 1 : 0)
  })
})

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / PAGE_SIZE)))
const paginated  = computed(() => {
  const s = (page.value - 1) * PAGE_SIZE
  return filtered.value.slice(s, s + PAGE_SIZE)
})

function setFilter(p) { filter.value = p; page.value = 1 }

function sortBy(key) {
  if (sortKey.value === key) sortDir.value *= -1
  else { sortKey.value = key; sortDir.value = -1 }
  page.value = 1
}

function sortArrow(key) {
  if (sortKey.value !== key) return ''
  return sortDir.value === 1 ? '↑' : '↓'
}

onMounted(async () => {
  try {
    const r = await api.experiments()
    allExperiments.value = r.data.experiments || []
    note.value = r.data.note || ''
  } catch {
    note.value = 'Failed to load experiments.'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.page-header { margin-bottom: 24px; }
.page-title  { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
.page-sub    { color: var(--text2); font-size: .95rem; }

.filters { padding: 16px 20px; margin-bottom: 20px; }
.filter-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
.filter-group { display: flex; flex-direction: column; gap: 6px; }
.filter-group label { font-size: .76rem; color: var(--text3); font-weight: 500; text-transform: uppercase; letter-spacing: .04em; }
.tab-group { display: flex; gap: 4px; flex-wrap: wrap; }
.tab {
  padding: 5px 12px; border-radius: 7px; border: 1px solid var(--border);
  background: transparent; color: var(--text3); font-size: .78rem; transition: all .15s;
}
.tab:hover { color: var(--text); border-color: var(--blue); }
.tab.active { background: var(--blue); color: #fff; border-color: var(--blue); }

.count-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.count { font-size: .85rem; color: var(--text2); }
.note  { font-size: .76rem; color: var(--text3); }

.table-card { padding: 0; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  padding: 11px 14px; text-align: left;
  font-size: .72rem; font-weight: 600; letter-spacing: .05em;
  color: var(--text3); text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--text); }
.sort-arrow { color: var(--blue); }
td { padding: 11px 14px; font-size: .83rem; border-bottom: 1px solid rgba(42,48,80,.5); }
.exp-row:hover td { background: rgba(99,102,241,.04); }
.exp-id { color: var(--text2); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gini-pill { font-weight: 600; }
.row-actions { display: flex; gap: 6px; }

.pagination { display: flex; justify-content: center; align-items: center; gap: 14px; margin-top: 20px; }
.page-info { font-size: .84rem; color: var(--text2); }
</style>
