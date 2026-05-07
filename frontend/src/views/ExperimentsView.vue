<template>
  <div>
    <div class="page-header">
      <div>
        <h1 class="page-title">Experiments</h1>
        <p class="page-sub">All simulation runs tracked in the DuckDB experiment index.</p>
      </div>
      <router-link to="/run" class="btn btn-primary">
        <span>▶</span> New Run
      </router-link>
    </div>

    <!-- Filters bar -->
    <div class="filters card">
      <div class="filter-row">
        <div class="filter-group">
          <label>Policy</label>
          <div class="tab-row">
            <button v-for="p in policyFilters" :key="p"
              class="ftab" :class="{ active: filter === p }"
              @click="setFilter(p)">
              {{ p || 'All' }}
            </button>
          </div>
        </div>
        <div class="search-wrap">
          <input v-model="search" placeholder="Search by experiment ID…" class="search-input" />
        </div>
      </div>
    </div>

    <!-- Count + note -->
    <div class="count-bar" v-if="!loading">
      <span class="count">{{ filtered.length }} experiment{{ filtered.length !== 1 ? 's' : '' }}</span>
      <span v-if="note" class="note">{{ note }}</span>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="empty card"><span class="spin">⟳</span> Loading…</div>

    <!-- Empty -->
    <div v-else-if="!filtered.length" class="empty card">
      <div v-if="!allExperiments.length">
        No runs yet. <router-link to="/run">Launch one →</router-link>
      </div>
      <div v-else>No experiments match the current filter.</div>
    </div>

    <!-- Table -->
    <div v-else class="card table-card">
      <table>
        <thead>
          <tr>
            <th class="sortable" @click="sortBy('experiment_id')">
              Experiment <span class="arrow">{{ sortArrow('experiment_id') }}</span>
            </th>
            <th>Policy</th>
            <th class="sortable" @click="sortBy('seed')">
              Seed <span class="arrow">{{ sortArrow('seed') }}</span>
            </th>
            <th class="sortable" @click="sortBy('gini')">
              Gini <span class="arrow">{{ sortArrow('gini') }}</span>
            </th>
            <th class="sortable" @click="sortBy('wealth_mean')">
              Wealth <span class="arrow">{{ sortArrow('wealth_mean') }}</span>
            </th>
            <th>Status</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="e in paginated" :key="e.experiment_id"
            class="exp-row" @click="$router.push(`/results/${e.experiment_id}`)">
            <td class="mono exp-id" :title="e.experiment_id">{{ e.experiment_id }}</td>
            <td>
              <span class="badge badge-teal" style="font-size:.67rem">{{ e.policy_type || '—' }}</span>
            </td>
            <td class="mono dim">{{ e.seed ?? '—' }}</td>
            <td class="mono">
              <span :style="giniColor(e.gini)" style="font-weight:600">{{ fmt(e.gini) }}</span>
            </td>
            <td class="mono">{{ fmt(e.wealth_mean) }}</td>
            <td><StatusBadge :status="e.status || 'complete'" /></td>
            <td @click.stop>
              <div class="row-acts">
                <router-link :to="`/results/${e.experiment_id}`"
                  class="btn btn-ghost btn-sm">Results</router-link>
                <router-link :to="`/monitor/${e.experiment_id}`"
                  class="btn btn-outline btn-sm">Monitor</router-link>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div class="pagination" v-if="totalPages > 1">
      <button class="btn btn-outline btn-sm" :disabled="page === 1" @click="page--">← Prev</button>
      <span class="page-info">{{ page }} / {{ totalPages }}</span>
      <button class="btn btn-outline btn-sm" :disabled="page === totalPages" @click="page++">Next →</button>
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

const fmt      = v => (v != null) ? Number(v).toFixed(3) : '—'
const giniColor = g => g != null ? { color: `hsl(${Math.round((1-g)*120)},60%,58%)` } : {}

const filtered = computed(() => {
  let list = allExperiments.value
  if (filter.value) list = list.filter(e => e.policy_type === filter.value)
  if (search.value) {
    const q = search.value.toLowerCase()
    list = list.filter(e => e.experiment_id?.toLowerCase().includes(q))
  }
  return [...list].sort((a,b) => {
    const av = a[sortKey.value] ?? '', bv = b[sortKey.value] ?? ''
    return sortDir.value * (av < bv ? -1 : av > bv ? 1 : 0)
  })
})

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / PAGE_SIZE)))
const paginated  = computed(() => filtered.value.slice((page.value-1)*PAGE_SIZE, page.value*PAGE_SIZE))

function setFilter(p) { filter.value = p; page.value = 1 }
function sortBy(key) {
  if (sortKey.value === key) sortDir.value *= -1
  else { sortKey.value = key; sortDir.value = -1 }
  page.value = 1
}
function sortArrow(key) {
  if (sortKey.value !== key) return ''
  return sortDir.value === 1 ? ' ↑' : ' ↓'
}

onMounted(async () => {
  try {
    const r = await api.experiments()
    allExperiments.value = r.data.experiments || []
    note.value = r.data.note || ''
  } catch {
    note.value = 'Failed to load experiments.'
  } finally { loading.value = false }
})
</script>

<style scoped>
.page-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  flex-wrap: wrap; gap: 16px; margin-bottom: 22px;
}

/* ── Filters ────────────────────────────────────────────────────── */
.filters { padding: 14px 18px; margin-bottom: 16px; }
.filter-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.filter-group { display: flex; flex-direction: column; gap: 6px; }
.filter-group label { font-size: .72rem; color: var(--text3); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }

.tab-row { display: flex; gap: 4px; flex-wrap: wrap; }
.ftab {
  padding: 5px 11px; border-radius: 7px;
  border: 1px solid var(--border); background: transparent;
  font-size: .76rem; color: var(--text3);
  transition: all .15s; cursor: pointer;
}
.ftab:hover  { color: var(--text); border-color: var(--blue2); }
.ftab.active { background: var(--blue); color: #fff; border-color: var(--blue); }

.search-wrap { flex: 1; min-width: 200px; max-width: 280px; }
.search-input { width: 100%; }

.count-bar {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px;
}
.count { font-size: .82rem; color: var(--text2); }
.note  { font-size: .74rem; color: var(--text3); }

/* ── Table ──────────────────────────────────────────────────────── */
.table-card { padding: 0; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  padding: 11px 14px; text-align: left;
  font-size: .7rem; font-weight: 600;
  letter-spacing: .05em; color: var(--text3); text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--text2); }
.arrow { color: var(--blue2); }
td {
  padding: 10px 14px; font-size: .82rem;
  border-bottom: 1px solid rgba(42,48,80,.4);
  vertical-align: middle;
}
.exp-row { cursor: pointer; transition: background .12s; }
.exp-row:hover td { background: rgba(99,102,241,.04); }
.exp-id { color: var(--text2); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dim { color: var(--text3); }
.row-acts { display: flex; gap: 5px; }

/* ── Pagination ─────────────────────────────────────────────────── */
.pagination { display: flex; justify-content: center; align-items: center; gap: 14px; margin-top: 18px; }
.page-info { font-size: .82rem; color: var(--text2); }
</style>
