<template>
  <div class="he-wrap">
    <!-- intro -->
    <div v-if="phase === 'intro'" class="he-card intro-card">
      <div class="he-logo">◈ BGF Human Evaluation</div>
      <h2>Agent Realism Study</h2>
      <p class="he-sub">
        You will see <strong>{{ vignettes.length }} scenarios</strong>. Each shows two AI agents
        (A and B) making decisions in an economic game over several rounds.
      </p>

      <div class="rule-box">
        <div class="rule-title">The Game</div>
        <div class="rule-row"><span class="chip chip-work">work</span> +8 wealth — safe, solo action</div>
        <div class="rule-row"><span class="chip chip-save">save</span> +4 wealth — personal buffer</div>
        <div class="rule-row"><span class="chip chip-coop">cooperate</span> −3 now, +12 shared if others cooperate too</div>
      </div>

      <p class="he-sub">
        For each agent pair, rate how <em>realistic</em> each agent feels on a 1–7 scale
        and choose which one seemed more human-like overall.
      </p>

      <div v-if="loadError" class="he-error">{{ loadError }}</div>

      <div class="he-pid-row">
        <label>Prolific ID (leave blank if not from Prolific)</label>
        <input v-model="prolificPid" class="he-input" placeholder="e.g. 5f3a..." maxlength="64" />
      </div>

      <button class="btn-primary" :disabled="loading" @click="startStudy">
        {{ loading ? 'Loading…' : 'Start Study →' }}
      </button>
    </div>

    <!-- vignette -->
    <div v-else-if="phase === 'vignette'" class="he-vignette-wrap">
      <div class="he-progress-bar">
        <div class="he-progress-fill" :style="{ width: progressPct + '%' }"></div>
      </div>
      <div class="he-progress-label">Scenario {{ currentIdx + 1 }} of {{ vignettes.length }}</div>

      <div class="he-scenario-banner">{{ currentVignette.scenario }}</div>

      <div class="he-agents-grid">
        <div v-for="side in ['A','B']" :key="side" class="he-agent-col">
          <div class="he-agent-header">Agent {{ side }}</div>
          <div class="he-decisions">
            <div
              v-for="d in currentVignette.agents[side].decisions"
              :key="d.round"
              class="he-decision-row"
            >
              <div class="he-round-badge">R{{ d.round }}</div>
              <span :class="['he-action', 'he-action-' + d.action]">{{ d.action }}</span>
              <span class="he-rationale">{{ d.rationale }}</span>
            </div>
          </div>
          <div class="he-wealth-row">
            Final wealth: <strong>{{ currentVignette.agents[side].final_wealth }}</strong>
          </div>

          <div class="he-rating-row">
            <label class="he-rating-label">Realism (1 = robotic, 7 = very human-like)</label>
            <div class="he-stars">
              <button
                v-for="n in 7"
                :key="n"
                :class="['he-star', { active: currentRatings[side] >= n }]"
                @click="currentRatings[side] = n"
              >{{ n }}</button>
            </div>
          </div>
        </div>
      </div>

      <div class="he-prefer-row">
        <label class="he-prefer-label">Which agent felt more realistic overall?</label>
        <div class="he-prefer-btns">
          <button
            v-for="opt in ['A', 'B', 'tie']"
            :key="opt"
            :class="['he-prefer-btn', { selected: currentPreferred === opt }]"
            @click="currentPreferred = opt"
          >
            {{ opt === 'tie' ? 'About the same' : 'Agent ' + opt }}
          </button>
        </div>
      </div>

      <div class="he-comment-row">
        <label>Optional comment (what made it feel realistic or not?)</label>
        <textarea
          v-model="currentComment"
          class="he-comment"
          rows="2"
          maxlength="500"
          placeholder="Optional…"
        ></textarea>
      </div>

      <div v-if="submitError" class="he-error">{{ submitError }}</div>

      <button
        class="btn-primary"
        :disabled="!canSubmit || submitting"
        @click="submitRating"
      >
        {{ submitting ? 'Saving…' : currentIdx + 1 < vignettes.length ? 'Next →' : 'Finish ✓' }}
      </button>
    </div>

    <!-- done -->
    <div v-else-if="phase === 'done'" class="he-card done-card">
      <div class="done-icon">✓</div>
      <h2>Thank you!</h2>
      <p>Your {{ ratings.length }} rating{{ ratings.length !== 1 ? 's' : '' }} have been saved.</p>
      <p class="he-sub">
        Agent B uses ESS survey data to ground its decisions in empirically measured trust,
        risk tolerance, and social norms. Agent A has no such grounding — it relies on the
        LLM's default cooperative bias.
      </p>
      <div class="he-summary">
        <div v-for="r in ratings" :key="r.vignette_id" class="he-summary-row">
          <span class="he-summary-vid">{{ r.vignette_id }}</span>
          <span>A: {{ r.realism_a }}/7</span>
          <span>B: {{ r.realism_b }}/7</span>
          <span class="he-summary-pref">preferred: {{ r.preferred }}</span>
        </div>
      </div>
      <button class="btn-secondary" @click="resetStudy">Start over</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { api } from '../api/index.js'

const phase = ref('intro')
const prolificPid = ref('')
const vignettes = ref([])
const loading = ref(false)
const loadError = ref('')
const currentIdx = ref(0)
const currentRatings = ref({ A: 0, B: 0 })
const currentPreferred = ref('')
const currentComment = ref('')
const submitting = ref(false)
const submitError = ref('')
const ratings = ref([])

const currentVignette = computed(() => vignettes.value[currentIdx.value] || null)
const progressPct = computed(() =>
  vignettes.value.length ? ((currentIdx.value) / vignettes.value.length) * 100 : 0
)
const canSubmit = computed(() =>
  currentRatings.value.A > 0 && currentRatings.value.B > 0 && currentPreferred.value !== ''
)

async function startStudy() {
  loading.value = true
  loadError.value = ''
  try {
    const pid = prolificPid.value.trim()
    const res = await api.humanEvalScenarios(pid)
    vignettes.value = res.data.vignettes
    phase.value = 'vignette'
  } catch (e) {
    loadError.value = e?.response?.data?.error || e.message || 'Failed to load scenarios'
  } finally {
    loading.value = false
  }
}

async function submitRating() {
  submitting.value = true
  submitError.value = ''
  try {
    const record = {
      vignette_id: currentVignette.value.id,
      prolific_pid: prolificPid.value.trim() || 'anonymous',
      realism_a: currentRatings.value.A,
      realism_b: currentRatings.value.B,
      preferred: currentPreferred.value,
      comment: currentComment.value,
    }
    await api.humanEvalRating(record)
    ratings.value.push(record)

    if (currentIdx.value + 1 < vignettes.value.length) {
      currentIdx.value++
      currentRatings.value = { A: 0, B: 0 }
      currentPreferred.value = ''
      currentComment.value = ''
    } else {
      phase.value = 'done'
    }
  } catch (e) {
    submitError.value = e?.response?.data?.error || e.message || 'Save failed — please retry'
  } finally {
    submitting.value = false
  }
}

function resetStudy() {
  phase.value = 'intro'
  currentIdx.value = 0
  currentRatings.value = { A: 0, B: 0 }
  currentPreferred.value = ''
  currentComment.value = ''
  ratings.value = []
  submitError.value = ''
}
</script>

<style scoped>
.he-wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1rem;
  min-height: 80vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* ── Cards ─────────────────────────────────────────────────── */
.he-card {
  background: var(--surface1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2.5rem 2rem;
  width: 100%;
  max-width: 560px;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

.he-logo { font-size: .75rem; letter-spacing: .12em; color: var(--blue1); text-transform: uppercase; }

h2 { margin: 0; font-size: 1.4rem; color: var(--text0); }

.he-sub { margin: 0; font-size: .9rem; color: var(--text2); line-height: 1.6; }

/* ── Rule box ──────────────────────────────────────────────── */
.rule-box {
  background: var(--surface2);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  display: flex;
  flex-direction: column;
  gap: .5rem;
}
.rule-title { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text3); margin-bottom: .2rem; }
.rule-row { display: flex; align-items: center; gap: .6rem; font-size: .88rem; color: var(--text1); }

.chip { border-radius: 4px; padding: 1px 7px; font-size: .78rem; font-weight: 600; }
.chip-work { background: #1a2e1a; color: #4caf50; }
.chip-save { background: #1a2540; color: #5b8dee; }
.chip-coop { background: #2a1a2e; color: #b06fd8; }

/* ── PID row ───────────────────────────────────────────────── */
.he-pid-row { display: flex; flex-direction: column; gap: .4rem; font-size: .88rem; color: var(--text2); }
.he-input {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text0);
  padding: .5rem .75rem;
  font-size: .9rem;
  width: 100%;
  box-sizing: border-box;
}
.he-input:focus { outline: none; border-color: var(--blue1); }

/* ── Progress ──────────────────────────────────────────────── */
.he-vignette-wrap { width: 100%; display: flex; flex-direction: column; gap: 1.2rem; }

.he-progress-bar {
  height: 3px;
  background: var(--surface2);
  border-radius: 2px;
  overflow: hidden;
}
.he-progress-fill {
  height: 100%;
  background: var(--blue1);
  transition: width .3s ease;
}
.he-progress-label { font-size: .78rem; color: var(--text3); }

.he-scenario-banner {
  background: var(--surface1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .75rem 1rem;
  font-size: .9rem;
  color: var(--text1);
}

/* ── Agents grid ───────────────────────────────────────────── */
.he-agents-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
@media (max-width: 600px) { .he-agents-grid { grid-template-columns: 1fr; } }

.he-agent-col {
  background: var(--surface1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: .8rem;
}
.he-agent-header {
  font-size: .8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--blue1);
}

.he-decisions { display: flex; flex-direction: column; gap: .45rem; }
.he-decision-row {
  display: flex;
  align-items: flex-start;
  gap: .5rem;
  font-size: .82rem;
}
.he-round-badge {
  min-width: 24px;
  text-align: center;
  font-size: .7rem;
  color: var(--text3);
  padding-top: 1px;
}
.he-action {
  border-radius: 4px;
  padding: 1px 6px;
  font-size: .72rem;
  font-weight: 600;
  white-space: nowrap;
}
.he-action-work  { background: #1a2e1a; color: #4caf50; }
.he-action-save  { background: #1a2540; color: #5b8dee; }
.he-action-cooperate { background: #2a1a2e; color: #b06fd8; }
.he-action-steal { background: #2e1a1a; color: #e57373; }

.he-rationale { color: var(--text2); line-height: 1.4; }

.he-wealth-row { font-size: .82rem; color: var(--text2); border-top: 1px solid var(--border); padding-top: .6rem; }

/* ── Rating ────────────────────────────────────────────────── */
.he-rating-row { display: flex; flex-direction: column; gap: .4rem; }
.he-rating-label { font-size: .78rem; color: var(--text3); }

.he-stars { display: flex; gap: 4px; }
.he-star {
  width: 30px;
  height: 30px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text2);
  font-size: .82rem;
  cursor: pointer;
  transition: background .15s, color .15s, border-color .15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.he-star.active {
  background: var(--blue1);
  color: #fff;
  border-color: var(--blue1);
}
.he-star:hover { border-color: var(--blue1); }

/* ── Preference ────────────────────────────────────────────── */
.he-prefer-row {
  display: flex;
  flex-direction: column;
  gap: .6rem;
}
.he-prefer-label { font-size: .85rem; color: var(--text2); }
.he-prefer-btns { display: flex; gap: .6rem; flex-wrap: wrap; }
.he-prefer-btn {
  padding: .45rem 1rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text1);
  font-size: .88rem;
  cursor: pointer;
  transition: background .15s, border-color .15s;
}
.he-prefer-btn.selected {
  background: var(--blue1);
  color: #fff;
  border-color: var(--blue1);
}

/* ── Comment ───────────────────────────────────────────────── */
.he-comment-row { display: flex; flex-direction: column; gap: .4rem; font-size: .85rem; color: var(--text2); }
.he-comment {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text0);
  padding: .5rem .75rem;
  font-size: .88rem;
  resize: vertical;
  font-family: inherit;
  width: 100%;
  box-sizing: border-box;
}
.he-comment:focus { outline: none; border-color: var(--blue1); }

/* ── Error ─────────────────────────────────────────────────── */
.he-error {
  background: #2e1a1a;
  border: 1px solid #7b3333;
  border-radius: 6px;
  color: #e57373;
  padding: .6rem .9rem;
  font-size: .85rem;
}

/* ── Done ──────────────────────────────────────────────────── */
.done-card { align-items: center; text-align: center; }
.done-icon { font-size: 2.5rem; color: #4caf50; }

.he-summary { display: flex; flex-direction: column; gap: .4rem; width: 100%; }
.he-summary-row {
  display: flex;
  gap: 1rem;
  justify-content: center;
  font-size: .84rem;
  color: var(--text2);
}
.he-summary-vid { font-weight: 600; color: var(--text1); }
.he-summary-pref { color: var(--blue1); }

/* ── Buttons ───────────────────────────────────────────────── */
.btn-primary {
  padding: .65rem 1.4rem;
  border-radius: 8px;
  border: none;
  background: var(--blue1);
  color: #fff;
  font-size: .92rem;
  font-weight: 600;
  cursor: pointer;
  align-self: flex-start;
  transition: opacity .15s;
}
.btn-primary:disabled { opacity: .45; cursor: not-allowed; }
.btn-primary:not(:disabled):hover { opacity: .85; }

.btn-secondary {
  padding: .55rem 1.2rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text1);
  font-size: .88rem;
  cursor: pointer;
}
.btn-secondary:hover { border-color: var(--blue1); }
</style>
