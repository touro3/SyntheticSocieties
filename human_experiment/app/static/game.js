/**
 * BGF Human Baseline — Game Client
 * Communicates with human_experiment/server/server.py via fetch().
 */

const API = "http://localhost:5100";

let state = {
  sessionId:   null,
  neighbors:   [],
  selectedNeighbor: null,
  wealth:      50,
  stress:      0.3,
  roundId:     0,
  totalRounds: 10,
  actionLocked: false,
};

// ── Navigation ────────────────────────────────────────────────────────────────

function showScreen(id) {
  document.querySelectorAll(".card, [id^='screen-']").forEach(el => {
    if (el.id && el.id.startsWith("screen-")) el.classList.add("hidden");
  });
  document.getElementById(id).classList.remove("hidden");
}

// ── Survey → Game ─────────────────────────────────────────────────────────────

async function startGame() {
  const trust = parseInt(document.getElementById("trust-slider").value);
  const risk  = parseInt(document.getElementById("risk-slider").value);

  document.getElementById("start-btn").disabled = true;

  try {
    const resp = await fetch(`${API}/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pre_trust: trust, pre_risk: risk }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    state.sessionId   = data.session_id;
    state.neighbors   = data.neighbors;
    state.wealth      = data.initial_wealth;
    state.stress      = data.initial_stress;
    state.totalRounds = data.total_rounds;
    state.roundId     = 0;

    renderNeighbors();
    updateStats();
    showScreen("screen-game");
  } catch (err) {
    alert("Could not connect to server: " + err.message);
    document.getElementById("start-btn").disabled = false;
  }
}

// ── Neighbor selection ────────────────────────────────────────────────────────

function renderNeighbors() {
  const wrap = document.getElementById("neighbors-wrap");
  wrap.innerHTML = "";
  state.neighbors.forEach(n => {
    const btn = document.createElement("button");
    btn.className = "neighbor-btn";
    btn.textContent = n;
    btn.dataset.neighbor = n;
    btn.onclick = () => selectNeighbor(n);
    wrap.appendChild(btn);
  });
}

function selectNeighbor(name) {
  state.selectedNeighbor = name;
  document.querySelectorAll(".neighbor-btn").forEach(b => {
    b.classList.toggle("selected", b.dataset.neighbor === name);
  });
  document.getElementById("coop-btn").disabled = false;
}

// ── Submit action ─────────────────────────────────────────────────────────────

async function submitAction(action) {
  if (state.actionLocked) return;
  if (action === "cooperate" && !state.selectedNeighbor) {
    showFeedback("Select a neighbor first to cooperate.", "neutral");
    return;
  }

  state.actionLocked = true;
  setButtonsEnabled(false);

  try {
    const resp = await fetch(`${API}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        action,
        target: state.selectedNeighbor,
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    state.wealth  = data.wealth;
    state.stress  = data.stress;
    state.roundId = data.round_id;

    // Feedback message
    let msg = "";
    let tone = "neutral";
    if (action === "work") {
      msg = `You worked. Wealth: ${data.wealth_delta > 0 ? "+" : ""}${data.wealth_delta} → ${data.wealth}`;
      tone = "positive";
    } else if (action === "save") {
      msg = `You rested. Stress reduced. Wealth: ${data.wealth}`;
      tone = "positive";
    } else if (action === "cooperate") {
      msg = `You gave 5 to ${state.selectedNeighbor} (they receive 7.5). Wealth: ${data.wealth}`;
      tone = "positive";
    }
    showFeedback(msg, tone);

    updateStats();
    resetNeighborSelection();

    if (data.done) {
      await finishGame();
    } else {
      state.actionLocked = false;
      setButtonsEnabled(true);
    }
  } catch (err) {
    showFeedback("Error: " + err.message, "negative");
    state.actionLocked = false;
    setButtonsEnabled(true);
  }
}

// ── UI updates ────────────────────────────────────────────────────────────────

function updateStats() {
  document.getElementById("round-tag").textContent =
    `Round ${state.roundId + 1} / ${state.totalRounds}`;
  document.getElementById("wealth-val").textContent = state.wealth.toFixed(1);
  document.getElementById("stress-val").textContent = (state.stress * 100).toFixed(0) + "%";

  // Wealth bar: 100 = full (starting 50, possible range 0-200)
  const wPct = Math.min(100, (state.wealth / 100) * 100);
  document.getElementById("wealth-bar").style.width = wPct + "%";
  document.getElementById("stress-bar").style.width = (state.stress * 100) + "%";
}

function showFeedback(msg, tone) {
  const el = document.getElementById("feedback");
  el.textContent = msg;
  el.className = `feedback show ${tone}`;
}

function setButtonsEnabled(enabled) {
  document.querySelectorAll(".action-btn").forEach(b => {
    if (b.id === "coop-btn") {
      b.disabled = !enabled || !state.selectedNeighbor;
    } else {
      b.disabled = !enabled;
    }
  });
}

function resetNeighborSelection() {
  state.selectedNeighbor = null;
  document.querySelectorAll(".neighbor-btn").forEach(b => b.classList.remove("selected"));
  document.getElementById("coop-btn").disabled = true;
}

// ── Game completion ───────────────────────────────────────────────────────────

async function finishGame() {
  try {
    const resp = await fetch(`${API}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const data = await resp.json();

    document.getElementById("completion-code").textContent =
      data.completion_code || "BGF-DONE";

    const finalStats = document.getElementById("final-stats");
    finalStats.innerHTML = `
      <div class="stat-box">
        <div class="label">Final Wealth</div>
        <div class="value">${data.final_wealth?.toFixed(1) ?? state.wealth.toFixed(1)}</div>
      </div>
      <div class="stat-box">
        <div class="label">Cooperation Rate</div>
        <div class="value">${((data.cooperation_rate ?? 0) * 100).toFixed(0)}%</div>
      </div>
      <div class="stat-box">
        <div class="label">Rounds Played</div>
        <div class="value">${state.totalRounds}</div>
      </div>
    `;

    showScreen("screen-complete");
  } catch (err) {
    alert("Game complete! Code: BGF-" + (state.sessionId || "DONE").slice(0, 8).toUpperCase());
    showScreen("screen-complete");
  }
}
