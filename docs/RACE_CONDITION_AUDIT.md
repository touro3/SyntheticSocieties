# Race Condition & Concurrency Audit

**Date:** 2026-05-28
**Scope:** All parallel/distributed code paths in SyntheticSocieties (Flask API threads, kernel signal handlers, file-based IPC, batched LLM backend, DuckDB/parquet tracker, crash-recovery scans).
**Method:** Static review of every concurrency surface (`threading.Thread`, `Lock`, `asyncio`, `multiprocessing`, shared module globals, multi-writer files).

---

## Severity legend

- **High** — silent data loss or corruption under normal operation.
- **Medium** — data loss only under crash / hard-kill; or inefficiency without correctness loss.
- **Low** — race exists but mitigated by retry, single-writer invariant, or astronomical improbability.

---

## High-severity findings

### H1. `tracker/utils/tracker.py:56–72` — Parquet tracker read-modify-write has no lock

Two simulations finishing simultaneously both read `experiment_index.parquet`, both compute `concat`, and both call `write_parquet`. The second write clobbers the first; one experiment's record is silently dropped.

```python
if tracker_path.exists():
    existing = pl.read_parquet(tracker_path)        # racy read
    ...
    updated = pl.concat([existing, new_df], ...)
updated.write_parquet(tracker_path)                  # racy write — last writer wins
```

**Impact:** With the current n=500 sweep (4 concurrent runs), any two finishing within the same RMW window lose one registry entry. Downstream `tracker.analytics` queries and the `/experiments` API endpoint will not see the lost run.

**Fix:** Acquire an `fcntl.flock` on the parquet file (or a sidecar `.lock`) around the read-write block. Alternatively switch the registry to append-only JSONL and compact on read.

---

## Medium-severity findings

### M1. `api/app.py:162–166, 410–412` — `_OAI_CLIENTS` dict mutated without lock

`_DESIGN_CACHE` is protected by `_DESIGN_CACHE_LOCK`. The sibling `_OAI_CLIENTS` cache is not. Concurrent requests can both miss the cache, both construct an `openai.OpenAI` client, and both insert — wasting a connection and possibly triggering the LRU eviction (`while len(_OAI_CLIENTS) > _OAI_CLIENTS_MAX`) during another thread's iteration.

```python
_OAI_CLIENTS: dict[str, Any] = {}
def _oai_clients_put(prefix, client):
    _OAI_CLIENTS[prefix] = client                    # no lock
    while len(_OAI_CLIENTS) > _OAI_CLIENTS_MAX:
        _OAI_CLIENTS.pop(next(iter(_OAI_CLIENTS)))  # iter + pop, racy
```

**Impact:** Mostly wasted clients. The LRU eviction loop iterating while another thread mutates can in theory raise `RuntimeError: dictionary changed size during iteration` on CPython.

**Fix:** Add `_OAI_CLIENTS_LOCK = threading.Lock()` and wrap `_oai_clients_get` / `_oai_clients_put`. Mirror the existing `_DESIGN_CACHE_LOCK` pattern.

### M2. `api/app.py:243–244` — `interview_responses.jsonl` appended without fsync

Append-mode write of new interview responses; no `flush()`/`fsync()`. On SIGKILL/OOM the last few records can be lost. The anchor pipeline tolerates missing entries, but data loss is still possible.

**Fix:** `fh.write(...); fh.flush(); os.fsync(fh.fileno())` — or accept the loss and document it.

### M3. `simulation/crash_recovery.py:202–208` — `run_state.json` is not atomically written

Direct `path.write_text(...)` — a concurrent reader (the `/status` endpoint, `scan_incomplete_runs`) can observe a truncated file if the writer crashes mid-call. Risk is low because the file is small and `write_text` buffers, but the surrounding `heartbeat.json`/`checkpoint.json` writes correctly use `_atomic_write_json` with `os.replace`. Use the same helper here for consistency.

**Fix:** Route through `simulation.kernel._atomic_write_json` (or copy the pattern: tmp file + `os.replace`).

### M4. `simulation/ipc.py:150` — IPC response not atomic; client busy-waits on `JSONDecodeError`

```python
resp_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
```

Client at `ipc.py:328–339` defensively catches `JSONDecodeError` and retries on next poll, so correctness is preserved — but every collision burns a poll cycle. Trivial fix.

**Fix:** Write to `resp_path.with_suffix(".tmp")` and `os.replace(tmp, resp_path)`.

---

## Low-severity findings

### L1. `simulation/ipc.py:313–321` — IPC command UUID is truncated to 16 hex chars

`request_id = uuid.uuid4().hex[:16]` gives 64 bits of entropy. Collision is astronomically unlikely for human-scale use, but a command file is overwritten without check, so a collision would silently lose one request. Pair with M4's atomic-rename fix and additionally check `cmd_path.exists()` before write, or use the full UUID.

### L2. `simulation/crash_recovery.py:315–346` — `scan_incomplete_runs()` races with live writers

The scan is read-only and used only for advisory status, so the inconsistency (a run shown at round N when it's already at N+1) is benign by contract. Document this if the function is ever repurposed for write decisions.

---

## Surfaces explicitly verified safe

| Surface | File | Why safe |
|--------|------|----------|
| Heartbeat / checkpoint writes | `simulation/kernel.py:30–40, 642` | Uses `_atomic_write_json` → POSIX `os.replace` |
| Resume path for `events.jsonl` | `scripts/run_config_simulation.py:614` | Passes `overwrite=(resume_exp_id is None)`; resume always appends |
| IPC injection queue | `simulation/ipc.py:95, 232, 500` | Guarded by `self._lock` (`threading.Lock`) |
| Batched GPU backend | `decision/fast_batched_backend.py` | Single kernel owns the backend; no cross-thread access |
| `_DESIGN_CACHE` | `api/app.py` | Guarded by `_DESIGN_CACHE_LOCK` |

No `multiprocessing.Pool`, no `asyncio`, no `concurrent.futures` is used inside the simulation loop. The Flask layer relies on the WSGI server's thread/worker model; `create_app()` mutates only the module globals listed above.

---

## Investigation: the size=0 `events.jsonl` files

We observed `mx_A_n500_s6` and `mx_B_n500_s6` with `events.jsonl` truncated to 0 bytes (mtime 2026-05-26 17:51). Audit conclusion:

- The kernel resume path (`scripts/run_config_simulation.py:614`) correctly threads `overwrite=False` when `--resume` is supplied, so the `EventLogger` opens in append mode. **The documented resume code is not the cause.**
- `_atomic_write_json` is used only for heartbeat/checkpoint/run_state, never for events.
- No code path opens `events.jsonl` in `"w"` mode on resume.

Most likely root cause for the observed truncation: a **bare** (non-resume) launch was issued at some point against an existing exp_id (e.g. `run_config_simulation.py --config ... --experiment-id mx_A_n500_s6` without `--resume`), which constructs `EventLogger(..., overwrite=True)` and truncates the file before the kernel has computed a round to write. Recommend adding a guard in `EventLogger` (or in `run_config_simulation.py`) that refuses to truncate an existing non-empty `events.jsonl` unless `--force` is also passed.

---

## Recommended fix priority

1. **H1** — tracker parquet lock (immediate; affects every multi-run sweep including the active n=500 set).
2. **M3** — atomic `run_state.json` write (5-line change; eliminates a class of status-endpoint flakes).
3. **M1** — `_OAI_CLIENTS` lock (cheap correctness fix).
4. **Guard against accidental `events.jsonl` truncation** — add `--force` requirement when `experiments/<id>/events.jsonl` is non-empty.
5. M4, M2, L1 — defer; mitigated or marginal.
