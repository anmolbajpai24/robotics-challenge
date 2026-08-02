// Thin API layer over the FastAPI backend (dashboard/main.py, port 8011).
// Endpoints this frontend knows about:
//   GET /health
//   GET /summary            frozen-protocol table plus counts
//   GET /runs               all runs with provenance
//   GET /videos             list of playable rollout videos (added on Day 19)
//
// The normalizers below are defensive: if the backend's field names differ
// from the guesses here, the UI falls back to showing raw JSON and the fix
// is a one-line mapping change in normalizeSummary / normalizeVideos.

const TIMEOUT_MS = 6000

async function fetchJSON(path) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(path, { signal: ctrl.signal })
    if (!res.ok) return { ok: false, status: res.status, data: null }
    const data = await res.json()
    return { ok: true, status: res.status, data }
  } catch (err) {
    return { ok: false, status: 0, data: null, error: String(err) }
  } finally {
    clearTimeout(timer)
  }
}

export async function getHealth() {
  return fetchJSON('/health')
}

export async function getSummary() {
  return fetchJSON('/summary')
}

export async function getRuns() {
  return fetchJSON('/runs')
}

export async function getVideos() {
  return fetchJSON('/videos')
}

// Try to pull rows of {name, value} out of whatever /summary returns.
// Returns null when the shape is unrecognized, in which case the UI shows
// the raw JSON instead of guessing wrong.
export function normalizeSummary(data) {
  if (!data || typeof data !== 'object') return null

  // Real shape (verified against dashboard/main.py, Day 19):
  // { counts: {...}, frozen_table: [ { policy, success_rate, protocol, ... } ] }
  // success_rate is pc_success, already a percentage: 0.8 means 0.8%, not 80%.
  if (Array.isArray(data.frozen_table) && data.frozen_table.length > 0) {
    const rows = data.frozen_table.map(frozenTableEntryToRow).filter(Boolean)
    if (rows.length > 0) return rows
  }

  // Shape guess 1: { frozen: [ {policy|name, success_rate|success|value, ...} ] }
  const arrayKeys = ['frozen', 'frozen_500', 'table', 'policies', 'rows', 'runs']
  for (const key of arrayKeys) {
    if (Array.isArray(data[key]) && data[key].length > 0) {
      const rows = data[key].map(entryToRow).filter(Boolean)
      if (rows.length > 0) return rows
    }
  }

  // Shape guess 2: top-level array
  if (Array.isArray(data) && data.length > 0) {
    const rows = data.map(entryToRow).filter(Boolean)
    if (rows.length > 0) return rows
  }

  // Shape guess 3: flat object of policy -> number
  const entries = Object.entries(data).filter(([, v]) => typeof v === 'number')
  if (entries.length >= 2) {
    return entries.map(([name, value]) => ({ name, value: formatRate(value) }))
  }

  return null
}

function frozenTableEntryToRow(entry) {
  if (!entry || typeof entry !== 'object') return null
  const name = entry.policy ?? entry.id ?? null
  const rate = entry.success_rate
  if (name == null || typeof rate !== 'number') return null
  return { name: String(name), value: `${rate.toFixed(1)}%`, extra: entry.protocol ?? '' }
}

function entryToRow(entry) {
  if (!entry || typeof entry !== 'object') return null
  const name = entry.policy ?? entry.name ?? entry.id ?? entry.run_id ?? null
  const raw =
    entry.success_rate ?? entry.success ?? entry.pc_success ?? entry.value ?? null
  if (name == null || raw == null) return null
  return { name: String(name), value: formatRate(raw), extra: entry.protocol ?? entry.provenance_source ?? '' }
}

function formatRate(v) {
  if (typeof v !== 'number') return String(v)
  // Rates may arrive as 0.398 or 39.8; show what arrived, no rounding games.
  return v <= 1 ? `${(v * 100).toFixed(1)}%` : `${v}%`
}

export function normalizeVideos(data) {
  if (!data) return null
  const list = Array.isArray(data) ? data : data.videos
  if (!Array.isArray(list) || list.length === 0) return null
  return list
    .map((v, i) => {
      if (typeof v === 'string') return { id: String(i), title: v.split('/').pop(), url: v }
      if (v && typeof v === 'object') {
        const url = v.url ?? v.path ?? v.src ?? null
        if (!url) return null
        return { id: String(v.id ?? i), title: v.title ?? v.name ?? url.split('/').pop(), url }
      }
      return null
    })
    .filter(Boolean)
}
