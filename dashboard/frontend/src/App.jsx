import { useEffect, useState } from 'react'
import { getHealth, getSummary, getVideos, normalizeSummary, normalizeVideos } from './api.js'

// The frozen numbers are committed facts with saved evidence. They render
// even if the API is down, and the live panel below shows what the API
// itself serves so the two can be compared at a glance.

const PUSHT_FROZEN = [
  { name: 'pretrained-diffusion', value: '61.0%', note: 'official checkpoint, migrated' },
  { name: 'diffusion-scratch', value: '39.8%', note: 'mine, 90k steps, one overnight' },
  { name: 'act-pusht', value: '0.8%', note: 'ACT at default settings' }
]

const XARM_FROZEN = [
  { name: 'bc-mlp-blind (day 14)', value: '0.0%', note: 'TD-MPC demos, no cube in state' },
  { name: 'bc-mlp-blind-control (day 16)', value: '4.2%', note: 'oracle demos, cube stripped' },
  { name: 'bc-mlp-sighted (day 16)', value: '99.0%', note: 'oracle demos, cube xyz in state' }
]

const OFF_RULER = [
  { name: 'act-pusht, n_action_steps=8', value: '11%', note: 'inference knob sweep, not the frozen protocol' },
  { name: 'act-aloha-pretrained, transfer cube', value: '7/10', note: '10 episodes, qualitative' },
  { name: 'scripted oracle, xarm lift', value: '20/20', note: 'privileged: reads true cube position. Validates the benchmark' }
]

function Table({ rows }) {
  return (
    <table>
      <thead>
        <tr><th>policy</th><th>success</th><th></th></tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.name}>
            <td className="name">{r.name}</td>
            <td className="num">{r.value}</td>
            <td className="note">{r.note ?? r.extra ?? ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Leaderboard({ apiUp }) {
  const [summary, setSummary] = useState(null)
  const [rawSummary, setRawSummary] = useState(null)
  const [fetched, setFetched] = useState(false)

  useEffect(() => {
    getSummary().then(res => {
      setFetched(true)
      if (res.ok) {
        setRawSummary(res.data)
        setSummary(normalizeSummary(res.data))
      }
    })
  }, [])

  return (
    <>
      <section className="card">
        <h2>PushT <span className="tag frozen">frozen protocol</span></h2>
        <p className="protocol">500 episodes · batch_size=4 · use_async_envs=false · seed 1000 · ±4 pts sampling error</p>
        <Table rows={PUSHT_FROZEN} />
      </section>

      <section className="card">
        <h2>xarm lift <span className="tag frozen">frozen protocol</span></h2>
        <p className="protocol">500 episodes · reset(seed=1000+i) · 300-step cap · success: cube ≥ 15cm above spawn (patched check)</p>
        <Table rows={XARM_FROZEN} />
      </section>

      <section className="card">
        <h2>Off-ruler results <span className="tag offruler">not comparable above</span></h2>
        <p className="protocol">real numbers, different protocols. Listed separately on purpose</p>
        <Table rows={OFF_RULER} />
      </section>

      <section className="card">
        <h2>Live from the API <span className="tag probe">/summary</span></h2>
        <p className="protocol">served by dashboard/main.py, the same files every table above came from</p>
        {!fetched && <p className="note">querying…</p>}
        {fetched && !rawSummary && (
          <div className="banner">
            API not reachable{apiUp ? ' at /summary' : ''}. Start it with uvicorn and reload. The frozen tables above are static copies of committed results.
          </div>
        )}
        {rawSummary && summary && <Table rows={summary} />}
        {rawSummary && !summary && (
          <>
            <div className="banner">
              Response shape not recognized by the normalizer. Raw payload below. Fix: adjust normalizeSummary() in src/api.js to map these fields.
            </div>
            <div className="rawjson">{JSON.stringify(rawSummary, null, 2)}</div>
          </>
        )}
      </section>
    </>
  )
}

function Episodes() {
  const [videos, setVideos] = useState(null)
  const [fetched, setFetched] = useState(false)
  const [current, setCurrent] = useState(null)

  useEffect(() => {
    getVideos().then(res => {
      setFetched(true)
      if (res.ok) {
        const list = normalizeVideos(res.data)
        setVideos(list)
        if (list && list.length > 0) setCurrent(list[0])
      }
    })
  }, [])

  return (
    <section className="card">
      <h2>Episode player</h2>
      {!fetched && <p className="note">querying…</p>}
      {fetched && !videos && (
        <div className="banner">
          No /videos endpoint answering yet. It gets added during integration: a route in dashboard/main.py that lists and serves saved rollout mp4s (read-only, evidence dirs untouched). This player picks them up automatically.
        </div>
      )}
      {videos && current && (
        <div className="player">
          <video key={current.url} controls autoPlay muted loop src={current.url} />
          <div className="vidlist">
            {videos.map(v => (
              <button
                key={v.id}
                className={current && v.id === current.id ? 'playing' : ''}
                onClick={() => setCurrent(v)}
              >
                {v.title}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default function App() {
  const [tab, setTab] = useState('leaderboard')
  const [apiUp, setApiUp] = useState(null)

  useEffect(() => {
    getHealth().then(res => setApiUp(res.ok))
  }, [])

  return (
    <div className="wrap">
      <header className="site">
        <h1>robotics-challenge <span>· eval dashboard</span></h1>
        <div className={'apidot ' + (apiUp === null ? '' : apiUp ? 'on' : 'off')}>
          {apiUp === null ? 'api: checking' : apiUp ? 'api: up' : 'api: down'}
        </div>
      </header>
      <nav className="tabs">
        <button className={tab === 'leaderboard' ? 'active' : ''} onClick={() => setTab('leaderboard')}>
          Leaderboard
        </button>
        <button className={tab === 'episodes' ? 'active' : ''} onClick={() => setTab('episodes')}>
          Episodes
        </button>
      </nav>

      {tab === 'leaderboard' ? <Leaderboard apiUp={apiUp} /> : <Episodes />}

      <footer className="site">
        Built by <a href="https://x.com/anmol_bajpai24" target="_blank" rel="noreferrer">Anmol Bajpai</a>
      </footer>
    </div>
  )
}
