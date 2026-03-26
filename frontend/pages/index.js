import { useEffect, useRef, useState, useCallback } from 'react'

const API = 'http://localhost:8000'

export default function Home() {
  const [state, setState] = useState(null)
  const [running, setRunning] = useState(false)
  const [aiEnabled, setAiEnabled] = useState(false)
  const [videos, setVideos] = useState([])
  const [selectedVideo, setSelectedVideo] = useState('test2.mp4')
  const [conf, setConf] = useState(0.4)
  const [fps, setFps] = useState(10)
  const [frameUrl, setFrameUrl] = useState(null)
  const eventsEndRef = useRef(null)

  // Fetch available videos on mount
  useEffect(() => {
    fetch(`${API}/videos`)
      .then(r => r.json())
      .then(d => { if (d.videos?.length) { setVideos(d.videos); setSelectedVideo(d.videos[0]) } })
      .catch(() => {})
  }, [])

  // Poll state
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      fetch(`${API}/state`)
        .then(r => r.json())
        .then(d => {
          setState(d)
          setRunning(d.video_running)
          setAiEnabled(d.ai_enabled)
        })
        .catch(() => {})
    }, 500)
    return () => clearInterval(id)
  }, [running])

  // Poll frame
  useEffect(() => {
    if (!running) return
    let cancelled = false
    const poll = async () => {
      while (!cancelled) {
        try {
          const r = await fetch(`${API}/frame`)
          if (r.ok && r.status !== 204) {
            const blob = await r.blob()
            setFrameUrl(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(blob) })
          }
        } catch {}
        await new Promise(r => setTimeout(r, 200))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [running])

  // Auto-scroll events
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state?.events?.length])

  const handleStart = async () => {
    try {
      const r = await fetch(`${API}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: selectedVideo, conf, fps })
      })
      if (r.ok) setRunning(true)
    } catch {}
  }

  const handleStop = async () => {
    await fetch(`${API}/stop`, { method: 'POST' }).catch(() => {})
    setRunning(false)
  }

  const handleReset = async () => {
    await fetch(`${API}/reset`, { method: 'POST' }).catch(() => {})
    setRunning(false)
    setState(null)
    setFrameUrl(null)
  }

  const handleToggleAi = async () => {
    try {
      const r = await fetch(`${API}/toggle-ai`, { method: 'POST' })
      const d = await r.json()
      setAiEnabled(d.ai_enabled)
    } catch {}
  }

  const active = state?.entities?.active || []
  const gone = state?.entities?.gone || []
  const events = state?.events || []
  const summary = state?.summary || {}

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: running ? '#22c55e' : '#6b7280' }} />
          <h1 className="text-lg font-semibold tracking-tight">CCTV Live Recognition Demo</h1>
          {running && state?.fps > 0 && (
            <span className="text-xs text-gray-500 ml-2">{state.fps} FPS</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Video selector */}
          <select
            value={selectedVideo}
            onChange={e => setSelectedVideo(e.target.value)}
            disabled={running}
            className="bg-gray-800 text-sm rounded px-2 py-1.5 border border-gray-700 disabled:opacity-50"
          >
            {videos.map(v => <option key={v} value={v}>{v}</option>)}
          </select>

          {/* Confidence */}
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <span>Conf:</span>
            <input type="number" min="0.1" max="0.9" step="0.05" value={conf}
              onChange={e => setConf(parseFloat(e.target.value))}
              disabled={running}
              className="w-14 bg-gray-800 rounded px-1.5 py-1 border border-gray-700 text-sm disabled:opacity-50" />
          </div>

          {/* FPS */}
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <span>FPS:</span>
            <input type="number" min="1" max="30" value={fps}
              onChange={e => setFps(parseInt(e.target.value))}
              disabled={running}
              className="w-12 bg-gray-800 rounded px-1.5 py-1 border border-gray-700 text-sm disabled:opacity-50" />
          </div>

          {/* Control buttons */}
          {!running ? (
            <button onClick={handleStart} className="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-sm font-medium transition">
              Start
            </button>
          ) : (
            <button onClick={handleStop} className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-sm font-medium transition">
              Stop
            </button>
          )}
          <button onClick={handleReset} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition">
            Reset
          </button>
          <button onClick={handleToggleAi}
            className={`px-3 py-1.5 rounded text-sm font-medium transition ${aiEnabled ? 'bg-purple-600 hover:bg-purple-500' : 'bg-gray-700 hover:bg-gray-600'}`}>
            AI {aiEnabled ? 'ON' : 'OFF'}
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex h-[calc(100vh-57px)]">
        {/* Left: Video + Events */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Video feed */}
          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden relative">
            {frameUrl ? (
              <img src={frameUrl} alt="Live feed" className="max-w-full max-h-full object-contain" />
            ) : (
              <div className="text-gray-600 text-sm">
                {running ? 'Waiting for frames...' : 'Select a video and click Start'}
              </div>
            )}
          </div>

          {/* Events + Summary bar */}
          <div className="h-56 flex border-t border-gray-800">
            {/* Event log */}
            <div className="flex-1 flex flex-col min-w-0">
              <div className="px-3 py-1.5 bg-gray-900 border-b border-gray-800 flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Live Events</span>
                <span className="text-xs text-gray-600">{events.length} events</span>
              </div>
              <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5 bg-gray-950 font-mono text-xs">
                {events.length === 0 && <div className="text-gray-600 italic">No events yet</div>}
                {events.map((e, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-gray-500 shrink-0">{e.time}</span>
                    <EventIcon type={e.type} />
                    <span className={
                      e.type === 'appeared' ? 'text-green-400' :
                      e.type === 'left' ? 'text-red-400' :
                      e.type === 'reappeared' ? 'text-yellow-400' :
                      e.type === 'ai_analysis' ? 'text-purple-400' :
                      'text-gray-300'
                    }>{e.message}</span>
                  </div>
                ))}
                <div ref={eventsEndRef} />
              </div>
            </div>

            {/* Summary panel */}
            <div className="w-64 border-l border-gray-800 bg-gray-900 p-3 flex flex-col gap-3">
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Summary</span>
              <div className="grid grid-cols-2 gap-2">
                <StatCard label="Active" value={summary.active_count ?? 0} color="text-green-400" />
                <StatCard label="Departed" value={summary.gone_count ?? 0} color="text-red-400" />
                <StatCard label="Total Unique" value={summary.total_unique ?? 0} color="text-blue-400" />
                <StatCard label="Session" value={summary.session_duration_str ?? '0s'} color="text-gray-300" />
              </div>
              {state?.ai_summary && (
                <div className="mt-1 p-2 bg-purple-900/30 rounded text-xs text-purple-300 border border-purple-800/50">
                  <span className="font-semibold">AI:</span> {state.ai_summary}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right sidebar: Tracked entities */}
        <div className="w-72 border-l border-gray-800 flex flex-col bg-gray-900">
          <div className="px-3 py-2 border-b border-gray-800">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Tracked Entities</span>
            <span className="ml-2 text-xs text-gray-600">{active.length + gone.length}</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {active.length === 0 && gone.length === 0 && (
              <div className="text-gray-600 text-xs italic p-2">No entities detected yet</div>
            )}
            {active.map(e => <EntityCard key={e.id} entity={e} />)}
            {gone.length > 0 && active.length > 0 && (
              <div className="text-xs text-gray-600 px-2 pt-2 pb-1 uppercase tracking-wider font-semibold">Departed</div>
            )}
            {gone.map(e => <EntityCard key={e.id} entity={e} />)}
          </div>
        </div>
      </div>
    </div>
  )
}

function EntityCard({ entity }) {
  const isGone = entity.status === 'gone'
  const borderColor = isGone ? 'border-gray-700' : entity.label === 'person' ? 'border-red-500/50' : 'border-orange-500/50'

  return (
    <div className={`flex items-center gap-2 p-2 rounded-lg bg-gray-800/50 border ${borderColor} ${isGone ? 'opacity-50' : ''}`}>
      {entity.thumbnail_b64 ? (
        <img
          src={`data:image/jpeg;base64,${entity.thumbnail_b64}`}
          alt={entity.display_name}
          className="w-10 h-10 rounded object-cover shrink-0"
        />
      ) : (
        <div className="w-10 h-10 rounded bg-gray-700 flex items-center justify-center shrink-0">
          <span className="text-gray-500 text-xs">{entity.label[0].toUpperCase()}</span>
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium truncate">{entity.display_name}</div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className={`w-1.5 h-1.5 rounded-full ${isGone ? 'bg-gray-500' : 'bg-green-500'}`} />
          <span>{isGone ? 'Gone' : `${entity.duration}s`}</span>
          <span className="text-gray-600">|</span>
          <span>{Math.round(entity.best_conf * 100)}%</span>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }) {
  return (
    <div className="bg-gray-800/50 rounded p-2">
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
    </div>
  )
}

function EventIcon({ type }) {
  const icons = {
    appeared: '+',
    left: '-',
    reappeared: '~',
    ai_analysis: '*',
  }
  const colors = {
    appeared: 'text-green-500',
    left: 'text-red-500',
    reappeared: 'text-yellow-500',
    ai_analysis: 'text-purple-500',
  }
  return <span className={`${colors[type] || 'text-gray-500'} font-bold shrink-0`}>{icons[type] || '·'}</span>
}
