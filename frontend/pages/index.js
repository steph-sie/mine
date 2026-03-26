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

  useEffect(() => {
    fetch(`${API}/videos`)
      .then(r => r.json())
      .then(d => { if (d.videos?.length) { setVideos(d.videos); setSelectedVideo(d.videos[0]) } })
      .catch(() => {})
  }, [])

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
  const radar = state?.radar || {}

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
          {/* Radar status indicator */}
          <div className="flex items-center gap-1.5 ml-3 px-2 py-0.5 rounded-full text-xs"
            style={{ background: radar.connected ? 'rgba(59,130,246,0.15)' : 'rgba(107,114,128,0.15)' }}>
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: radar.connected ? '#3b82f6' : '#6b7280' }} />
            <span className={radar.connected ? 'text-blue-400' : 'text-gray-500'}>
              Radar {radar.connected ? `(${radar.track_count || 0})` : 'Off'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select value={selectedVideo} onChange={e => setSelectedVideo(e.target.value)}
            disabled={running} className="bg-gray-800 text-sm rounded px-2 py-1.5 border border-gray-700 disabled:opacity-50">
            {videos.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <span>Conf:</span>
            <input type="number" min="0.1" max="0.9" step="0.05" value={conf}
              onChange={e => setConf(parseFloat(e.target.value))} disabled={running}
              className="w-14 bg-gray-800 rounded px-1.5 py-1 border border-gray-700 text-sm disabled:opacity-50" />
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <span>FPS:</span>
            <input type="number" min="1" max="30" value={fps}
              onChange={e => setFps(parseInt(e.target.value))} disabled={running}
              className="w-12 bg-gray-800 rounded px-1.5 py-1 border border-gray-700 text-sm disabled:opacity-50" />
          </div>
          {!running ? (
            <button onClick={handleStart} className="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-sm font-medium transition">Start</button>
          ) : (
            <button onClick={handleStop} className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-sm font-medium transition">Stop</button>
          )}
          <button onClick={handleReset} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition">Reset</button>
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
            {/* Radar mini-map overlay */}
            {radar.connected && radar.tracks?.length > 0 && (
              <div className="absolute bottom-3 right-3">
                <RadarMiniMap tracks={radar.tracks} />
              </div>
            )}
          </div>

          {/* Events + Summary bar */}
          <div className="h-56 flex border-t border-gray-800">
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
                    <span className={eventColor(e.type)}>{e.message}</span>
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
                <StatCard label="Re-ID'd" value={summary.reidentified_count ?? 0} color="text-cyan-400" />
              </div>
              {/* Radar fusion stats */}
              {radar.connected && (
                <div className="grid grid-cols-2 gap-2">
                  <StatCard label="Fused" value={radar.fused_count ?? 0} color="text-indigo-400" />
                  <StatCard label="Radar Only" value={radar.radar_only_count ?? 0} color="text-blue-400" />
                </div>
              )}
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

function eventColor(type) {
  const map = {
    appeared: 'text-green-400',
    left: 'text-red-400',
    reappeared: 'text-yellow-400',
    reidentified: 'text-cyan-400',
    ai_analysis: 'text-purple-400',
    radar_track_ended: 'text-blue-400',
    fusion_confirmed: 'text-indigo-400',
  }
  return map[type] || 'text-gray-300'
}

function SourceBadge({ source }) {
  const styles = {
    camera: 'bg-green-900/50 text-green-400',
    radar: 'bg-blue-900/50 text-blue-400',
    fused: 'bg-indigo-900/50 text-indigo-400',
  }
  return (
    <span className={`text-[9px] px-1 rounded ${styles[source] || styles.camera}`}>
      {source === 'fused' ? 'Fused' : source === 'radar' ? 'Radar' : 'Cam'}
    </span>
  )
}

function EntityCard({ entity }) {
  const isGone = entity.status === 'gone'
  const borderColor = isGone ? 'border-gray-700' :
    entity.source === 'radar' ? 'border-blue-500/50' :
    entity.source === 'fused' ? 'border-indigo-500/50' :
    entity.label === 'person' ? 'border-red-500/50' : 'border-orange-500/50'

  const rd = entity.radar_data

  return (
    <div className={`flex items-center gap-2 p-2 rounded-lg bg-gray-800/50 border ${borderColor} ${isGone ? 'opacity-50' : ''}`}>
      {entity.thumbnail_b64 ? (
        <img src={`data:image/jpeg;base64,${entity.thumbnail_b64}`} alt={entity.display_name}
          className="w-10 h-10 rounded object-cover shrink-0" />
      ) : (
        <div className={`w-10 h-10 rounded flex items-center justify-center shrink-0 ${entity.source === 'radar' ? 'bg-blue-900/50' : 'bg-gray-700'}`}>
          <span className={`text-xs ${entity.source === 'radar' ? 'text-blue-400' : 'text-gray-500'}`}>
            {entity.source === 'radar' ? 'R' : entity.label[0].toUpperCase()}
          </span>
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium truncate flex items-center gap-1.5">
          {entity.display_name}
          <SourceBadge source={entity.source} />
          {entity.reidentified && <span className="text-[9px] bg-cyan-900/50 text-cyan-400 px-1 rounded">Re-ID</span>}
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className={`w-1.5 h-1.5 rounded-full ${isGone ? 'bg-gray-500' : 'bg-green-500'}`} />
          <span>{isGone ? 'Gone' : `${entity.duration}s`}</span>
          <span className="text-gray-600">|</span>
          <span>{Math.round(entity.best_conf * 100)}%</span>
        </div>
        {/* Radar data row */}
        {rd && (
          <div className="flex items-center gap-2 text-[10px] text-blue-400 mt-0.5">
            {rd.range_m != null && <span>{rd.range_m}m</span>}
            {rd.speed_ms != null && <span>{rd.speed_ms} m/s</span>}
            {rd.angle_deg != null && <span>{rd.angle_deg > 0 ? '+' : ''}{rd.angle_deg}</span>}
          </div>
        )}
      </div>
    </div>
  )
}

function RadarMiniMap({ tracks }) {
  const size = 160
  const cx = size / 2
  const cy = size - 10
  const maxRange = 60
  const scale = (size - 20) / maxRange

  return (
    <div className="bg-gray-900/90 rounded-lg border border-blue-900/50 p-1" style={{ backdropFilter: 'blur(4px)' }}>
      <div className="text-[9px] text-blue-400 text-center mb-0.5 font-semibold">RADAR</div>
      <svg width={size} height={size} className="block">
        {/* Range rings */}
        {[15, 30, 60].map(r => (
          <g key={r}>
            <circle cx={cx} cy={cy} r={r * scale} fill="none" stroke="rgba(59,130,246,0.15)" strokeWidth="1" />
            <text x={cx + 2} y={cy - r * scale + 10} fill="rgba(59,130,246,0.3)" fontSize="8">{r}m</text>
          </g>
        ))}
        {/* FOV lines */}
        <line x1={cx} y1={cy} x2={0} y2={10} stroke="rgba(59,130,246,0.2)" strokeWidth="1" />
        <line x1={cx} y1={cy} x2={size} y2={10} stroke="rgba(59,130,246,0.2)" strokeWidth="1" />
        {/* Center dot (radar position) */}
        <circle cx={cx} cy={cy} r="3" fill="#3b82f6" />
        {/* Targets */}
        {tracks.map((t, i) => {
          const angle = (t.polar?.angle_deg || 0) * Math.PI / 180
          const range = Math.min(t.polar?.range_m || 0, maxRange)
          const tx = cx + Math.sin(angle) * range * scale
          const ty = cy - Math.cos(angle) * range * scale
          const isHuman = t.class === 'Human'
          return (
            <g key={i}>
              <circle cx={tx} cy={ty} r={isHuman ? 4 : 6}
                fill={isHuman ? 'rgba(239,68,68,0.8)' : 'rgba(251,146,60,0.8)'}
                stroke={isHuman ? '#ef4444' : '#fb923c'} strokeWidth="1" />
              {t.velocity?.speed_ms > 0.5 && (
                <line x1={tx} y1={ty}
                  x2={tx + Math.sin((t.velocity.heading_deg || 0) * Math.PI / 180) * 10}
                  y2={ty - Math.cos((t.velocity.heading_deg || 0) * Math.PI / 180) * 10}
                  stroke={isHuman ? '#ef4444' : '#fb923c'} strokeWidth="1.5" markerEnd="url(#arrow)" />
              )}
            </g>
          )
        })}
        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="1" />
          </marker>
        </defs>
      </svg>
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
    reidentified: '=',
    ai_analysis: '*',
    radar_track_ended: 'R',
    fusion_confirmed: 'F',
  }
  const colors = {
    appeared: 'text-green-500',
    left: 'text-red-500',
    reappeared: 'text-yellow-500',
    reidentified: 'text-cyan-500',
    ai_analysis: 'text-purple-500',
    radar_track_ended: 'text-blue-500',
    fusion_confirmed: 'text-indigo-500',
  }
  return <span className={`${colors[type] || 'text-gray-500'} font-bold shrink-0`}>{icons[type] || '·'}</span>
}
