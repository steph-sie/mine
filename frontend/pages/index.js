import { useEffect, useRef, useState } from 'react'

export default function Home() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('idle')

  useEffect(() => {
    let stream
    async function startCamera() {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      if (videoRef.current) videoRef.current.srcObject = stream
    }
    startCamera()
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop())
    }
  }, [])

  useEffect(() => {
    let timer
    if (running) {
      timer = setInterval(captureAndSend, 700)
    }
    return () => clearInterval(timer)
  }, [running])

  async function captureAndSend() {
    if (!videoRef.current) return
    const video = videoRef.current
    const w = video.videoWidth
    const h = video.videoHeight
    if (!w || !h) return

    const off = document.createElement('canvas')
    off.width = w
    off.height = h
    const ctx = off.getContext('2d')
    ctx.drawImage(video, 0, 0, w, h)
    const blob = await new Promise(resolve => off.toBlob(resolve, 'image/jpeg', 0.7))

    const form = new FormData()
    form.append('file', blob, 'frame.jpg')

    try {
      setStatus('sending')
      const res = await fetch('http://localhost:8000/detect?conf=0.35', { method: 'POST', body: form })
      const data = await res.json()
      drawBoxes(data.boxes, w, h)
      setStatus('ok')
    } catch (e) {
      console.error(e)
      setStatus('error')
    }
  }

  function drawBoxes(boxes, vw, vh) {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const cw = video.clientWidth
    const ch = video.clientHeight
    canvas.width = cw
    canvas.height = ch
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, cw, ch)
    ctx.strokeStyle = '#00FF00'
    ctx.lineWidth = 2
    ctx.font = '14px Arial'
    ctx.fillStyle = '#00FF00'

    const xScale = cw / vw
    const yScale = ch / vh

    (boxes || []).forEach(b => {
      const x1 = b.x1 * xScale
      const y1 = b.y1 * yScale
      const x2 = b.x2 * xScale
      const y2 = b.y2 * yScale
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.fillText(`${b.label} ${b.conf.toFixed(2)}`, x1 + 4, y1 + 14)
    })
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">CCTV Live — Next.js + Tailwind</h1>
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2">
            <div className="canvas-wrapper relative">
              <video ref={videoRef} autoPlay muted playsInline className="w-full rounded-md" />
              <canvas ref={canvasRef} className="overlay-canvas" />
            </div>
            <div className="flex items-center gap-2 mt-3">
              <button onClick={() => setRunning(true)} className="px-3 py-1 bg-green-600 rounded">Start</button>
              <button onClick={() => setRunning(false)} className="px-3 py-1 bg-red-600 rounded">Stop</button>
              <div className="ml-4">Status: {status}</div>
            </div>
          </div>
          <div className="col-span-1">
            <div className="bg-gray-800 p-3 rounded h-96 overflow-auto">
              <h2 className="font-semibold mb-2">Recent Events</h2>
              <ul>
                <li>Logs appear here (backend writes to file)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
