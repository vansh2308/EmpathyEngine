import React, { useRef, useEffect, useState } from 'react'
import { Play, Pause, Volume2 } from 'lucide-react'
import { Button } from './ui/button'
import { ScrollingWaveform } from './ui/waveform'

interface WaveformDemoProps {
  audioUrl: string
}

export function WaveformDemo({ audioUrl }: WaveformDemoProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const context = new (window.AudioContext || (window as any).webkitAudioContext)()
    const analyser = context.createAnalyser()
    const source = context.createMediaElementAudioSource(audio)
    source.connect(analyser)
    analyser.connect(context.destination)
    analyser.fftSize = 2048

    const dataArray = new Uint8Array(analyser.frequencyBinCount)
    let animationId: number

    const draw = () => {
      analyser.getByteFrequencyData(dataArray)

      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.fillStyle = 'rgb(15, 23, 42)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      ctx.fillStyle = 'rgb(59, 130, 246)'
      const barWidth = (canvas.width / dataArray.length) * 2.5
      let x = 0

      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)
        x += barWidth + 1
      }

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => cancelAnimationFrame(animationId)
  }, [])

  const handlePlayPause = () => {
    const audio = audioRef.current
    if (!audio) return

    if (isPlaying) {
      audio.pause()
    } else {
      audio.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const formatTime = (time: number) => {
    if (isNaN(time)) return '0:00'
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="w-full">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      <canvas ref={canvasRef} className="w-full h-24 bg-slate-900 rounded-lg mb-4" />

      <div className="flex items-center gap-4 mb-4">
        <Button
          onClick={handlePlayPause}
          className="bg-blue-600 hover:bg-blue-700"
          size="sm"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>

        <div className="flex-1 bg-slate-200 rounded h-2">
          <div
            className="bg-blue-600 h-2 rounded transition-all"
            style={{
              width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
            }}
          />
        </div>

        <span className="text-sm text-gray-600 min-w-fit">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}
