interface ScrollingWaveformProps {
  audioUrl: string
}

export function ScrollingWaveform({ audioUrl }: ScrollingWaveformProps) {
  return (
    <div className="w-full bg-slate-900 rounded-lg h-24 flex items-center justify-center">
      <p className="text-white text-sm">Waveform Display</p>
    </div>
  )
}
