import React, { useState } from 'react'
import axios from 'axios'
import { Input } from './components/ui/input'
import { Button } from './components/ui/button'
import { WaveformDemo } from './components/waveformDemo'
import './index.css'

export default function App() {
  const [text, setText] = useState('GET OUT!!!')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [emotion, setEmotion] = useState('unknown')
  const [intensity, setIntensity] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSynthesizeAudio = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post('http://localhost:8000/synthesize', {
        text,
        return_debug: true,
      })

      // Extract emotion and intensity from debug field
      if (response.data.debug) {
        setEmotion(response.data.debug.primary_emotion || 'unknown')
        setIntensity(response.data.debug.intensity || 0)
      }

      // Convert base64 to blob and create object URL
      const binaryString = atob(response.data.audio_base64)
      const bytes = new Uint8Array(binaryString.length)
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i)
      }
      const blob = new Blob([bytes], { type: 'audio/mpeg' })
      const url = URL.createObjectURL(blob)
      setAudioUrl(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to synthesize audio')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">EmpathyEngine</h1>
          <p className="text-gray-600">Emotion-aware Text-to-Speech</p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Enter text to synthesize:
          </label>
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="GET OUT!!!"
            className="mb-4"
          />
          <Button
            onClick={handleSynthesizeAudio}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700"
          >
            {loading ? 'Generating...' : 'Send'}
          </Button>

          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}
        </div>

        {audioUrl && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-700">
                Emotion: <span className="text-blue-600 font-semibold">{emotion}</span>
              </p>
              <p className="text-sm font-medium text-gray-700">
                Intensity: <span className="text-blue-600 font-semibold">{intensity.toFixed(2)}</span>
              </p>
            </div>
            <WaveformDemo audioUrl={audioUrl} />
          </div>
        )}
      </div>
    </div>
  )
}
