import { exportBundle } from '../api/client'

interface Props {
  sessionId: string
}

export default function ExportPage({ sessionId }: Props) {
  const handleExport = async () => {
    try {
      const blob = await exportBundle(sessionId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'design_export.zip'
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
      alert('Failed to export')
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Export</h2>
      <p>Session: {sessionId}</p>
      <button onClick={handleExport}>Download Export Bundle</button>
    </div>
  )
}

