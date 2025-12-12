import { useState } from 'react'
import { importDxf } from '../api/client'

interface Props {
  onImport: (sessionId: string) => void
}

export default function ImportPage({ onImport }: Props) {
  const [uploading, setUploading] = useState(false)

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    try {
      const result = await importDxf(file)
      onImport(result.session_id)
    } catch (error) {
      console.error('Import failed:', error)
      alert('Failed to import DXF file')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Import DXF</h2>
      <input
        type="file"
        accept=".dxf"
        onChange={handleFile}
        disabled={uploading}
      />
      {uploading && <p>Uploading...</p>}
    </div>
  )
}

