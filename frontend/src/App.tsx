import { useState } from 'react'
import ImportPage from './pages/ImportPage'
import SelectChannelsPage from './pages/SelectChannelsPage'
import LabelPage from './pages/LabelPage'
import ExportPage from './pages/ExportPage'

type Screen = 'import' | 'select' | 'label' | 'export'

function App() {
  const [screen, setScreen] = useState<Screen>('import')
  const [sessionId, setSessionId] = useState<string | null>(null)

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ padding: '1rem', borderBottom: '1px solid #ccc' }}>
        <h1>Microfluidic DXF Converter</h1>
      </header>
      <main style={{ flex: 1, overflow: 'auto' }}>
        {screen === 'import' && (
          <ImportPage
            onImport={(sid) => {
              setSessionId(sid)
              setScreen('select')
            }}
          />
        )}
        {screen === 'select' && (
          <SelectChannelsPage
            sessionId={sessionId!}
            onContinue={() => setScreen('label')}
          />
        )}
        {screen === 'label' && (
          <LabelPage
            sessionId={sessionId!}
            onContinue={() => setScreen('export')}
          />
        )}
        {screen === 'export' && (
          <ExportPage sessionId={sessionId!} />
        )}
      </main>
    </div>
  )
}

export default App

