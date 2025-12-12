interface Props {
  sessionId: string
  onContinue: () => void
}

export default function LabelPage({ sessionId, onContinue }: Props) {
  return (
    <div style={{ padding: '2rem' }}>
      <h2>Label & Parameters</h2>
      <p>Session: {sessionId}</p>
      <button onClick={onContinue}>Continue</button>
    </div>
  )
}

