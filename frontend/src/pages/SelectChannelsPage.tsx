interface Props {
  sessionId: string
  onContinue: () => void
}

export default function SelectChannelsPage({ sessionId, onContinue }: Props) {
  return (
    <div style={{ padding: '2rem' }}>
      <h2>Select Channels</h2>
      <p>Session: {sessionId}</p>
      <button onClick={onContinue}>Continue</button>
    </div>
  )
}

