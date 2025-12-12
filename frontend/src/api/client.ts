const API_BASE = 'http://localhost:8000'

export async function importDxf(file: File): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/api/import-dxf`, {
    method: 'POST',
    body: formData,
  })
  return res.json()
}

export async function selectChannels(data: any): Promise<any> {
  const res = await fetch(`${API_BASE}/api/select-channels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function extractGraph(data: any): Promise<any> {
  const res = await fetch(`${API_BASE}/api/extract-graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function setDefaults(data: any): Promise<any> {
  const res = await fetch(`${API_BASE}/api/set-defaults`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function assignPorts(data: any): Promise<any> {
  const res = await fetch(`${API_BASE}/api/assign-ports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return res.json()
}

export async function exportBundle(sessionId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  return res.blob()
}

