// TODO: Add state management (context or zustand)
export interface AppState {
  sessionId: string | null
  layers: string[]
  selectedRegions: string[]
  nodes: any[]
  edges: any[]
  ports: any[]
}

export const initialState: AppState = {
  sessionId: null,
  layers: [],
  selectedRegions: [],
  nodes: [],
  edges: [],
  ports: [],
}

