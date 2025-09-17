import { create } from 'zustand'
import type { GraphOutputs } from './types'

interface GraphState {
  outputs: GraphOutputs
  setOutputs: (outputs: GraphOutputs) => void
}

export const useGraphStore = create<GraphState>((set) => ({
  outputs: {},
  setOutputs: (outputs) => set({ outputs }),
}))
