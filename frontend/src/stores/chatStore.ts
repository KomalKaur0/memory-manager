import { create } from 'zustand';
import { ChatState, ChatMessage, MemoryAccessEvent } from '../types/memory';

interface ChatStore extends ChatState {
  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  setTyping: (typing: boolean) => void;
  setCurrentInput: (input: string) => void;
  setMemoryVisualizationVisible: (visible: boolean) => void;
  
  // Simulation methods
  simulateAssistantResponse: () => void;
  simulateThinkingWithMemoryAccess: () => Promise<void>;
}

const generateResponseWithMemoryAccess = (): { content: string; memoryAccesses: MemoryAccessEvent[] } => {
  const responses = [
    {
      content: "I understand you're looking to implement a comprehensive AI memory system. Based on our previous discussions about graph-based memory and 3D visualization, I think we should focus on the core architecture first.",
      memoryAccesses: [
        { node_id: '1', access_type: 'read' as const, timestamp: Date.now() - 2000 },
        { node_id: '2', access_type: 'read' as const, timestamp: Date.now() - 1500 },
        { node_id: '3', access_type: 'traverse' as const, timestamp: Date.now() - 1000, connection_id: '2-3' },
        { node_id: '4', access_type: 'strengthen' as const, timestamp: Date.now() - 500, weight_change: 0.1 }
      ]
    },
    {
      content: "The memory retrieval system you described combines embedding search with graph traversal - this is a powerful approach. The 3D visualization will really help users understand how memories connect and influence each other.",
      memoryAccesses: [
        { node_id: '5', access_type: 'read' as const, timestamp: Date.now() - 1800 },
        { node_id: '3', access_type: 'read' as const, timestamp: Date.now() - 1200 },
        { node_id: '2', access_type: 'traverse' as const, timestamp: Date.now() - 800, connection_id: '5-2' }
      ]
    },
    {
      content: "Let's break this down into phases. We'll start with the React/Expo setup, then build the chat interface, and finally create the 3D memory visualization with Three.js. Each component will need to work together seamlessly.",
      memoryAccesses: [
        { node_id: '4', access_type: 'read' as const, timestamp: Date.now() - 2200 },
        { node_id: '3', access_type: 'read' as const, timestamp: Date.now() - 1600 },
        { node_id: '1', access_type: 'traverse' as const, timestamp: Date.now() - 1000, connection_id: '4-1' },
        { node_id: '2', access_type: 'strengthen' as const, timestamp: Date.now() - 400, weight_change: 0.15 }
      ]
    }
  ];

  return responses[Math.floor(Math.random() * responses.length)];
};

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  isTyping: false,
  currentInput: '',
  memoryVisualizationVisible: false,

  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: Date.now()
        }
      ]
    })),

  setTyping: (typing) =>
    set({ isTyping: typing }),

  setCurrentInput: (input) =>
    set({ currentInput: input }),

  setMemoryVisualizationVisible: (visible) =>
    set({ memoryVisualizationVisible: visible }),

  simulateAssistantResponse: () => {
    const { addMessage } = get();
    const response = generateResponseWithMemoryAccess();
    
    addMessage({
      content: response.content,
      role: 'assistant',
      memory_accesses: response.memoryAccesses
    });
  },

  simulateThinkingWithMemoryAccess: async () => {
    const { setTyping, setMemoryVisualizationVisible, simulateAssistantResponse } = get();
    
    // Import memory store dynamically to avoid circular dependency
    const { useMemoryStore } = await import('./memoryStore');
    const memoryStore = useMemoryStore.getState();
    
    // Clear any previous memory access states to avoid animation conflicts
    memoryStore.clearRecentAccesses();
    
    // Show thinking state and memory visualization
    setTyping(true);
    setMemoryVisualizationVisible(true);
    memoryStore.setThinking(true);
    
    // Simulate sequential memory access with realistic timing
    const thinkingSequence = [
      { nodeId: '1', accessType: 'read' as const, delay: 800 },
      { nodeId: '2', accessType: 'traverse' as const, delay: 1200 },
      { nodeId: '3', accessType: 'read' as const, delay: 600 },
      { nodeId: '5', accessType: 'strengthen' as const, delay: 900 },
      { nodeId: '4', accessType: 'read' as const, delay: 700 },
    ];
    
    // Execute memory access sequence
    let totalDelay = 0;
    for (const step of thinkingSequence) {
      await new Promise(resolve => setTimeout(resolve, step.delay));
      memoryStore.simulateMemoryAccess(step.nodeId, step.accessType);
      totalDelay += step.delay;
    }
    
    // Add a final pause for realism
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Add response and hide visualization
    simulateAssistantResponse();
    setTyping(false);
    setMemoryVisualizationVisible(false);
    memoryStore.setThinking(false);
  }
}));