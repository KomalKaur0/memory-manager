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

const generateDemoResponseWithMemoryAccess = (): { content: string; memoryAccesses: MemoryAccessEvent[] } => {
  const demoResponses = [
    {
      content: "That's an interesting question! Let me think about how your recent learning experiences might inform this. Your reinforcement learning studies and the psychology insights from Kahneman's work could provide a valuable framework for approaching this challenge.",
      memoryAccesses: [
        { node_id: 'demo_3', access_type: 'read' as const, timestamp: Date.now() - 2000 },
        { node_id: 'demo_5', access_type: 'read' as const, timestamp: Date.now() - 1500 },
        { node_id: 'demo_12', access_type: 'traverse' as const, timestamp: Date.now() - 1000, connection_id: 'demo_3-demo_5' }
      ]
    },
    {
      content: "This reminds me of your work on that React project and the AI research you've been doing. The TypeScript setup and your exploration of transformer architectures could be really relevant here. Also, your experience from the AI conference about human-AI collaboration seems applicable.",
      memoryAccesses: [
        { node_id: 'demo_4', access_type: 'read' as const, timestamp: Date.now() - 1800 },
        { node_id: 'demo_2', access_type: 'read' as const, timestamp: Date.now() - 1200 },
        { node_id: 'demo_12', access_type: 'traverse' as const, timestamp: Date.now() - 800, connection_id: 'demo_2-demo_12' },
        { node_id: 'demo_1', access_type: 'strengthen' as const, timestamp: Date.now() - 400, weight_change: 0.12 }
      ]
    },
    {
      content: "Your disciplined approach to the morning workout routine and guitar practice shows you understand the value of consistent effort. This same mindset could work well for this project, especially considering your goal-oriented thinking from those quarterly planning sessions.",
      memoryAccesses: [
        { node_id: 'demo_6', access_type: 'read' as const, timestamp: Date.now() - 2200 },
        { node_id: 'demo_10', access_type: 'read' as const, timestamp: Date.now() - 1600 },
        { node_id: 'demo_1', access_type: 'traverse' as const, timestamp: Date.now() - 1000, connection_id: 'demo_6-demo_1' }
      ]
    },
    {
      content: "I notice you have interests in both creative and analytical areas - like your photography work focusing on composition and your investment research into ESG funds. This balance of creative and logical thinking could bring a unique perspective to solving this problem.",
      memoryAccesses: [
        { node_id: 'demo_11', access_type: 'read' as const, timestamp: Date.now() - 1900 },
        { node_id: 'demo_9', access_type: 'read' as const, timestamp: Date.now() - 1300 },
        { node_id: 'demo_5', access_type: 'traverse' as const, timestamp: Date.now() - 700, connection_id: 'demo_11-demo_5' }
      ]
    }
  ];

  return demoResponses[Math.floor(Math.random() * demoResponses.length)];
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
    
    // Import memory store to check if demo data is loaded
    const { useMemoryStore } = require('./memoryStore');
    const memoryStore = useMemoryStore.getState();
    const nodes = Object.keys(memoryStore.nodes);
    const isDemoData = nodes.some(id => id.startsWith('demo_'));
    
    const response = isDemoData ? generateDemoResponseWithMemoryAccess() : generateResponseWithMemoryAccess();
    
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
    
    // Check if demo data is loaded to use appropriate node IDs
    const nodes = Object.keys(memoryStore.nodes);
    const isDemoData = nodes.some(id => id.startsWith('demo_'));
    
    // Simulate sequential memory access with realistic timing
    const thinkingSequence = isDemoData ? [
      { nodeId: 'demo_3', accessType: 'read' as const, delay: 800 },
      { nodeId: 'demo_5', accessType: 'traverse' as const, delay: 1200 },
      { nodeId: 'demo_12', accessType: 'read' as const, delay: 600 },
      { nodeId: 'demo_2', accessType: 'strengthen' as const, delay: 900 },
      { nodeId: 'demo_4', accessType: 'read' as const, delay: 700 },
    ] : [
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