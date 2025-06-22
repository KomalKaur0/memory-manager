import { create } from 'zustand';
import { MemoryGraphState, MemoryNode, MemoryAccessEvent } from '../types/memory';
import { apiService } from '../services/apiService';

interface MemoryStore extends MemoryGraphState {
  // Actions
  addNode: (node: MemoryNode) => void;
  selectNode: (nodeId: string | null) => void;
  addMemoryAccess: (access: MemoryAccessEvent) => void;
  setThinking: (thinking: boolean) => void;
  updateConnectionWeight: (nodeId: string, targetId: string, weightChange: number) => void;
  setActiveConnections: (connectionIds: string[]) => void;
  
  // Backend integration methods
  loadMemoryNodes: () => Promise<boolean>;
  refreshMemoryGraph: () => Promise<boolean>;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Mock data methods
  generateMockData: () => void;
  generateDemoData: () => void;
  simulateMemoryAccess: (nodeId: string, accessType: MemoryAccessEvent['access_type']) => void;
  clearRecentAccesses: () => void;
  
  // State
  loading: boolean;
  error: string | null;
  isUsingBackend: boolean;
  initialized: boolean;
}

// Generate mock embeddings (normally would come from embedding model)
const generateMockEmbedding = (concepts: string[], tags: string[]): number[] => {
  // Simulate 384-dimensional embedding (typical for sentence transformers)
  const embedding = new Array(384).fill(0);
  
  // Create semantic fingerprint based on concepts and tags
  const allTerms = [...concepts, ...tags];
  allTerms.forEach((term, index) => {
    const hash = Array.from(term).reduce((acc, char) => acc + char.charCodeAt(0), 0);
    for (let i = 0; i < 384; i++) {
      embedding[i] += Math.sin(hash * (i + 1) * 0.01) * 0.1;
    }
  });
  
  // Normalize embedding
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / magnitude);
};

const generateMockNodes = (): Record<string, MemoryNode> => {
  const mockNodes: Record<string, MemoryNode> = {};
  
  const nodeData = [
    {
      id: '1',
      tags: ['conversation', 'greeting'],
      summary: 'User introduction and project overview',
      content: 'User introduced themselves and explained the AI memory system project goals. Discussed the need for adaptive graph-based memory that combines semantic similarity with learned associations.',
      concepts: ['introduction', 'project', 'goals', 'adaptive_memory'],
      keywords: ['memory', 'system', 'AI', 'project', 'graph', 'semantic']
    },
    {
      id: '2',
      tags: ['technical', 'architecture'],
      summary: 'Memory graph structure and weighted connections',
      content: 'Detailed discussion about memory nodes containing tags, summary, content, and bidirectional connections with independent weights representing semantic relationships and usage patterns.',
      concepts: ['architecture', 'graph', 'nodes', 'weighted_edges'],
      keywords: ['graph', 'nodes', 'structure', 'connections', 'weights', 'bidirectional']
    },
    {
      id: '3',
      tags: ['visualization', 'vector_space'],
      summary: '3D vector space memory visualization',
      content: 'User wants 3D representation of memory space positioned based on embeddings, showing semantic clustering and connection weights in vector space.',
      concepts: ['visualization', 'vector_space', 'embeddings', 'clustering'],
      keywords: ['3D', 'visualization', 'vector', 'embeddings', 'clustering', 'semantic']
    },
    {
      id: '4',
      tags: ['frontend', 'react'],
      summary: 'React/Expo frontend with Three.js',
      content: 'Implementation using React/Expo for cross-platform development with Three.js for 3D rendering of the memory vector space.',
      concepts: ['frontend', 'react', 'expo', 'three_js'],
      keywords: ['react', 'expo', 'frontend', 'typescript', 'three.js', 'WebGL']
    },
    {
      id: '5',
      tags: ['memory_system', 'retrieval'],
      summary: 'Hybrid embedding + graph traversal system',
      content: 'Embedding-based similarity search for initial discovery, followed by pure graph traversal based on learned connection weights for associative memory exploration.',
      concepts: ['retrieval', 'embedding', 'search', 'graph_traversal', 'hybrid_system'],
      keywords: ['retrieval', 'embedding', 'search', 'traversal', 'weights', 'associations']
    },
    {
      id: '6',
      tags: ['technical', 'vector_space'],
      summary: 'High-dimensional embedding space projection',
      content: 'Memory embeddings exist in high-dimensional space (typically 384-1536 dimensions) and must be projected to 3D for visualization using PCA or t-SNE techniques.',
      concepts: ['embeddings', 'high_dimensional', 'projection', 'dimensionality_reduction'],
      keywords: ['embeddings', 'PCA', 't-SNE', 'dimensionality', 'projection', 'visualization']
    },
    {
      id: '7',
      tags: ['conversation', 'memory_access'],
      summary: 'Real-time memory access visualization during AI thinking',
      content: 'Show which memories are being accessed in real-time as the AI formulates responses, highlighting the traversal path through the memory graph.',
      concepts: ['real_time', 'memory_access', 'ai_thinking', 'traversal_visualization'],
      keywords: ['real-time', 'access', 'thinking', 'traversal', 'visualization', 'highlighting']
    }
  ];

  // Generate mock embeddings and position nodes in vector space
  nodeData.forEach((data, index) => {
    // Generate realistic embedding based on semantic content
    const embedding = generateMockEmbedding(data.concepts, data.tags);
    
    // Use PCA-like projection to 3D (simplified simulation)
    // In reality, this would use actual dimensionality reduction algorithms
    const position_3d: [number, number, number] = [
      embedding.slice(0, 128).reduce((sum, val) => sum + val, 0) * 10,
      embedding.slice(128, 256).reduce((sum, val) => sum + val, 0) * 8,
      embedding.slice(256, 384).reduce((sum, val) => sum + val, 0) * 10,
    ];
    
    const node: MemoryNode = {
      ...data,
      connections: {},
      embedding,
      position_3d,
      created_at: Date.now() - (nodeData.length - index) * 60000,
      last_accessed: Date.now() - Math.random() * 300000,
      access_count: Math.floor(Math.random() * 50) + 1
    };
    mockNodes[node.id] = node;
  });

  // Add semantic connections based on embedding similarity and usage patterns
  // Project overview (1) → Architecture (2): Strong connection
  mockNodes['1'].connections['2'] = { outbound_weight: 0.85, inbound_weight: 0.80, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Architecture (2) → Visualization (3): Very strong connection (both technical)
  mockNodes['2'].connections['3'] = { outbound_weight: 0.92, inbound_weight: 0.88, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Visualization (3) → Frontend (4): Strong connection (implementation related)
  mockNodes['3'].connections['4'] = { outbound_weight: 0.78, inbound_weight: 0.75, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Architecture (2) → Retrieval System (5): Strong technical connection
  mockNodes['2'].connections['5'] = { outbound_weight: 0.89, inbound_weight: 0.82, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Visualization (3) → Vector Space (6): Very strong connection (same domain)
  mockNodes['3'].connections['6'] = { outbound_weight: 0.94, inbound_weight: 0.91, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Vector Space (6) → Real-time Access (7): Medium connection
  mockNodes['6'].connections['7'] = { outbound_weight: 0.68, inbound_weight: 0.65, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Project Overview (1) → Real-time Access (7): Conceptual connection
  mockNodes['1'].connections['7'] = { outbound_weight: 0.72, inbound_weight: 0.70, last_accessed: Date.now(), creation_date: Date.now() };
  
  // Frontend (4) → Real-time Access (7): Implementation connection
  mockNodes['4'].connections['7'] = { outbound_weight: 0.81, inbound_weight: 0.76, last_accessed: Date.now(), creation_date: Date.now() };

  return mockNodes;
};

const generateDemoNodes = (): Record<string, MemoryNode> => {
  const demoNodes: Record<string, MemoryNode> = {};
  
  const nodeData = [
    {
      id: 'demo_1',
      tags: ['personal', 'work', 'productivity'],
      summary: 'Meeting about quarterly goals and team restructuring',
      content: 'Attended quarterly planning meeting with Sarah and team. Discussed new OKRs for Q4, potential team expansion, and the new remote work policy. Key takeaway: need to increase customer satisfaction metrics by 15%.',
      concepts: ['planning', 'goals', 'team', 'metrics', 'satisfaction'],
      keywords: ['quarterly', 'OKRs', 'team', 'remote', 'satisfaction', 'metrics']
    },
    {
      id: 'demo_2',
      tags: ['technical', 'AI', 'research'],
      summary: 'Research on transformer architecture improvements',
      content: 'Deep dive into latest transformer variants including MoE (Mixture of Experts) models. Found interesting paper on reducing computational overhead while maintaining performance. Could be relevant for our NLP pipeline optimization.',
      concepts: ['transformers', 'MoE', 'optimization', 'NLP', 'performance'],
      keywords: ['transformer', 'experts', 'computational', 'performance', 'NLP', 'optimization']
    },
    {
      id: 'demo_3',
      tags: ['learning', 'course', 'machine_learning'],
      summary: 'Completed module on reinforcement learning fundamentals',
      content: 'Finished the RL basics course covering Q-learning, policy gradients, and actor-critic methods. The hands-on CartPole environment was particularly insightful for understanding exploration vs exploitation tradeoffs.',
      concepts: ['reinforcement_learning', 'Q_learning', 'policy_gradients', 'exploration', 'exploitation'],
      keywords: ['reinforcement', 'Q-learning', 'policy', 'actor-critic', 'CartPole', 'exploration']
    },
    {
      id: 'demo_4',
      tags: ['project', 'frontend', 'react'],
      summary: 'Started new React project with TypeScript and Zustand',
      content: 'Initialized a new frontend project using React 18, TypeScript, and Zustand for state management. Set up modern development stack with Vite, ESLint, and testing framework. Project will be a collaborative task management tool.',
      concepts: ['React', 'TypeScript', 'Zustand', 'frontend', 'development'],
      keywords: ['React', 'TypeScript', 'Zustand', 'Vite', 'ESLint', 'task', 'management']
    },
    {
      id: 'demo_5',
      tags: ['book', 'psychology', 'learning'],
      summary: 'Reading "Thinking, Fast and Slow" insights on cognitive biases',
      content: 'Key insights from Kahneman\'s work on System 1 vs System 2 thinking. Understanding how cognitive shortcuts can lead to systematic errors. Particularly relevant for improving decision-making in product development.',
      concepts: ['cognitive_biases', 'System_1', 'System_2', 'decision_making', 'psychology'],
      keywords: ['Kahneman', 'cognitive', 'biases', 'thinking', 'decision', 'shortcuts', 'errors']
    },
    {
      id: 'demo_6',
      tags: ['health', 'exercise', 'routine'],
      summary: 'New morning workout routine showing good results',
      content: 'Started a 30-minute morning routine combining cardio and strength training. Week 3 shows improved energy levels throughout the day and better sleep quality. Planning to add yoga for flexibility.',
      concepts: ['exercise', 'routine', 'cardio', 'strength', 'energy', 'sleep'],
      keywords: ['workout', 'morning', 'cardio', 'strength', 'energy', 'sleep', 'yoga', 'flexibility']
    },
    {
      id: 'demo_7',
      tags: ['cooking', 'recipe', 'italian'],
      summary: 'Mastered homemade pasta and marinara sauce',
      content: 'Successfully made fresh pasta from scratch using 00 flour and eggs. The marinara sauce with San Marzano tomatoes, fresh basil, and good olive oil made all the difference. Family loved it!',
      concepts: ['cooking', 'pasta', 'sauce', 'italian', 'homemade'],
      keywords: ['pasta', 'flour', 'marinara', 'tomatoes', 'basil', 'olive oil', 'italian', 'homemade']
    },
    {
      id: 'demo_8',
      tags: ['travel', 'planning', 'vacation'],
      summary: 'Planning summer trip to Japan for cherry blossom season',
      content: 'Researching optimal timing for cherry blossoms in different regions. Tokyo and Kyoto seem best for late March to early April. Need to book accommodations soon and plan itinerary around cultural sites and food experiences.',
      concepts: ['travel', 'Japan', 'cherry_blossom', 'planning', 'culture'],
      keywords: ['Japan', 'cherry', 'blossom', 'Tokyo', 'Kyoto', 'accommodations', 'cultural', 'food']
    },
    {
      id: 'demo_9',
      tags: ['investment', 'finance', 'research'],
      summary: 'Research on sustainable investing and ESG funds',
      content: 'Exploring ESG (Environmental, Social, Governance) investment options. Found several funds with strong track records in renewable energy and responsible tech companies. Need to balance returns with personal values.',
      concepts: ['investing', 'ESG', 'sustainability', 'renewable_energy', 'values'],
      keywords: ['ESG', 'environmental', 'social', 'governance', 'renewable', 'energy', 'sustainable', 'funds']
    },
    {
      id: 'demo_10',
      tags: ['music', 'learning', 'guitar'],
      summary: 'Progress on fingerpicking guitar technique',
      content: 'Been practicing Travis picking pattern for acoustic guitar. Can now play simple folk songs smoothly. The coordination between thumb and fingers is finally clicking. Next goal: barre chords.',
      concepts: ['music', 'guitar', 'fingerpicking', 'technique', 'coordination'],
      keywords: ['guitar', 'fingerpicking', 'Travis', 'acoustic', 'folk', 'coordination', 'barre', 'chords']
    },
    {
      id: 'demo_11',
      tags: ['photography', 'hobby', 'nature'],
      summary: 'Nature photography workshop on composition techniques',
      content: 'Attended workshop focusing on rule of thirds, leading lines, and natural framing. Practiced with macro lens for flower photography. The golden hour lighting techniques will be useful for landscape shots.',
      concepts: ['photography', 'composition', 'nature', 'macro', 'lighting'],
      keywords: ['photography', 'composition', 'rule', 'thirds', 'leading', 'lines', 'macro', 'golden', 'hour']
    },
    {
      id: 'demo_12',
      tags: ['networking', 'conference', 'AI'],
      summary: 'AI conference insights on future of human-AI collaboration',
      content: 'Key takeaways from the AI Summit: emphasis on human-AI collaboration rather than replacement. Interesting discussions on AI ethics, bias mitigation, and the importance of diverse training data. Met several researchers working on interpretable AI.',
      concepts: ['AI', 'collaboration', 'ethics', 'bias', 'interpretability'],
      keywords: ['AI', 'summit', 'collaboration', 'ethics', 'bias', 'mitigation', 'interpretable', 'researchers']
    }
  ];

  // Generate more realistic demo embeddings and positions
  nodeData.forEach((data, index) => {
    const embedding = generateMockEmbedding(data.concepts, data.tags);
    
    // Create more clustered positioning for demo - group related topics
    let baseX = 0, baseY = 0, baseZ = 0;
    
    // Cluster technical/AI topics
    if (data.tags.some(tag => ['technical', 'AI', 'machine_learning', 'frontend'].includes(tag))) {
      baseX = -15; baseY = 5; baseZ = -10;
    }
    // Cluster personal development topics
    else if (data.tags.some(tag => ['learning', 'book', 'psychology', 'course'].includes(tag))) {
      baseX = 10; baseY = 8; baseZ = 5;
    }
    // Cluster lifestyle topics
    else if (data.tags.some(tag => ['health', 'cooking', 'music', 'photography'].includes(tag))) {
      baseX = 5; baseY = -8; baseZ = 12;
    }
    // Cluster professional topics
    else if (data.tags.some(tag => ['work', 'project', 'investment', 'networking'].includes(tag))) {
      baseX = -8; baseY = -5; baseZ = -8;
    }
    // Default cluster for misc topics
    else {
      baseX = 0; baseY = 15; baseZ = 0;
    }
    
    const position_3d: [number, number, number] = [
      baseX + (Math.random() - 0.5) * 8,
      baseY + (Math.random() - 0.5) * 6,
      baseZ + (Math.random() - 0.5) * 8,
    ];
    
    const node: MemoryNode = {
      ...data,
      connections: {},
      embedding,
      position_3d,
      created_at: Date.now() - (nodeData.length - index) * 86400000, // Spread over 12 days
      last_accessed: Date.now() - Math.random() * 7200000, // Last 2 hours
      access_count: Math.floor(Math.random() * 25) + 5
    };
    demoNodes[node.id] = node;
  });

  // Create meaningful connections between related memories
  const connections = [
    // Technical cluster connections
    { from: 'demo_2', to: 'demo_12', weight: 0.89 }, // AI research ↔ AI conference
    { from: 'demo_3', to: 'demo_2', weight: 0.76 }, // RL course ↔ AI research
    { from: 'demo_4', to: 'demo_1', weight: 0.72 }, // React project ↔ work goals
    
    // Learning and development
    { from: 'demo_3', to: 'demo_5', weight: 0.68 }, // RL course ↔ psychology book
    { from: 'demo_5', to: 'demo_1', weight: 0.74 }, // Psychology ↔ work decisions
    { from: 'demo_12', to: 'demo_5', weight: 0.71 }, // AI conference ↔ decision making
    
    // Lifestyle and hobbies
    { from: 'demo_6', to: 'demo_10', weight: 0.63 }, // Exercise ↔ guitar (discipline)
    { from: 'demo_7', to: 'demo_8', weight: 0.82 }, // Cooking ↔ Japan travel (food culture)
    { from: 'demo_10', to: 'demo_11', weight: 0.69 }, // Guitar ↔ photography (creative hobbies)
    
    // Professional development
    { from: 'demo_1', to: 'demo_9', weight: 0.67 }, // Work goals ↔ investing
    { from: 'demo_12', to: 'demo_4', weight: 0.73 }, // AI conference ↔ tech project
    
    // Cross-domain interesting connections
    { from: 'demo_5', to: 'demo_11', weight: 0.58 }, // Psychology ↔ photography (perception)
    { from: 'demo_6', to: 'demo_5', weight: 0.71 }, // Exercise ↔ psychology (mental health)
    { from: 'demo_2', to: 'demo_4', weight: 0.79 }, // AI research ↔ React project
    { from: 'demo_8', to: 'demo_11', weight: 0.65 }, // Travel ↔ photography
  ];

  // Add bidirectional connections
  connections.forEach(({ from, to, weight }) => {
    const variance = 0.05; // Small variance for realistic asymmetry
    const outbound = weight;
    const inbound = weight + (Math.random() - 0.5) * variance;
    
    demoNodes[from].connections[to] = {
      outbound_weight: Math.max(0.1, Math.min(1, outbound)),
      inbound_weight: Math.max(0.1, Math.min(1, inbound)),
      last_accessed: Date.now() - Math.random() * 3600000, // Last hour
      creation_date: Date.now() - Math.random() * 86400000 * 7 // Last week
    };
    
    demoNodes[to].connections[from] = {
      outbound_weight: Math.max(0.1, Math.min(1, inbound)),
      inbound_weight: Math.max(0.1, Math.min(1, outbound)),
      last_accessed: Date.now() - Math.random() * 3600000,
      creation_date: Date.now() - Math.random() * 86400000 * 7
    };
  });

  return demoNodes;
};

export const useMemoryStore = create<MemoryStore>((set, get) => ({
  nodes: {},
  selectedNode: null,
  activeConnections: [],
  recentAccesses: [],
  isThinking: false,
  loading: false,
  error: null,
  isUsingBackend: false,
  initialized: false,

  addNode: (node) =>
    set((state) => ({
      nodes: { ...state.nodes, [node.id]: node }
    })),

  selectNode: (nodeId) =>
    set({ selectedNode: nodeId }),

  addMemoryAccess: (access) =>
    set((state) => ({
      recentAccesses: [access, ...state.recentAccesses.slice(0, 99)] // Keep last 100 accesses
    })),

  setThinking: (thinking) =>
    set({ isThinking: thinking }),

  updateConnectionWeight: (nodeId, targetId, weightChange) =>
    set((state) => {
      const node = state.nodes[nodeId];
      if (!node || !node.connections[targetId]) return state;

      const updatedNode = {
        ...node,
        connections: {
          ...node.connections,
          [targetId]: {
            ...node.connections[targetId],
            outbound_weight: Math.max(0, Math.min(1, node.connections[targetId].outbound_weight + weightChange)),
            last_accessed: Date.now()
          }
        }
      };

      return {
        nodes: { ...state.nodes, [nodeId]: updatedNode }
      };
    }),

  setActiveConnections: (connectionIds) =>
    set({ activeConnections: connectionIds }),

  generateMockData: () =>
    set({ nodes: generateMockNodes(), initialized: true }),
    
  generateDemoData: () =>
    set({ nodes: generateDemoNodes(), initialized: true }),

  simulateMemoryAccess: (nodeId, accessType) => {
    const { addMemoryAccess, nodes } = get();
    const node = nodes[nodeId];
    if (!node) return;

    const access: MemoryAccessEvent = {
      node_id: nodeId,
      access_type: accessType,
      timestamp: Date.now()
    };

    addMemoryAccess(access);

    // Update node access count and timestamp
    set((state) => ({
      nodes: {
        ...state.nodes,
        [nodeId]: {
          ...node,
          last_accessed: Date.now(),
          access_count: node.access_count + 1
        }
      }
    }));
  },

  clearRecentAccesses: () =>
    set({ recentAccesses: [] }),

  // Backend integration methods
  setLoading: (loading) => set({ loading }),
  
  setError: (error) => set({ error }),

  loadMemoryNodes: async () => {
    const { setLoading, setError, isUsingBackend } = get();
    
    // Prevent multiple simultaneous loads
    if (get().loading) {
      console.log('🔄 Memory nodes already loading, skipping...');
      return false;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      console.log('🔄 Loading memory nodes from backend...');
      
      const response = await apiService.getMemoryNodes();
      
      if (response.success && response.data) {
        const nodeCount = Object.keys(response.data).length;
        console.log('✅ Successfully loaded memory nodes:', nodeCount);
        
        // Only proceed if we actually have nodes
        if (nodeCount > 0) {
          // Convert backend response to frontend format
          const backendNodes = response.data;
          const frontendNodes: Record<string, MemoryNode> = {};
          let totalConnections = 0;
          
          Object.entries(backendNodes).forEach(([nodeId, backendNode]: [string, any]) => {
            const connectionCount = Object.keys(backendNode.connections || {}).length;
            totalConnections += connectionCount;
            
            // Map backend format to frontend format
            frontendNodes[nodeId] = {
              id: backendNode.id,
              tags: backendNode.tags || [],
              summary: backendNode.summary || '',
              content: backendNode.content || '',
              concepts: backendNode.keywords || [], // Use keywords as concepts fallback
              keywords: backendNode.keywords || [],
              connections: convertBackendConnections(backendNode.connections || {}),
              embedding: backendNode.embedding || [], // May not be present
              position_3d: backendNode.position_3d || [0, 0, 0],
              created_at: Date.now(), // Use current time as fallback
              last_accessed: Date.now(), // Use current time as fallback  
              access_count: backendNode.access_count || 0
            };
            
          });
          
          set({ 
            nodes: frontendNodes, 
            isUsingBackend: true,
            loading: false,
            error: null,
            initialized: true
          });
          
          return true;
        } else {
          console.warn('⚠️ Backend returned 0 nodes');
          set({ loading: false, initialized: true }); // Mark as initialized even with 0 nodes
          return false;
        }
      } else {
        console.warn('⚠️ Backend response unsuccessful, falling back to mock data');
        set({ error: 'Failed to load from backend', loading: false, initialized: true });
        return false;
      }
    } catch (error) {
      console.error('❌ Error loading memory nodes from backend:', error);
      set({ 
        error: error instanceof Error ? error.message : 'Unknown error', 
        loading: false, 
        initialized: true 
      });
      return false;
    }
  },

  refreshMemoryGraph: async () => {
    const { loadMemoryNodes } = get();
    return await loadMemoryNodes();
  }
}));

// Helper function to convert backend connection format to frontend format
function convertBackendConnections(backendConnections: any): Record<string, any> {
  const frontendConnections: Record<string, any> = {};
  
  Object.entries(backendConnections).forEach(([targetId, connection]: [string, any]) => {
    const weight = connection.weight ?? 0;
    
    frontendConnections[targetId] = {
      outbound_weight: weight,
      inbound_weight: weight, // Backend might not have separate inbound weight
      last_accessed: Date.now(),
      creation_date: Date.now()
    };
  });
  
  return frontendConnections;
}