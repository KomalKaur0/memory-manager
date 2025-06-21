import { create } from 'zustand';
import { MemoryGraphState, MemoryNode, MemoryAccessEvent } from '../types/memory';

interface MemoryStore extends MemoryGraphState {
  // Actions
  addNode: (node: MemoryNode) => void;
  selectNode: (nodeId: string | null) => void;
  addMemoryAccess: (access: MemoryAccessEvent) => void;
  setThinking: (thinking: boolean) => void;
  updateConnectionWeight: (nodeId: string, targetId: string, weightChange: number) => void;
  setActiveConnections: (connectionIds: string[]) => void;
  
  // Mock data methods
  generateMockData: () => void;
  simulateMemoryAccess: (nodeId: string, accessType: MemoryAccessEvent['access_type']) => void;
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

export const useMemoryStore = create<MemoryStore>((set, get) => ({
  nodes: {},
  selectedNode: null,
  activeConnections: [],
  recentAccesses: [],
  isThinking: false,

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
    set({ nodes: generateMockNodes() }),

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
  }
}));