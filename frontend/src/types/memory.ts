export interface MemoryConnection {
  outbound_weight: number;
  inbound_weight: number;
  last_accessed: number;
  creation_date: number;
}

export interface MemoryNode {
  id: string;
  tags: string[];
  summary: string;
  content: string;
  concepts: string[];
  keywords: string[];
  connections: Record<string, MemoryConnection>;
  embedding?: number[];
  position_3d: [number, number, number]; // 3D coordinates for visualization
  created_at: number;
  last_accessed: number;
  access_count: number;
}

export interface MemoryAccessEvent {
  node_id: string;
  access_type: 'read' | 'write' | 'strengthen' | 'traverse';
  timestamp: number;
  connection_id?: string;
  weight_change?: number;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: number;
  memory_accesses?: MemoryAccessEvent[];
  thinking?: boolean;
}

export interface MemoryGraphState {
  nodes: Record<string, MemoryNode>;
  selectedNode: string | null;
  activeConnections: string[];
  recentAccesses: MemoryAccessEvent[];
  isThinking: boolean;
}

export interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  currentInput: string;
  memoryVisualizationVisible: boolean;
}