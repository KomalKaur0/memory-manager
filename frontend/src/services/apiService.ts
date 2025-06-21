import { API_CONFIG, ENDPOINTS, getHeaders } from '../config/api';
import { MemoryNode, ChatMessage, MemoryAccessEvent } from '../types/memory';

// Generic API response type
interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

// Simple API Service class
class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BASE_URL;
  }

  // Generic fetch method
  private async fetchApi<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          ...getHeaders(),
          ...options.headers,
        },
        timeout: API_CONFIG.TIMEOUT,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        data,
        success: true,
      };
    } catch (error) {
      console.error('API Error:', error);
      return {
        data: null as T,
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Memory Node Operations
  async getMemoryNodes(): Promise<ApiResponse<Record<string, MemoryNode>>> {
    return this.fetchApi<Record<string, MemoryNode>>(ENDPOINTS.MEMORY.NODES);
  }

  async getMemoryNode(nodeId: string): Promise<ApiResponse<MemoryNode>> {
    return this.fetchApi<MemoryNode>(ENDPOINTS.MEMORY.NODE_BY_ID(nodeId));
  }

  async searchMemory(query: string, limit: number = 10): Promise<ApiResponse<MemoryNode[]>> {
    return this.fetchApi<MemoryNode[]>(
      `${ENDPOINTS.MEMORY.SEARCH}?query=${encodeURIComponent(query)}&limit=${limit}`
    );
  }

  async recordMemoryAccess(access: MemoryAccessEvent): Promise<ApiResponse<void>> {
    return this.fetchApi<void>(ENDPOINTS.MEMORY.ACCESS, {
      method: 'POST',
      body: JSON.stringify(access),
    });
  }

  // Chat Operations
  async sendMessage(content: string): Promise<ApiResponse<ChatMessage>> {
    return this.fetchApi<ChatMessage>(ENDPOINTS.CHAT.SEND_MESSAGE, {
      method: 'POST',
      body: JSON.stringify({
        content,
        timestamp: Date.now(),
      }),
    });
  }

  async getChatMessages(limit: number = 50): Promise<ApiResponse<ChatMessage[]>> {
    return this.fetchApi<ChatMessage[]>(
      `${ENDPOINTS.CHAT.GET_MESSAGES}?limit=${limit}`
    );
  }

  // Stream chat response with memory access events
  async *streamChatResponse(content: string): AsyncGenerator<{
    type: 'message' | 'memory_access' | 'thinking' | 'complete';
    data: any;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}${ENDPOINTS.CHAT.STREAM}`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ content }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data;
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error);
      yield {
        type: 'complete',
        data: { error: error instanceof Error ? error.message : 'Stream error' },
      };
    }
  }

  // Analytics Operations
  async getMemoryStats(): Promise<ApiResponse<{
    total_nodes: number;
    total_connections: number;
    avg_connection_weight: number;
    most_accessed_nodes: MemoryNode[];
  }>> {
    return this.fetchApi(ENDPOINTS.ANALYTICS.MEMORY_STATS);
  }

  async getAccessPatterns(timeRange: string = '24h'): Promise<ApiResponse<MemoryAccessEvent[]>> {
    return this.fetchApi(
      `${ENDPOINTS.ANALYTICS.ACCESS_PATTERNS}?range=${timeRange}`
    );
  }

  // Connection Health Check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        timeout: 5000,
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  // Update base URL (useful for switching between ngrok tunnels)
  updateBaseUrl(newUrl: string) {
    this.baseUrl = newUrl;
  }
}

// Export singleton instance
export const apiService = new ApiService();