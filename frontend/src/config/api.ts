// API Configuration - simplified for local development
export const API_CONFIG = {
  // Backend runs locally, frontend uses mock data
  BASE_URL: 'http://localhost:8000',
  TIMEOUT: 30000,
};

// API Endpoints
export const ENDPOINTS = {
  // Memory Management
  MEMORY: {
    NODES: '/api/memory/nodes',
    NODE_BY_ID: (id: string) => `/api/memory/nodes/${id}`,
    CONNECTIONS: '/api/memory/connections',
    SEARCH: '/api/memory/search',
    ACCESS: '/api/memory/access',
  },
  
  // Chat
  CHAT: {
    SEND_MESSAGE: '/api/chat/send',
    GET_MESSAGES: '/api/chat/messages',
    STREAM: '/api/chat/stream',
  },
  
  // Agents
  AGENTS: {
    FILTER: '/api/agents/filter',
    RELEVANCE: '/api/agents/relevance',
    CONNECTION: '/api/agents/connection',
  },
  
  // Analytics
  ANALYTICS: {
    MEMORY_STATS: '/api/analytics/memory-stats',
    ACCESS_PATTERNS: '/api/analytics/access-patterns',
    CONNECTION_STRENGTH: '/api/analytics/connection-strength',
  },
};

// Headers
export const getHeaders = () => ({
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  // Add authentication headers here when needed
  // 'Authorization': `Bearer ${getAuthToken()}`,
});

// Environment setup instructions
export const SETUP_INSTRUCTIONS = `
💻 Local Development Setup:

1. Start your backend server:
   cd /home/max/mem-manager
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

2. Start Expo development server:
   cd frontend
   npm start

3. Choose your platform:
   - Press 'w' for web browser
   - Press 'a' for Android emulator  
   - Press 'i' for iOS simulator

Uses mock data for now - perfect for UI development! 🎉
`;

export default API_CONFIG;