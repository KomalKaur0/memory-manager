// API Configuration - use environment variables for backend URL
const getBackendUrl = () => {
  // Check environment variables (Expo uses EXPO_PUBLIC_ prefix)
  const envUrl = process.env.EXPO_PUBLIC_API_BASE_URL;
  const localUrl = process.env.EXPO_PUBLIC_LOCAL_URL;
  
  console.log('🔧 Environment URLs:', { envUrl, localUrl });
  
  // Prefer environment URL, fallback to localhost
  return envUrl || localUrl || 'http://localhost:8000';
};

const getFallbackUrls = () => {
  // Get fallback URLs from environment
  const fallbackEnv = process.env.EXPO_PUBLIC_FALLBACK_URLS;
  
  if (fallbackEnv) {
    return fallbackEnv.split(',').map(url => url.trim());
  }
  
  // Default fallbacks
  return [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://192.168.156.157:8000',
  ];
};

export const API_CONFIG = {
  BASE_URL: getBackendUrl(),
  TIMEOUT: 30000,
  FALLBACK_URLS: getFallbackUrls(),
  DEBUG_MODE: process.env.EXPO_PUBLIC_DEBUG_MODE === 'true',
  AUTO_CONNECT: process.env.EXPO_PUBLIC_AUTO_CONNECT === 'true',
};

console.log('🌐 API Configuration:', {
  baseUrl: API_CONFIG.BASE_URL,
  fallbackCount: API_CONFIG.FALLBACK_URLS.length,
  debugMode: API_CONFIG.DEBUG_MODE,
  autoConnect: API_CONFIG.AUTO_CONNECT,
});

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
    GET_MESSAGES: '/api/chat/history',
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